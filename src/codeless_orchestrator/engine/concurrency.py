from __future__ import annotations

import json
from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from ..providers.base import LLMProvider
from ..providers.types import ChatMessage, ChatRequest, ChatResponse, ToolCall
from ..tools.execution import execute_many
from ..tools.registry import ToolEntry
from ..runtime.structured_output import ensure_structured_output, build_system_hint
from ..runtime.filters import pre_provider_filter, post_provider_filter
from ..runtime.policies import get_policy


@dataclass(frozen=True)
class BranchSpec:
    agent_id: str
    label: str
    provider: LLMProvider
    provider_tool_specs: list[Any]
    tool_cfg: Any  # ToolsConfig
    name_to_entry: dict[str, ToolEntry]
    overrides_by_name: dict[str, dict[str, Any]]
    system_message: ChatMessage | None
    structured_output: Any | None
    safety: Any | None


@dataclass
class BranchResult:
    assistant: ChatMessage
    tool_messages: list[ChatMessage]
    finish_reason: str | None
    usage: Any | None


def _build_request(
    *,
    spec: BranchSpec,
    model_params: Any,
    base_messages: list[ChatMessage],
) -> ChatRequest:
    req_messages = (
        [spec.system_message] + base_messages if spec.system_message else list(base_messages)
    )
    return ChatRequest(
        model=model_params.model_id,
        messages=req_messages,
        tools=spec.provider_tool_specs or None,
        tool_choice=(
            "auto"
            if spec.tool_cfg.policy == "Auto" or spec.tool_cfg.policy == "Required"
            else "none"
        ),
        temperature=model_params.temperature,
        top_p=model_params.top_p,
        max_tokens=model_params.max_tokens,
        stop=model_params.stop or None,
        seed=model_params.seed,
        json_mode_enabled=model_params.json_mode_enabled,
        response_format_schema=None,
        metadata={},
        timeout=None,
    )


def _run_single_branch(
    *,
    spec: BranchSpec,
    model_params: Any,
    base_messages: list[ChatMessage],
) -> BranchResult:
    # One assistant turn for the branch, with an optional tool sub-loop for that turn
    # Safety pre-filter
    safety_cfg = getattr(spec, "safety", None)
    policy = get_policy(getattr(safety_cfg, "policy_ref", None)) if safety_cfg is not None else None
    pre = pre_provider_filter(base_messages, safety_cfg=safety_cfg, policy=policy) if safety_cfg is not None else None
    # Use redacted messages if available
    req = _build_request(
        spec=spec,
        model_params=model_params,
        base_messages=(pre.redacted_messages if pre is not None else base_messages),
    )
    so = getattr(spec, "structured_output", None)
    if pre is not None and pre.blocked:
        assistant = pre.refusal_message or ChatMessage(role="assistant", content="")
        if safety_cfg is not None:
            assistant = post_provider_filter(assistant, safety_cfg=safety_cfg, policy=policy)
        resp_usage = None
        resp_finish = "content_filter"
    elif so is not None and getattr(so, "enabled", False):
        hint = build_system_hint(getattr(so, "schema_", None) or None)
        assistant, usage, finish_reason, _attempts = ensure_structured_output(
            spec.provider,
            req,
            policy_cfg=so,
            post_cfg=getattr(so, "post_process", None),
            system_hint=hint,
        )
        if safety_cfg is not None:
            assistant = post_provider_filter(assistant, safety_cfg=safety_cfg, policy=policy)
        resp_usage = usage
        resp_finish = finish_reason
    else:
        resp: ChatResponse = spec.provider.chat(req)
        assistant = resp.message
        if safety_cfg is not None:
            assistant = post_provider_filter(assistant, safety_cfg=safety_cfg, policy=policy)
        resp_usage = resp.usage
        resp_finish = resp.finish_reason

    # Execute this turn's tool calls (if any) concurrently (bounded) and preserve order
    tool_messages: list[ChatMessage] = []
    tcs: list[ToolCall] = list(assistant.tool_calls or [])
    if tcs:
        # Limit number of calls in this turn
        max_calls = spec.tool_cfg.max_calls_per_turn or 0
        limit = len(tcs) if max_calls <= 0 else min(max_calls, len(tcs))
        timeout_s = (spec.tool_cfg.timeout_ms or 0) / 1000.0 if spec.tool_cfg.timeout_ms else None
        parallel = max(1, int(spec.tool_cfg.parallelism or 1))
        tool_messages = execute_many(
            tcs[:limit],
            name_to_entry=spec.name_to_entry,
            overrides_by_name=spec.overrides_by_name,
            timeout_s=timeout_s,
            max_workers=parallel,
        )

    return BranchResult(
        assistant=assistant,
        tool_messages=tool_messages,
        finish_reason=resp_finish,
        usage=resp_usage,
    )


def run_branches(
    specs: Iterable[tuple[BranchSpec, Any, list[ChatMessage]]],
    *,
    max_workers: int,
) -> tuple[dict[str, BranchResult], list[str]]:
    """Run multiple branches concurrently.

    Args:
        specs: Iterable of (BranchSpec, model_params, base_messages) tuples.
        max_workers: Concurrency across branches.

    Returns:
        A mapping of agent_id -> BranchResult and a list of agent_ids in the order
        they completed (first finished first).
    """
    results: dict[str, BranchResult] = {}
    finished_order: list[str] = []

    specs_list = list(specs)
    if not specs_list:
        return results, finished_order

    n_workers = max(1, min(max_workers, len(specs_list)))
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        fut_to_id: dict[Future[BranchResult], str] = {}
        for spec, model_params, base_msgs in specs_list:
            fut = pool.submit(
                _run_single_branch,
                spec=spec,
                model_params=model_params,
                base_messages=list(base_msgs),
            )
            fut_to_id[fut] = spec.agent_id

        for fut in as_completed(fut_to_id):
            aid = fut_to_id[fut]
            try:
                res = fut.result()
            finally:
                finished_order.append(aid)
            results[aid] = res

    return results, finished_order


def merge_first_finish(
    results: dict[str, BranchResult],
    finished_order: list[str],
) -> tuple[ChatMessage, dict[str, Any]]:
    """Pick the first finished branch's assistant message."""
    if not results:
        return ChatMessage(role="assistant", content=""), {
            "strategy": "first_finish",
            "picked": None,
        }
    picked_id = finished_order[0] if finished_order else sorted(results.keys())[0]
    picked = results[picked_id].assistant
    discarded = [aid for aid in results.keys() if aid != picked_id]
    meta = {"strategy": "first_finish", "picked": picked_id, "discarded": discarded}
    return picked, meta


def merge_aggregate(results: dict[str, BranchResult]) -> tuple[ChatMessage, dict[str, Any]]:
    """Concatenate assistant outputs deterministically by agent id order."""
    parts: list[str] = []
    order = sorted(results.keys())
    for aid in order:
        msg = results[aid].assistant
        text = msg.content if isinstance(msg.content, str) else json.dumps(msg.content)
        parts.append(f"{aid}: {text}")
    merged_text = "\n".join(parts)
    return ChatMessage(role="assistant", content=merged_text), {
        "strategy": "aggregate",
        "order": order,
    }


def merge_vote_length(results: dict[str, BranchResult]) -> tuple[ChatMessage, dict[str, Any]]:
    """Pick the longest assistant text; ties broken by agent id order."""
    if not results:
        return ChatMessage(role="assistant", content=""), {
            "strategy": "vote",
            "picked": None,
            "lengths": {},
        }
    lengths: dict[str, int] = {}
    for aid, br in results.items():
        text = (
            br.assistant.content
            if isinstance(br.assistant.content, str)
            else json.dumps(br.assistant.content)
        )
        lengths[aid] = len(text or "")
    # Stable by (length desc, agent id asc)
    picked_id = sorted(lengths.keys(), key=lambda x: (-lengths[x], x))[0]
    picked = results[picked_id].assistant
    meta = {"strategy": "vote", "picked": picked_id, "lengths": lengths}
    return picked, meta


__all__ = [
    "BranchSpec",
    "BranchResult",
    "run_branches",
    "merge_first_finish",
    "merge_aggregate",
    "merge_vote_length",
]
