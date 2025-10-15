from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

from langgraph.graph import StateGraph

from ....logging import get_logger
from ....providers.types import ChatMessage
from ...errors import GraphCompileError
from ..builder import BuildChildFn, RegionBuildResult
from ..context import AgentArtifacts, CompileContext
from ..types import ConcurrentRegion
from ...state import EngineState


_log = get_logger(__name__)

try:  # pragma: no cover - legacy compatibility shim
    from ...graph_builders import concurrent as _legacy_concurrent_module
except Exception:  # pragma: no cover - legacy compatibility shim
    _legacy_concurrent_module = None


def _select_merge_strategy(region: ConcurrentRegion) -> str:
    """Choose a merge strategy in parity with the legacy builder."""

    if _legacy_concurrent_module is not None and hasattr(
        _legacy_concurrent_module, "_select_merge_strategy"
    ):  # pragma: no cover - delegated for compatibility
        try:
            next_agent_id = None
            if region.join is not None and getattr(region.join, "agents", []):
                next_agent_id = region.join.agents[0]
            choice = _legacy_concurrent_module._select_merge_strategy(next_agent_id)
            if choice:
                return choice
        except Exception:
            pass

    if region.strategy:
        return region.strategy
    return "first_finish" if region.join is not None else "aggregate"


@dataclass(slots=True)
class _BranchApp:
    branch_id: str
    label: str | None
    result: RegionBuildResult
    artifacts: AgentArtifacts


def _clone_state(state: EngineState) -> EngineState:
    cloned: EngineState = dict(state)
    cloned["messages"] = list(state.get("messages", []))
    cloned["metadata"] = dict((state.get("metadata", {}) or {}))
    cloned["tool_calls"] = None
    cloned["agent_cursor"] = None
    # concurrent/group buckets should not leak across branches
    cloned["concurrent"] = None
    cloned["group"] = None
    return cloned


def _extract_branch_result(
    *,
    branch_id: str,
    branch_state: EngineState,
    baseline: int,
    duration_ms: float,
    error: str | None,
) -> dict[str, Any]:
    messages = list(branch_state.get("messages", []))
    appended = messages[baseline:]
    assistant: ChatMessage | None = None
    for msg in reversed(appended):
        if getattr(msg, "role", None) == "assistant":
            assistant = msg
            break
    tool_messages = [msg for msg in appended if getattr(msg, "role", None) == "tool"]
    meta = branch_state.get("metadata", {}) or {}
    usage = meta.get("_last_usage")
    finish_reason = meta.get("_last_finish_reason")
    return {
        "assistant": assistant,
        "tool_messages": tool_messages,
        "usage": usage,
        "finish_reason": finish_reason,
        "error": error,
        "duration_ms": duration_ms,
        "appended": appended,
    }


def _effective_parallelism(ctx: CompileContext, branches: list[_BranchApp]) -> int:
    par_values: list[int] = []
    for branch in branches:
        try:
            par_values.append(int(getattr(branch.artifacts.tool_cfg, "parallelism", 1) or 1))
        except Exception:
            continue
    if not par_values:
        return max(1, len(branches))
    return max(1, min(len(branches), min(par_values)))


def _merge_results(
    *,
    region: ConcurrentRegion,
    bucket: dict[str, Any],
) -> tuple[ChatMessage, list[ChatMessage], dict[str, Any]]:
    results = bucket.get("results", [])
    order = bucket.get("order", [])
    strategy = bucket.get("strategy") or _select_merge_strategy(region)
    if not results:
        return ChatMessage(role="assistant", content=""), [], {"strategy": strategy, "picked": None}
    results_by_id = {item["branch_id"]: item for item in results}

    def _message_text(res: dict[str, Any]) -> str:
        assistant = res.get("assistant")
        if isinstance(assistant, ChatMessage) and isinstance(assistant.content, str):
            return assistant.content
        if res.get("error"):
            return f"[error] {res['error']}"
        return ""

    picked_id: str | None = None
    if strategy == "first_finish":
        for bid in order:
            res = results_by_id.get(bid)
            if res and res.get("error") is None and res.get("assistant") is not None:
                picked_id = bid
                break
        if picked_id is None:
            picked_id = order[0] if order else results[0]["branch_id"]
        chosen = results_by_id.get(picked_id, results[0])
        assistant = chosen.get("assistant") or ChatMessage(
            role="assistant",
            content=_message_text(chosen),
        )
        return assistant, list(chosen.get("tool_messages", [])), {"strategy": strategy, "picked": picked_id}

    if strategy == "vote":
        scored = sorted(
            results,
            key=lambda item: (-len(_message_text(item)), item["branch_id"]),
        )
        chosen = scored[0]
        assistant = chosen.get("assistant") or ChatMessage(
            role="assistant",
            content=_message_text(chosen),
        )
        picked_id = chosen["branch_id"]
        return assistant, list(chosen.get("tool_messages", [])), {"strategy": strategy, "picked": picked_id}

    # aggregate (default)
    ordered = sorted(results, key=lambda item: item["branch_id"])
    parts: list[str] = []
    for item in ordered:
        label = item.get("label") or item["branch_id"]
        parts.append(f"{label}: {_message_text(item)}")
    content = "\n\n".join(parts)
    tool_msgs = [tm for item in ordered for tm in item.get("tool_messages", [])]
    return ChatMessage(role="assistant", content=content), tool_msgs, {"strategy": strategy, "picked": None}


def build_concurrent(
    *,
    region: ConcurrentRegion,
    graph: StateGraph,
    ctx: CompileContext,
    build_child: BuildChildFn,
) -> RegionBuildResult:
    entry_key = ctx.region_entry_key(region.id)
    exit_key = ctx.region_exit_key(region.id)
    ctx.ensure_passthrough_node(graph, entry_key)
    ctx.ensure_passthrough_node(graph, exit_key)

    branch_apps: list[_BranchApp] = []
    for branch in region.branches:
        child_res = build_child(branch.region, False)
        artifacts = ctx.agent_artifacts(branch.entry_agent)
        branch_apps.append(
            _BranchApp(
                branch_id=branch.entry_agent,
                label=branch.label,
                result=child_res,
                artifacts=artifacts,
            )
        )

    join_res: RegionBuildResult | None = None
    if region.join is not None:
        join_res = build_child(region.join, True)

    fanout_key = f"fanout__{region.id}"
    join_key = f"join__{region.id}"
    graph.add_node(fanout_key, _fanout_factory(region, branch_apps, ctx))
    graph.add_node(join_key, _join_factory(region))

    graph.add_edge(entry_key, fanout_key)
    graph.add_edge(fanout_key, join_key)

    if join_res is not None:
        graph.add_edge(join_key, join_res.entry_key)
        graph.add_edge(join_res.exit_key, exit_key)
    else:
        graph.add_edge(join_key, exit_key)

    notes = list(region.notes)
    for branch in branch_apps:
        notes.extend(branch.result.notes)
    if join_res is not None:
        notes.extend(join_res.notes)

    return RegionBuildResult(region=region, entry_key=entry_key, exit_key=exit_key, notes=notes)


def _fanout_factory(
    region: ConcurrentRegion,
    branch_apps: list[_BranchApp],
    ctx: CompileContext,
) -> Callable[[EngineState], EngineState]:
    max_workers = _effective_parallelism(ctx, branch_apps)
    strategy = _select_merge_strategy(region)

    def _fanout(state: EngineState) -> EngineState:
        bucket = state.get("concurrent")
        if bucket is None or not isinstance(bucket, dict):
            bucket = {}
        state["concurrent"] = bucket

        results: list[dict[str, Any]] = []
        finished_order: list[str] = []

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {}
            for branch in branch_apps:
                branch_state = _clone_state(state)
                baseline = len(branch_state.get("messages", []))

                def _task(
                    artifacts=branch.artifacts,
                    b_state=branch_state,
                    base=baseline,
                    bid=branch.branch_id,
                ):
                    start = perf_counter()
                    err: str | None = None
                    try:
                        final_state = artifacts.agent_fn(b_state)
                        if final_state.get("tool_calls"):
                            final_state = artifacts.tools_fn(final_state)
                    except Exception as exc:  # pragma: no cover - defensive path
                        _log.warning("Concurrent branch '%s' failed: %s", bid, exc)
                        err = str(exc)
                        final_state = b_state
                    duration_ms = (perf_counter() - start) * 1000.0
                    return _extract_branch_result(
                        branch_id=bid,
                        branch_state=final_state,
                        baseline=base,
                        duration_ms=duration_ms,
                        error=err,
                    )

                futures[pool.submit(_task)] = branch

            for fut in as_completed(futures):
                branch = futures[fut]
                result = fut.result()
                result["branch_id"] = branch.branch_id
                result["label"] = branch.label
                results.append(result)
                finished_order.append(branch.branch_id)

        bucket[region.id] = {
            "results": results,
            "order": finished_order,
            "strategy": strategy,
        }
        state["concurrent"] = bucket
        return state

    return _fanout


def _join_factory(region: ConcurrentRegion) -> Callable[[EngineState], EngineState]:
    def _join(state: EngineState) -> EngineState:
        bucket = state.get("concurrent") or {}
        current = bucket.get(region.id)
        if not current:
            return state

        assistant, tool_messages, merge_info = _merge_results(region=region, bucket=current)

        messages = state.setdefault("messages", [])
        messages.extend(tool_messages)
        messages.append(assistant)

        meta = state.setdefault("metadata", {})
        info = {
            "strategy": merge_info.get("strategy"),
            "branches": [
                {
                    "id": res.get("branch_id"),
                    "label": res.get("label"),
                    "finish_reason": res.get("finish_reason"),
                    "error": res.get("error"),
                    "duration_ms": res.get("duration_ms"),
                }
                for res in current.get("results", [])
            ],
        }
        if merge_info.get("picked") is not None:
            info["picked"] = merge_info.get("picked")
        conc_meta = dict(meta.get("concurrent", {}))
        conc_meta[region.id] = info
        meta["concurrent"] = conc_meta
        picked_id = merge_info.get("picked")
        if picked_id is not None:
            for branch in info["branches"]:
                if branch.get("id") == picked_id:
                    meta["_last_finish_reason"] = branch.get("finish_reason")
                    break
        elif info["branches"]:
            meta["_last_finish_reason"] = info["branches"][0].get("finish_reason")
        state["metadata"] = meta
        bucket.pop(region.id, None)
        state["concurrent"] = bucket
        return state

    return _join


__all__ = ["build_concurrent"]
