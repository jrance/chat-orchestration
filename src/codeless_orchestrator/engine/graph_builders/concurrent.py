from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from ...config.schema import AgentNode, GraphConfig, ModelParams, ToolNode, ToolsConfig
from ...providers.base import LLMProvider
from ...providers.types import ChatMessage, ChatRequest
from ...tools.execution import execute_tool_call
from ...tools.registry import ToolEntry, ToolRegistry
from ..concurrency import (
    BranchResult,
    BranchSpec,
    merge_aggregate,
    merge_first_finish,
    merge_vote_length,
    run_branches,
)
from ..state import EngineState
from ...runtime.history import assemble_prompt
from ...runtime.filters import pre_provider_filter, post_provider_filter
from ...runtime.policies import get_policy
from ...runtime.safety import redact_text


def _build_system_message(agent: AgentNode) -> ChatMessage | None:
    sys = agent.data.system_instructions or ""
    guide = agent.data.style_guide or ""
    parts: list[str] = []
    if sys:
        parts.append(sys)
    if guide:
        parts.append(guide)
    if not parts:
        return None
    return ChatMessage(role="system", content="\n\n".join(parts))


@dataclass
class _AgentRuntime:
    node: AgentNode
    provider: LLMProvider
    tool_cfg: ToolsConfig
    name_to_entry: dict[str, ToolEntry]
    provider_tool_specs: list[Any]
    overrides_by_name: dict[str, dict[str, Any]]
    system_message: ChatMessage | None

    def agent_fn(self, state: EngineState) -> EngineState:
        state["agent_cursor"] = self.node.id
        mp = self.node.data.model
        meta = state.get("metadata", {}) or {}
        preamble = (
            meta.get("org_preamble") if self.node.data.context.inject_org_preamble else None
        )
        vars_req = meta.get("vars") or {}
        req_messages = assemble_prompt(self.node, state.get("messages", []), preamble, vars_req)
        safety_cfg = self.node.data.safety
        policy = get_policy(getattr(safety_cfg, "policy_ref", None))
        pre = pre_provider_filter(req_messages, safety_cfg=safety_cfg, policy=policy)
        if pre.blocked:
            assistant = post_provider_filter(pre.refusal_message or ChatMessage(role="assistant", content=""), safety_cfg=safety_cfg, policy=policy)
            state["messages"].append(assistant)
            state["tool_calls"] = None
            state["steps"] = int(state.get("steps", 0)) + 1
            meta2 = state.get("metadata", {})
            meta2["_last_finish_reason"] = "content_filter"
            if pre.warnings:
                meta2["warnings"] = pre.warnings
            state["metadata"] = meta2
            return state
        req = ChatRequest(
            model=mp.model_id,
            messages=pre.redacted_messages,
            tools=self.provider_tool_specs or None,
            tool_choice=(
                "auto"
                if self.tool_cfg.policy == "Auto" or self.tool_cfg.policy == "Required"
                else "none"
            ),
            temperature=mp.temperature,
            top_p=mp.top_p,
            max_tokens=mp.max_tokens,
            stop=mp.stop or None,
            seed=mp.seed,
            json_mode_enabled=mp.json_mode_enabled,
            response_format_schema=None,
            metadata={
                **(state.get("metadata", {}) or {}),
                "agent_id": self.node.id,
                "agent_label": self.node.label,
            },
            timeout=None,
        )
        resp = self.provider.chat(req)
        assistant = post_provider_filter(resp.message, safety_cfg=safety_cfg, policy=policy)
        state["messages"].append(assistant)
        state["tool_calls"] = assistant.tool_calls or None
        state["steps"] = int(state.get("steps", 0)) + 1
        meta = state.get("metadata", {})
        meta["_last_finish_reason"] = resp.finish_reason
        meta["_last_usage"] = resp.usage
        state["metadata"] = meta
        return state

    def tools_fn(self, state: EngineState) -> EngineState:
        tool_calls = list(state.get("tool_calls") or [])
        if not tool_calls:
            return state

        max_calls = self.tool_cfg.max_calls_per_turn or 0
        limit = len(tool_calls) if max_calls <= 0 else min(max_calls, len(tool_calls))
        timeout_s = (self.tool_cfg.timeout_ms or 0) / 1000.0 if self.tool_cfg.timeout_ms else None

        for tc in tool_calls[:limit]:
            tool_name = tc.name
            if tool_name not in self.name_to_entry:
                # Mirror sequential's behavior; exceptions caught by graph
                raise Exception(f"Tool not attached: {tool_name}")
            entry = self.name_to_entry[tool_name]
            overrides = self.overrides_by_name.get(tool_name) or {}
            result_obj = execute_tool_call(
                entry.impl,
                tc.arguments_json or "",
                parameter_overrides=overrides,
                timeout_s=timeout_s,
                redact_for_logging=bool(getattr(self.node.data.safety, "pii_redaction", True)),
            )
            content_json = json.dumps(result_obj)
            if getattr(self.node.data.safety, "pii_redaction", True):
                content_json = redact_text(content_json)
            state["messages"].append(
                ChatMessage(role="tool", content=content_json, tool_call_id=tc.id)
            )

        state["tool_calls"] = None
        return state


def _runtime_for_agent(
    *,
    node: AgentNode,
    provider: LLMProvider,
    registry: ToolRegistry,
    overrides_by_name: dict[str, dict[str, Any]],
) -> _AgentRuntime:
    tool_cfg = node.data.tools
    attached_ids = list(tool_cfg.attached or [])
    provider_tool_specs = registry.to_provider_tool_specs(attached_ids) if attached_ids else []
    name_to_entry: dict[str, ToolEntry] = {}
    for tool_id in attached_ids:
        entry = registry.get(tool_id)
        name_to_entry[entry.impl.name] = entry
    # Defer system assembly per turn to include org preamble/vars and history window
    return _AgentRuntime(
        node=node,
        provider=provider,
        tool_cfg=tool_cfg,
        name_to_entry=name_to_entry,
        provider_tool_specs=provider_tool_specs,
        overrides_by_name=overrides_by_name,
        system_message=None,
    )


def _select_merge_strategy(next_agent_id: str | None) -> Literal["first_finish", "aggregate"]:
    # Heuristic: if there is an explicit next agent (join), prefer first-finish
    return "first_finish" if next_agent_id else "aggregate"


def build(
    cfg: GraphConfig,
    fanout_root: str,
    branches: list[str],
    next_agent_id: str | None,
    *,
    provider_resolver: Callable[[Any], LLMProvider],
    registry: ToolRegistry,
    merge_strategy: Literal["first_finish", "aggregate", "vote"] | None = None,
) -> tuple[Any, str, LLMProvider, dict[str, ToolEntry], ToolsConfig, ModelParams]:
    """Build a concurrent fan-out/fan-in LangGraph app.

    Returns a tuple of (app, root_agent, root_provider, root_tools_attached,
    root_tool_config, root_model_params).
    """

    agents_by_id: dict[str, AgentNode] = {n.id: n for n in cfg.nodes if isinstance(n, AgentNode)}

    # Build branch specs
    branch_specs: dict[str, BranchSpec] = {}
    branch_tool_parallelisms: list[int] = []
    for aid in branches:
        node = agents_by_id[aid]
        provider = provider_resolver(node.data.model)
        # Build overrides map for this agent
        overrides_by_name: dict[str, dict[str, Any]] = {}
        attached_ids = list(node.data.tools.attached or [])
        for n in cfg.nodes:
            if isinstance(n, ToolNode) and n.data.tool_id in attached_ids:
                overrides_by_name[n.data.name] = dict(n.data.parameter_overrides or {})
        # Registry and tool specs
        attached_tool_specs = registry.to_provider_tool_specs(attached_ids) if attached_ids else []
        name_to_entry: dict[str, ToolEntry] = {}
        for tool_id in attached_ids:
            entry = registry.get(tool_id)
            name_to_entry[entry.impl.name] = entry
        branch_specs[aid] = BranchSpec(
            agent_id=aid,
            label=node.label,
            provider=provider,
            provider_tool_specs=attached_tool_specs,
            tool_cfg=node.data.tools,
            name_to_entry=name_to_entry,
            overrides_by_name=overrides_by_name,
            system_message=_build_system_message(node),
            structured_output=node.data.structured_output,
            safety=node.data.safety,
        )
        branch_tool_parallelisms.append(int(node.data.tools.parallelism or 1))

    # Compute cross-branch concurrency throttle: min of branch tool parallelisms
    cross_parallelism = max(1, min(branch_tool_parallelisms) if branch_tool_parallelisms else 1)

    # Optionally build next-agent runtime
    next_runtime: _AgentRuntime | None = None
    if next_agent_id:
        nnode = agents_by_id[next_agent_id]
        provider = provider_resolver(nnode.data.model)
        # Build overrides for the next agent
        overrides_map: dict[str, dict[str, Any]] = {}
        attached_ids = list(nnode.data.tools.attached or [])
        for n in cfg.nodes:
            if isinstance(n, ToolNode) and n.data.tool_id in attached_ids:
                overrides_map[n.data.name] = dict(n.data.parameter_overrides or {})
        next_runtime = _runtime_for_agent(
            node=nnode,
            provider=provider,
            registry=registry,
            overrides_by_name=overrides_map,
        )

    graph = StateGraph(EngineState)

    group_id = f"{fanout_root}__fanout"

    def fanout_fn(state: EngineState) -> EngineState:
        # Initialize concurrent bucket
        bucket: dict[str, Any] = {
            "expected": list(branches),
            "finished": False,
            "finished_order": [],
            "assistants": {},
            "tool_messages": {},
        }
        state["concurrent"] = bucket
        # Prepare specs iterable with static model params and current messages snapshot
        base_messages = list(state.get("messages", []))
        meta = state.get("metadata", {}) or {}
        iterable = []
        for aid in branches:
            spec = branch_specs[aid]
            node = agents_by_id[aid]
            mp = node.data.model
            preamble = meta.get("org_preamble") if node.data.context.inject_org_preamble else None
            vars_req = meta.get("vars") or {}
            msgs_for_turn = assemble_prompt(node, base_messages, preamble, vars_req)
            # Pass messages already assembled; spec.system_message remains None
            iterable.append((spec, mp, msgs_for_turn))
        results_map, finished_order = run_branches(iterable, max_workers=cross_parallelism)
        # Persist minimal results for join step
        assistants: dict[str, ChatMessage] = {}
        tmsgs: dict[str, list[ChatMessage]] = {}
        # Append tool messages to transcript to mirror sequential semantics
        for aid, br in results_map.items():
            for tm in br.tool_messages:
                state["messages"].append(tm)
            assistants[aid] = br.assistant
            tmsgs[aid] = list(br.tool_messages)
        bucket["assistants"] = assistants
        bucket["tool_messages"] = tmsgs
        bucket["finished_order"] = list(finished_order)
        return state

    def join_fn(state: EngineState) -> EngineState:
        bucket_any = state.get("concurrent", {})
        bucket = bucket_any if isinstance(bucket_any, dict) else {}
        assistants = bucket.get("assistants", {}) or {}
        finished_order = list(bucket.get("finished_order", []) or [])
        # Adapt assistants map into BranchResult minimal objects
        res_objs: dict[str, BranchResult] = {
            aid: BranchResult(assistant=am, tool_messages=[], finish_reason=None, usage=None)
            for aid, am in assistants.items()
        }
        strategy = merge_strategy or _select_merge_strategy(next_agent_id)
        if strategy == "first_finish":
            merged, m = merge_first_finish(res_objs, finished_order)
        elif strategy == "vote":
            merged, m = merge_vote_length(res_objs)
        else:
            merged, m = merge_aggregate(res_objs)
        state["messages"].append(merged)
        meta = state.get("metadata", {})
        meta_conc = dict(meta.get("concurrent", {}))
        meta_conc.update(m)
        meta["concurrent"] = meta_conc
        state["metadata"] = meta
        bucket["finished"] = True
        return state

    # Nodes
    fanout_key = f"fanout__{group_id}"
    join_key = f"join__{group_id}"
    graph.add_node(fanout_key, fanout_fn)
    graph.add_node(join_key, join_fn)

    # Optional next agent nodes
    if next_runtime is not None:
        graph.add_node(f"agent__{next_agent_id}", next_runtime.agent_fn)
        graph.add_node(f"tools__{next_agent_id}", next_runtime.tools_fn)

    # Edges
    graph.add_edge(START, fanout_key)
    graph.add_edge(fanout_key, join_key)
    if next_runtime is not None:
        # After join, resume with the next agent turn
        def _cond_factory(my_tools_key: str) -> Callable[[EngineState], str]:
            def _should_continue(state: EngineState) -> str:
                tcs = state.get("tool_calls") or []
                if tcs:
                    return my_tools_key
                return END

            return _should_continue

        graph.add_conditional_edges(
            f"agent__{next_agent_id}", _cond_factory(f"tools__{next_agent_id}")
        )
        graph.add_edge(join_key, f"agent__{next_agent_id}")
        graph.add_edge(f"tools__{next_agent_id}", f"agent__{next_agent_id}")
    else:
        graph.add_edge(join_key, END)

    app = graph.compile()

    # Select a representative root for compiled metadata:
    # if there is a next agent, use it; else use first branch
    if next_agent_id is not None:
        root_id = next_agent_id
    else:
        root_id = branches[0]

    root_node = agents_by_id[root_id]
    root_provider = provider_resolver(root_node.data.model)

    # Prepare tool registry map for root agent for compiled artifact
    attached_ids = list(root_node.data.tools.attached or [])
    root_name_to_entry: dict[str, ToolEntry] = {}
    for tool_id in attached_ids:
        entry = registry.get(tool_id)
        root_name_to_entry[entry.impl.name] = entry

    return (
        app,
        root_id,
        root_provider,
        root_name_to_entry,
        root_node.data.tools,
        root_node.data.model,
        root_node.data.structured_output,
        root_node.data.safety,
    )


__all__ = ["build"]
