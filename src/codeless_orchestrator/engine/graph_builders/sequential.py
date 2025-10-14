from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from langgraph.graph import END, START, StateGraph

from ...config.schema import AgentNode, GraphConfig, ModelParams, ToolNode, ToolsConfig
from ...providers.base import LLMProvider
from ...providers.types import ChatMessage, ChatRequest
from ...tools.execution import execute_tool_call
from ...tools.registry import ToolEntry, ToolRegistry
from ..errors import ToolExecutionError
from ..state import EngineState
from ...runtime.structured_output import ensure_structured_output, build_system_hint
from ...runtime.filters import pre_provider_filter, post_provider_filter
from ...runtime.policies import get_policy
from ...runtime.safety import redact_text
from ...runtime.history import assemble_prompt


def _build_initial_system_message(agent: AgentNode) -> ChatMessage | None:
    sys = agent.data.system_instructions or ""
    guide = agent.data.style_guide or ""
    text_parts: list[str] = []
    if sys:
        text_parts.append(sys)
    if guide:
        text_parts.append(guide)
    if not text_parts:
        return None
    return ChatMessage(role="system", content="\n\n".join(text_parts))


class _AgentRuntime:
    def __init__(
        self,
        *,
        node: AgentNode,
        provider: LLMProvider,
        tool_cfg: ToolsConfig,
        name_to_entry: dict[str, ToolEntry],
        provider_tool_specs: list[Any],
        overrides_by_name: dict[str, dict[str, Any]],
    ) -> None:
        self.node = node
        self.provider = provider
        self.tool_cfg = tool_cfg
        self.name_to_entry = name_to_entry
        self.provider_tool_specs = provider_tool_specs
        self.overrides_by_name = overrides_by_name
        self.system_message = _build_initial_system_message(node)

    def agent_fn(self, state: EngineState) -> EngineState:
        # Set cursor for visibility
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
        so = self.node.data.structured_output
        if getattr(so, "enabled", False):
            hint = build_system_hint(getattr(so, "schema_", None) or None)
            assistant, usage, finish_reason, attempts = ensure_structured_output(
                self.provider, req, policy_cfg=so, post_cfg=getattr(so, "post_process", None), system_hint=hint
            )
            assistant = post_provider_filter(assistant, safety_cfg=safety_cfg, policy=policy)
            meta2 = state.get("metadata", {})
            meta2["_last_finish_reason"] = finish_reason
            meta2["_last_usage"] = usage
            state["metadata"] = meta2
            state["steps"] = int(state.get("steps", 0)) + int(attempts)
        else:
            resp = self.provider.chat(req)
            assistant = post_provider_filter(resp.message, safety_cfg=safety_cfg, policy=policy)
            state["steps"] = int(state.get("steps", 0)) + 1
        state["messages"].append(assistant)
        state["tool_calls"] = assistant.tool_calls or None
        if not getattr(so, "enabled", False):
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
                raise ToolExecutionError(tool_name=tool_name, reason="Tool not attached")
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


def build(
    cfg: GraphConfig,
    order: list[str],
    *,
    provider_resolver: Callable[[Any], LLMProvider],
    registry: ToolRegistry,
)-> tuple[Any, str, LLMProvider, dict[str, ToolEntry], ToolsConfig, ModelParams, Any, Any]:
    """Build a sequential multi-agent LangGraph app.

    Returns a tuple of (app, root_agent, root_provider, root_tools_attached,
    root_tool_config, root_model_params).
    """

    # Map id -> AgentNode
    agents_by_id: dict[str, AgentNode] = {
        n.id: n for n in cfg.nodes if isinstance(n, AgentNode)
    }

    # Build per-agent runtime wiring
    runtimes: dict[str, _AgentRuntime] = {}
    for aid in order:
        agent_node = agents_by_id[aid]
        provider = provider_resolver(agent_node.data.model)
        tool_cfg = agent_node.data.tools
        attached_ids = list(tool_cfg.attached or [])
        provider_tool_specs = registry.to_provider_tool_specs(attached_ids) if attached_ids else []

        # Build mapping: tool name -> ToolEntry
        name_to_entry: dict[str, ToolEntry] = {}
        for tool_id in attached_ids:
            entry = registry.get(tool_id)
            name_to_entry[entry.impl.name] = entry

        # Build parameter overrides map from ToolNodes
        overrides_by_name: dict[str, dict[str, Any]] = {}
        for n in cfg.nodes:
            if isinstance(n, ToolNode) and n.data.tool_id in attached_ids:
                overrides_by_name[n.data.name] = dict(n.data.parameter_overrides or {})

        runtimes[aid] = _AgentRuntime(
            node=agent_node,
            provider=provider,
            tool_cfg=tool_cfg,
            name_to_entry=name_to_entry,
            provider_tool_specs=provider_tool_specs,
            overrides_by_name=overrides_by_name,
        )

    # Build LangGraph
    graph = StateGraph(EngineState)

    # Add nodes and edges per agent
    for _idx, aid in enumerate(order):
        rt = runtimes[aid]
        agent_key = f"agent__{aid}"
        tools_key = f"tools__{aid}"
        graph.add_node(agent_key, rt.agent_fn)
        graph.add_node(tools_key, rt.tools_fn)

    # Wire transitions
    for idx, aid in enumerate(order):
        agent_key = f"agent__{aid}"
        tools_key = f"tools__{aid}"
        next_id = order[idx + 1] if idx + 1 < len(order) else None

        def _cond_factory(
            my_tools_key: str, my_next_id: str | None
        ) -> Callable[[EngineState], str]:
            def _should_continue(state: EngineState) -> str:
                tcs = state.get("tool_calls") or []
                if tcs:
                    return my_tools_key
                return f"agent__{my_next_id}" if my_next_id is not None else END

            return _should_continue

        graph.add_conditional_edges(agent_key, _cond_factory(tools_key, next_id))
        graph.add_edge(tools_key, agent_key)

    # Start edge
    first = order[0]
    graph.add_edge(START, f"agent__{first}")

    app = graph.compile()

    # Root metadata for executor/streaming
    root = agents_by_id[first]
    root_rt = runtimes[first]
    return (
        app,
        first,
        root_rt.provider,
        root_rt.name_to_entry,
        root.data.tools,
        root.data.model,
        root.data.structured_output,
        root.data.safety,
    )


__all__ = ["build"]
