from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from langgraph.graph import END, START, StateGraph

from ..config.schema import AgentNode, GraphConfig, ToolNode, ToolsConfig
from ..providers.base import LLMProvider
from ..providers.gemini import GeminiProvider
from ..providers.openai import OpenAIProvider
from ..providers.types import ChatMessage, ChatRequest
from ..tools.execution import execute_tool_call
from ..tools.registry import ToolEntry, ToolRegistry, default_registry
from .tool_bindings import resolve_attached_tools
from .errors import (
    GraphCompileError,
    ToolExecutionError,
)
from .state import EngineState
from .types import CompiledSingleAgent, ProviderResolver
from ..runtime.history import assemble_prompt
from ..runtime.structured_output import ensure_structured_output, build_system_hint
from ..runtime.filters import pre_provider_filter, post_provider_filter
from ..runtime.policies import get_policy
from .regions.compose import OrchestrationResolver
from .regions.types import LeafRegion, collect_region_tree


def _default_provider_resolver() -> ProviderResolver:
    def _resolver(params: Any) -> LLMProvider:
        provider = params.provider
        if provider == "openai":
            return OpenAIProvider()
        if provider == "gemini":
            return GeminiProvider()
        raise GraphCompileError(f"Unsupported provider: {provider}")

    return _resolver


def _build_initial_system_message(agent: AgentNode) -> ChatMessage | None:
    sys = agent.data.system_instructions or ""
    guide = agent.data.style_guide or ""
    text_parts = []
    if sys:
        text_parts.append(sys)
    if guide:
        text_parts.append(guide)
    if not text_parts:
        return None
    return ChatMessage(role="system", content="\n\n".join(text_parts))


def compile_single_agent(
    cfg: GraphConfig,
    *,
    provider_resolver: Callable[[Any], LLMProvider] | None = None,
    registry: ToolRegistry | None = None,
) -> CompiledSingleAgent:
    """Compile a single agent.codeless config into a runnable LangGraph app.

    - Resolves provider
    - Attaches tool specs from registry
    - Builds a StateGraph with agent -> tools -> agent loop
    """
    # Find single agent node
    agents = [n for n in cfg.nodes if isinstance(n, AgentNode)]
    if len(agents) != 1:
        raise GraphCompileError(
            f"Expected exactly one agent.codeless node, found {len(agents)}"
        )
    agent_node = agents[0]

    # Provider
    resolver = provider_resolver or _default_provider_resolver()
    provider = resolver(agent_node.data.model)

    # Registry and resolved tool bindings/specs
    reg = registry or default_registry
    bindings = resolve_attached_tools(agent_node, cfg, reg)
    provider_tool_specs = [b.tool_spec for b in bindings]

    # Build mapping: function name -> binding
    bindings_by_fn: dict[str, Any] = {b.function_name: b for b in bindings}

    tool_cfg: ToolsConfig = agent_node.data.tools

    # Graph nodes

    def agent_fn(state: EngineState) -> EngineState:
        # Build request
        mp = agent_node.data.model
        meta = state.get("metadata", {}) or {}
        telemetry = meta.get("_telemetry") if isinstance(meta, dict) else None
        try:
            if telemetry is not None and hasattr(telemetry, "set_provider_model"):
                telemetry.set_provider_model(getattr(mp, "provider", None), getattr(mp, "model_id", None))
        except Exception:
            pass
        preamble = meta.get("org_preamble") if agent_node.data.context.inject_org_preamble else None
        vars_req = meta.get("vars") or {}
        req_messages = assemble_prompt(agent_node, state.get("messages", []), preamble, vars_req)
        # Safety pre-filter (PII redaction & injection defense)
        safety_cfg = agent_node.data.safety
        policy = get_policy(getattr(safety_cfg, "policy_ref", None))
        pre = pre_provider_filter(req_messages, safety_cfg=safety_cfg, policy=policy)
        if pre.blocked:
            # Short-circuit with refusal
            assistant = pre.refusal_message or ChatMessage(role="assistant", content="")
            assistant = post_provider_filter(assistant, safety_cfg=safety_cfg, policy=policy)
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
            tools=provider_tool_specs or None,
            tool_choice=(
                "auto"
                if tool_cfg.policy == "Auto" or tool_cfg.policy == "Required"
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
                "agent_id": agent_node.id,
                "agent_label": agent_node.label,
            },
            timeout=None,
        )
        # Structured output path
        so = agent_node.data.structured_output
        if getattr(so, "enabled", False):
            t0 = None
            try:
                if telemetry is not None and hasattr(telemetry, "start_timer"):
                    t0 = telemetry.start_timer("turn")
            except Exception:
                t0 = None
            hint = build_system_hint(getattr(so, "schema_", None) or None)
            final_msg, usage, finish_reason, attempts = ensure_structured_output(
                provider,
                req,
                policy_cfg=so,
                post_cfg=getattr(so, "post_process", None),
                system_hint=hint,
            )
            assistant = post_provider_filter(final_msg, safety_cfg=safety_cfg, policy=policy)
            state["messages"].append(assistant)
            state["tool_calls"] = assistant.tool_calls or None
            state["steps"] = int(state.get("steps", 0)) + int(attempts)
            meta2 = state.get("metadata", {})
            meta2["_last_finish_reason"] = finish_reason
            meta2["_last_usage"] = usage
            if pre.warnings:
                meta2["warnings"] = pre.warnings
            state["metadata"] = meta2
            # Telemetry turn record
            try:
                had_tools = bool(assistant.tool_calls)
                if t0 is not None and hasattr(telemetry, "stop_timer"):
                    dur_ms = telemetry.stop_timer(t0)
                else:
                    import time as _t

                    dur_ms = 0.0
                if telemetry is not None and hasattr(telemetry, "record_turn"):
                    telemetry.record_turn(duration_ms=dur_ms, finish_reason=finish_reason, had_tool_calls=had_tools)
                if telemetry is not None and usage is not None and hasattr(telemetry, "usage"):
                    telemetry.usage(
                        getattr(usage, "prompt_tokens", None),
                        getattr(usage, "completion_tokens", None),
                        getattr(usage, "total_tokens", None),
                    )
            except Exception:
                pass
            return state
        # Normal chat path
        t0 = None
        try:
            if telemetry is not None and hasattr(telemetry, "start_timer"):
                t0 = telemetry.start_timer("turn")
        except Exception:
            t0 = None
        try:
            resp = provider.chat(req)
        except Exception as _e:
            try:
                if telemetry is not None and hasattr(telemetry, "record_error"):
                    telemetry.record_error("provider", str(_e))
            except Exception:
                pass
            raise
        # Append assistant message
        assistant = post_provider_filter(resp.message, safety_cfg=safety_cfg, policy=policy)
        state["messages"].append(assistant)
        # Extract tool_calls for next step
        state["tool_calls"] = assistant.tool_calls or None
        state["steps"] = int(state.get("steps", 0)) + 1
        # Store finish metadata for executor to read
        meta = state.get("metadata", {})
        meta["_last_finish_reason"] = resp.finish_reason
        meta["_last_usage"] = resp.usage
        state["metadata"] = meta
        # Telemetry turn record
        try:
            had_tools = bool(assistant.tool_calls)
            if t0 is not None and hasattr(telemetry, "stop_timer"):
                dur_ms = telemetry.stop_timer(t0)
            else:
                import time as _t

                dur_ms = 0.0
            if telemetry is not None and hasattr(telemetry, "record_turn"):
                telemetry.record_turn(
                    duration_ms=dur_ms, finish_reason=resp.finish_reason, had_tool_calls=had_tools
                )
            if telemetry is not None and resp.usage is not None and hasattr(telemetry, "usage"):
                telemetry.usage(
                    getattr(resp.usage, "prompt_tokens", None),
                    getattr(resp.usage, "completion_tokens", None),
                    getattr(resp.usage, "total_tokens", None),
                )
        except Exception:
            pass
        return state

    def should_continue(state: EngineState) -> str:
        tcs = state.get("tool_calls") or []
        if tcs:
            return "tools"
        return END

    def tools_fn(state: EngineState) -> EngineState:
        tool_calls = list(state.get("tool_calls") or [])
        if not tool_calls:
            return state
        # determine how many to run this turn
        max_calls = tool_cfg.max_calls_per_turn or 0
        limit = len(tool_calls) if max_calls <= 0 else min(max_calls, len(tool_calls))
        timeout_s = (tool_cfg.timeout_ms or 0) / 1000.0 if tool_cfg.timeout_ms else None
        telemetry = (state.get("metadata", {}) or {}).get("_telemetry")

        for _idx, tc in enumerate(tool_calls[:limit]):
            fn_name = tc.name
            binding = bindings_by_fn.get(fn_name)
            if binding is None:
                raise ToolExecutionError(tool_name=fn_name, reason="Unknown tool function (did you attach the tool?)")
            entry = binding.entry
            overrides = binding.overrides or {}
            # Execute
            result_obj = execute_tool_call(
                entry.impl,
                tc.arguments_json or "",
                parameter_overrides=overrides,
                timeout_s=timeout_s,
                redact_for_logging=bool(getattr(agent_node.data.safety, "pii_redaction", True)),
                telemetry=telemetry,
            )
            # Append tool message with tool_call_id and JSON content
            content_json = json.dumps(result_obj)
            if getattr(agent_node.data.safety, "pii_redaction", True):
                from ..runtime.safety import redact_text

                content_json = redact_text(content_json)
            state["messages"].append(
                ChatMessage(
                    role="tool",
                    content=content_json,
                    tool_call_id=tc.id,
                )
            )

        # Clear pending calls
        state["tool_calls"] = None
        return state

    # Build LangGraph
    graph = StateGraph(EngineState)
    graph.add_node("agent", agent_fn)
    graph.add_node("tools", tools_fn)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")
    app = graph.compile()

    # Package compiled artifact
    leaf_region = LeafRegion(
        id=f"leaf:{agent_node.id}",
        agents=[agent_node.id],
        agent_id=agent_node.id,
        notes=["leaf"],
    )
    region_tree = collect_region_tree(leaf_region)

    return CompiledSingleAgent(
        agent_id=agent_node.id,
        graph=app,
        provider=provider,
        tools_attached={b.entry.impl.name: b.entry for b in bindings},
        tool_bindings_by_fn=bindings_by_fn,
        tool_config=tool_cfg,
        model_params=agent_node.data.model,
        structured_output=agent_node.data.structured_output,
        safety=agent_node.data.safety,
        region_tree=region_tree,
        region_notes=list(leaf_region.notes),
    )


def _detect_orchestration(cfg: GraphConfig) -> tuple[str, dict[str, Any]]:
    """Lightweight detection of orchestration kind for metadata endpoints.

    Returns a tuple of (kind, data). Currently used by the /compile route
    to label the orchestration as "single" when there is exactly one
    agent.codeless node; otherwise returns a generic label along with
    the root region kind if analysis succeeds.
    """
    # Single-agent fast path
    agents = [n for n in cfg.nodes if isinstance(n, AgentNode)]
    if len(agents) == 1:
        return "single", {"agents": [agents[0].id]}

    # Attempt region analysis for a more descriptive kind; fall back softly
    try:
        from .regions.analyzer import RegionAnalyzer

        region = RegionAnalyzer(cfg).analyze()
        # Map region kinds to a simple top-level label when possible
        root_kind = getattr(region, "kind", "unknown")
        label = "composed"
        if root_kind in ("sequential", "handoff", "concurrent", "groupchat", "leaf"):
            # Represent leaf with multiple agents (shouldn't usually happen) as composed
            label = "single" if root_kind == "leaf" and len(getattr(region, "agents", []) or []) == 1 else "composed"
        return label, {"rootRegionKind": root_kind}
    except Exception:
        # Non-fatal for metadata; caller should proceed
        return "unknown", {}
def compile_graph(
    cfg: GraphConfig,
    *,
    provider_resolver: Callable[[Any], LLMProvider] | None = None,
    registry: ToolRegistry | None = None,
) -> CompiledSingleAgent:
    resolver = provider_resolver or _default_provider_resolver()
    reg = registry or default_registry

    agents = [n for n in cfg.nodes if isinstance(n, AgentNode)]
    if len(agents) == 1:
        return compile_single_agent(cfg, provider_resolver=resolver, registry=reg)

    orchestrator = OrchestrationResolver(
        cfg=cfg,
        provider_resolver=resolver,
        registry=reg,
    )
    return orchestrator.compile()


__all__ = ["compile_single_agent", "compile_graph"]
