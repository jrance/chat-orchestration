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
    OrchestrationUnsupportedError,
    ToolExecutionError,
)
from .graph_builders import concurrent as conc_builder
from .graph_builders import groupchat as groupchat_builder
from .graph_builders import handoff as handoff_builder
from .graph_builders import sequential as seq_builder
from .state import EngineState
from .types import CompiledSingleAgent, ProviderResolver
from ..runtime.history import assemble_prompt
from ..runtime.structured_output import ensure_structured_output, build_system_hint
from ..runtime.filters import pre_provider_filter, post_provider_filter
from ..runtime.policies import get_policy


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
    )


def _agent_graph(cfg: GraphConfig) -> tuple[dict[str, AgentNode], list[tuple[str, str]]]:
    agents: dict[str, AgentNode] = {n.id: n for n in cfg.nodes if isinstance(n, AgentNode)}
    edges: list[tuple[str, str]] = []
    for e in cfg.edges:
        if e.from_ in agents and e.to in agents:
            edges.append((e.from_, e.to))
    return agents, edges


def _has_self_loop(aedges: list[tuple[str, str]]) -> bool:
    return any(u == v for u, v in aedges)


def _sccs(nodes: list[str], edges: list[tuple[str, str]]) -> list[list[str]]:
    # Tarjan's algorithm
    index = 0
    indices: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    stack: list[str] = []
    onstack: set[str] = set()
    result: list[list[str]] = []
    adj: dict[str, list[str]] = {n: [] for n in nodes}
    for u, v in edges:
        if u in adj:
            adj[u].append(v)

    def strongconnect(v: str) -> None:
        nonlocal index
        indices[v] = index
        lowlink[v] = index
        index += 1
        stack.append(v)
        onstack.add(v)
        for w in adj.get(v, []):
            if w not in indices:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in onstack:
                lowlink[v] = min(lowlink[v], indices[w])
        # If v is a root node, pop the stack and output an SCC
        if lowlink[v] == indices[v]:
            scc: list[str] = []
            while True:
                w = stack.pop()
                onstack.remove(w)
                scc.append(w)
                if w == v:
                    break
            result.append(scc)

    for n in nodes:
        if n not in indices:
            strongconnect(n)
    return result


def _detect_orchestration(cfg: GraphConfig) -> tuple[str, dict[str, Any]]:
    """Detect orchestration pattern.

    Returns (kind, data) where kind in {"single", "sequential", "handoff"}.
    """
    agents, aedges = _agent_graph(cfg)
    if not agents:
        raise GraphCompileError("No agent.codeless nodes found")
    if len(agents) == 1:
        return "single", {"only": next(iter(agents.values()))}

    if not aedges:
        raise OrchestrationUnsupportedError(
            "Multiple agents without edges are unsupported (connect them or reduce to one agent)"
        )

    # Group chat detection: any SCC size >= 2 or a self-loop
    node_ids = list(agents.keys())
    scc_list = _sccs(node_ids, aedges)
    # Identify SCCs that represent cycles (size >=2) or a self loop present
    cyc_sccs: list[list[str]] = [s for s in scc_list if len(s) >= 2]
    if not cyc_sccs and _has_self_loop(aedges):
        # Find node(s) with self-loop
        loop_nodes = [u for u, v in aedges if u == v]
        if loop_nodes:
            cyc_sccs = [[loop_nodes[0]]]
    if cyc_sccs:
        # Use the first detected SCC; order participants by appearance in cfg.nodes for determinism
        scc_set = set(cyc_sccs[0])
        ordered_participants = [
            n.id
            for n in cfg.nodes
            if isinstance(n, AgentNode) and n.id in scc_set
        ]
        # Compute degrees inside SCC
        indeg_scc: dict[str, int] = {aid: 0 for aid in ordered_participants}
        outdeg_scc: dict[str, int] = {aid: 0 for aid in ordered_participants}
        internal_edges = [(u, v) for u, v in aedges if u in scc_set and v in scc_set]
        for u, v in internal_edges:
            outdeg_scc[u] += 1
            indeg_scc[v] += 1
        # Heuristic: moderator if exactly one node has indeg>=2 and outdeg>=2
        # OR label contains 'moderator'
        hub_candidates = [
            aid
            for aid in ordered_participants
            if indeg_scc[aid] >= 2 and outdeg_scc[aid] >= 2
        ]
        labels = {aid: agents[aid].label or "" for aid in ordered_participants}
        label_mod = [aid for aid, lab in labels.items() if "moderator" in lab.lower()]
        mode = "round_robin"
        moderator_id: str | None = None
        if len(hub_candidates) == 1:
            mode = "moderator"
            moderator_id = hub_candidates[0]
        if label_mod:
            mode = "moderator"
            moderator_id = label_mod[0]
        return "groupchat", {
            "participants": ordered_participants,
            "mode": mode,
            "moderator_id": moderator_id,
        }

    # indegree / outdegree and adjacency
    indeg: dict[str, int] = {aid: 0 for aid in agents}
    outdeg: dict[str, int] = {aid: 0 for aid in agents}
    adj: dict[str, list[str]] = {aid: [] for aid in agents}
    for u, v in aedges:
        outdeg[u] += 1
        indeg[v] += 1
        adj[u].append(v)

    roots = [aid for aid, d in indeg.items() if d == 0]
    if len(roots) != 1:
        raise OrchestrationUnsupportedError(
            "Multiple roots detected; fan-out/concurrency is planned for PR 8"
        )

    root = roots[0]

    # Check for cycles and reachability from root via Kahn's algorithm restricted to reachable set
    from collections import deque

    reach_indeg = indeg.copy()
    q = deque([root])
    visited: list[str] = []
    seen_set: set[str] = set()
    while q:
        u = q.popleft()
        visited.append(u)
        seen_set.add(u)
        for v in adj.get(u, []):
            reach_indeg[v] -= 1
            # Enqueue only when all parents seen in reachable subgraph
            if reach_indeg[v] == 0 and v not in seen_set:
                q.append(v)

    if len(visited) != len(agents):
        # Either cycles or disconnected components
        # Detect cycle quickly: if any node in reachable set still has indegree > 0
        has_cycle = any(d > 0 for d in reach_indeg.values())
        if has_cycle:
            raise OrchestrationUnsupportedError(
                "Agent graph contains cycles; group chat planned for PR 9"
            )
        raise OrchestrationUnsupportedError(
            "Agent graph has disconnected components; single-root, connected graphs only"
        )

    # Sequential if outdegree <= 1 for all and path covers all agents following unique next
    if all(outdeg[aid] <= 1 for aid in agents):
        # Construct order by following next pointers from root
        order: list[str] = []
        cur = root
        seen: set[str] = set()
        ok = True
        while True:
            order.append(cur)
            seen.add(cur)
            outs = adj.get(cur, [])
            if not outs:
                break
            if len(outs) > 1:
                ok = False
                break
            nxt = outs[0]
            if nxt in seen:
                ok = False
                break
            cur = nxt
        if ok and len(order) == len(agents):
            return "sequential", {"order": order}

    # Otherwise handoff if at least one node has outdegree > 1
    if any(outdeg[aid] > 1 for aid in agents):
        return "handoff", {"adjacency": adj, "root": root}

    # Fallback unsupported pattern
    raise OrchestrationUnsupportedError(
        "Unsupported agent graph pattern; requires either a chain or branching handoff"
    )


def _detect_concurrent(
    cfg: GraphConfig,
) -> tuple[bool, dict[str, Any]]:
    """Detect a simple concurrent fan-out/fan-in stage.

    Returns (True, data) if detected where data contains:
      - root: str (the fan-out source agent id)
      - branches: list[str]
      - next: str | None (common join agent if present)
    """
    # Heuristic opt-in: only consider when meta name hints at concurrency
    name_hint = (getattr(getattr(cfg, "meta", None), "name", "") or "").lower()
    if "concurrent" not in name_hint:
        return False, {}

    agents: dict[str, AgentNode] = {n.id: n for n in cfg.nodes if isinstance(n, AgentNode)}
    aedges: list[tuple[str, str]] = [
        (e.from_, e.to) for e in cfg.edges if e.from_ in agents and e.to in agents
    ]
    if not agents:
        return False, {}
    # Build degree/adjacency
    indeg: dict[str, int] = {aid: 0 for aid in agents}
    outdeg: dict[str, int] = {aid: 0 for aid in agents}
    adj: dict[str, list[str]] = {aid: [] for aid in agents}
    for u, v in aedges:
        outdeg[u] += 1
        indeg[v] += 1
        adj[u].append(v)

    # Single root only
    roots = [aid for aid, d in indeg.items() if d == 0]
    if len(roots) != 1:
        return False, {}
    root = roots[0]
    # Fan-out at root
    if outdeg[root] <= 1:
        return False, {}
    branches = list(adj[root])
    if not branches:
        return False, {}
    # Case 1: leaves
    if all(outdeg[b] == 0 for b in branches):
        return True, {"root": root, "branches": branches, "next": None}
    # Case 2: common join
    if all(outdeg[b] == 1 for b in branches):
        nexts = [adj[b][0] for b in branches]
        uniq = set(nexts)
        if len(uniq) == 1:
            return True, {"root": root, "branches": branches, "next": nexts[0]}
    return False, {}


def compile_graph(
    cfg: GraphConfig,
    *,
    provider_resolver: Callable[[Any], LLMProvider] | None = None,
    registry: ToolRegistry | None = None,
) -> CompiledSingleAgent:
    """Compile a graph config into a runnable LangGraph app.

    Supports single-agent, sequential multi-agent, and handoff multi-agent
    orchestrations. Returns a CompiledSingleAgent artifact with metadata seeded
    from the root agent for streaming compatibility.
    """
    resolver = provider_resolver or _default_provider_resolver()
    reg = registry or default_registry

    kind, data = _detect_orchestration(cfg)
    if kind == "single":
        return compile_single_agent(cfg, provider_resolver=resolver, registry=reg)
    if kind == "sequential":
        app, root_id, root_provider, tools_attached, tool_cfg, model_params, structured, safety = seq_builder.build(
            cfg, data["order"], provider_resolver=resolver, registry=reg
        )
    elif kind == "groupchat":
        app, root_id, root_provider, tools_attached, tool_cfg, model_params, structured, safety = (
            groupchat_builder.build(
                cfg,
                data["participants"],
                provider_resolver=resolver,
                registry=reg,
                mode=data.get("mode", "round_robin"),
                moderator_id=data.get("moderator_id"),
                max_turns=None,
            )
        )
    else:
        # Before falling back to handoff, try to detect a simple concurrent stage
        is_conc, cdata = _detect_concurrent(cfg)
        if is_conc:
            # Infer strategy inside builder; allows tests to monkeypatch
            app, root_id, root_provider, tools_attached, tool_cfg, model_params, structured, safety = (
                conc_builder.build(
                    cfg,
                    cdata["root"],
                    cdata["branches"],
                    cdata.get("next"),
                    provider_resolver=resolver,
                    registry=reg,
                    merge_strategy=None,
                )
            )
        else:
            app, root_id, root_provider, tools_attached, tool_cfg, model_params, structured, safety = (
                handoff_builder.build(
                    cfg,
                    data["adjacency"],
                    data["root"],
                    provider_resolver=resolver,
                    registry=reg,
                )
            )

    return CompiledSingleAgent(
        agent_id=root_id,
        graph=app,
        provider=root_provider,
        tools_attached=tools_attached,
        tool_bindings_by_fn=None,
        tool_config=tool_cfg,
        model_params=model_params,
        structured_output=structured,
        safety=safety,
    )


__all__ = ["compile_single_agent", "compile_graph"]
