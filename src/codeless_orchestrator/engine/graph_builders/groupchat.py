from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, cast

from langgraph.graph import END, START, StateGraph

from ...config.schema import AgentNode, GraphConfig, ModelParams, ToolNode, ToolsConfig
from ...logging import get_logger
from ...providers.base import LLMProvider
from ...providers.types import ChatMessage, ChatRequest
from ...tools.execution import execute_many
from ...tools.registry import ToolEntry, ToolRegistry
from ..group_policy import choose_next_speaker, pick_round_robin
from ..state import EngineState, GroupBucket
from ...runtime.structured_output import ensure_structured_output, build_system_hint
from ...runtime.filters import pre_provider_filter, post_provider_filter
from ...runtime.policies import get_policy
from ...runtime.history import assemble_prompt

_log = get_logger(__name__)

DEFAULT_MAX_TURNS = 6


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
        self.system_message = _build_system_message(node)

    def agent_fn(self, state: EngineState) -> EngineState:
        # Set cursor
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
        # Speaker attribution prefix for visible assistant content
        if isinstance(assistant.content, str) and assistant.content:
            prefix = f"[Agent: {self.node.label}] "
            assistant = ChatMessage(
                role="assistant",
                content=prefix + assistant.content,
                tool_calls=assistant.tool_calls,
            )
        state["messages"].append(assistant)
        state["tool_calls"] = assistant.tool_calls or None
        if not getattr(so, "enabled", False):
            meta = state.get("metadata", {})
            meta["_last_finish_reason"] = resp.finish_reason
            meta["_last_usage"] = resp.usage
            state["metadata"] = meta
        # Increment group turn only when a full assistant turn completes (no tool calls)
        if not assistant.tool_calls:
            gb = state.get("group")
            if gb is not None:
                gb["turn"] = gb.get("turn", 0) + 1
                state["group"] = gb
        return state

    def tools_fn(self, state: EngineState) -> EngineState:
        tool_calls = list(state.get("tool_calls") or [])
        if not tool_calls:
            return state
        max_calls = self.tool_cfg.max_calls_per_turn or 0
        limit = len(tool_calls) if max_calls <= 0 else min(max_calls, len(tool_calls))
        timeout_s = (self.tool_cfg.timeout_ms or 0) / 1000.0 if self.tool_cfg.timeout_ms else None
        par = int(self.tool_cfg.parallelism or 1)
        # Execute concurrently up to parallelism and preserve ordering
        # Redact tool output for logging if safety demands
        redact_log = bool(getattr(self.node.data.safety, "pii_redaction", True))
        msgs = execute_many(
            tool_calls[:limit],
            name_to_entry=self.name_to_entry,
            overrides_by_name=self.overrides_by_name,
            timeout_s=timeout_s,
            max_workers=par,
            redact_for_logging=redact_log,
        )
        for m in msgs:
            state["messages"].append(m)
        state["tool_calls"] = None
        return state


def build(
    cfg: GraphConfig,
    scc_participants: list[str],
    *,
    provider_resolver: Callable[[Any], LLMProvider],
    registry: ToolRegistry,
    mode: Literal["round_robin", "moderator"],
    moderator_id: str | None = None,
    max_turns: int | None = None,
) -> tuple[Any, str, LLMProvider, dict[str, ToolEntry], ToolsConfig, ModelParams, Any, Any]:
    """Build a group chat LangGraph app over the participant agent ids."""

    agents_by_id: dict[str, AgentNode] = {n.id: n for n in cfg.nodes if isinstance(n, AgentNode)}
    participants_all = [aid for aid in scc_participants if aid in agents_by_id]
    # Exclude moderator from speaking participants in moderator mode
    if mode == "moderator" and moderator_id:
        participants = [aid for aid in participants_all if aid != moderator_id]
    else:
        participants = participants_all
    if not participants:
        raise ValueError("No valid participants for group chat")

    # Build per-agent runtime wiring
    runtimes: dict[str, _AgentRuntime] = {}
    for aid in participants:
        node = agents_by_id[aid]
        provider = provider_resolver(node.data.model)
        tool_cfg = node.data.tools
        attached_ids = list(tool_cfg.attached or [])
        provider_tool_specs = registry.to_provider_tool_specs(attached_ids) if attached_ids else []
        name_to_entry: dict[str, ToolEntry] = {}
        for tool_id in attached_ids:
            entry = registry.get(tool_id)
            name_to_entry[entry.impl.name] = entry
        overrides_by_name: dict[str, dict[str, Any]] = {}
        for n in cfg.nodes:
            if isinstance(n, ToolNode) and n.data.tool_id in attached_ids:
                overrides_by_name[n.data.name] = dict(n.data.parameter_overrides or {})
        runtimes[aid] = _AgentRuntime(
            node=node,
            provider=provider,
            tool_cfg=tool_cfg,
            name_to_entry=name_to_entry,
            provider_tool_specs=provider_tool_specs,
            overrides_by_name=overrides_by_name,
        )

    # Determine moderator provider if moderator mode
    mod_provider: LLMProvider | None = None
    if mode == "moderator":
        if not moderator_id or moderator_id not in agents_by_id:
            # try heuristic: first label containing 'moderator'
            for aid in scc_participants:
                node_opt = agents_by_id.get(aid)
                if node_opt and ("moderator" in (node_opt.label or "").lower()):
                    moderator_id = aid
                    break
        if moderator_id and moderator_id in agents_by_id:
            mod_provider = provider_resolver(agents_by_id[moderator_id].data.model)
        else:
            # Fallback to the first participant's provider
            mod_provider = provider_resolver(agents_by_id[participants[0]].data.model)

    # Graph
    graph = StateGraph(EngineState)

    # Add per-agent nodes
    for aid in participants:
        rt = runtimes[aid]
        graph.add_node(f"speak__{aid}", rt.agent_fn)
        graph.add_node(f"tools__{aid}", rt.tools_fn)

    max_turns_eff = DEFAULT_MAX_TURNS if not max_turns or max_turns <= 0 else max_turns

    # select_next node
    def select_next(state: EngineState) -> EngineState:
        # Initialize group bucket if missing
        if state.get("group") is None:
            state["group"] = cast(
                GroupBucket,
                {
                    "participants": list(participants),
                    "speaker": None,
                    "turn": 0,
                    "max_turns": max_turns_eff,
                    "mode": mode,
                },
            )
        gb = cast(GroupBucket, state.get("group"))

        # End checks
        turn = gb.get("turn", 0)
        if turn >= gb.get("max_turns", max_turns_eff):
            gb["end"] = True
            state["group"] = gb
            return state

        current = gb.get("speaker")
        nxt: str | None
        if mode == "round_robin":
            nxt = pick_round_robin(participants, current)
        else:
            # Moderator chooses among participants
            assert mod_provider is not None
            nxt, end = choose_next_speaker(
                messages=state.get("messages", []),
                participants=participants,
                current_speaker=current,
                moderator_provider=mod_provider,
            )
            if end:
                gb["end"] = True
                state["group"] = gb
                return state
            if nxt not in participants:
                _log.warning("Moderator returned invalid id '%s'; defaulting to first", nxt)
                nxt = participants[0]

        gb["speaker"] = nxt
        state["group"] = gb
        return state

    def _edge_from_select(state: EngineState) -> str:
        gb = state.get("group")
        if gb is None:
            return END
        if gb.get("end") is True:
            return END
        spk = gb.get("speaker")
        if not isinstance(spk, str) or spk not in participants:
            return END
        return f"speak__{spk}"

    # Conditional per agent
    def _cond_factory(aid: str) -> Callable[[EngineState], str]:
        tools_key = f"tools__{aid}"

        def _should_continue(state: EngineState) -> str:
            tcs = state.get("tool_calls") or []
            return tools_key if tcs else "select_next"

        return _should_continue

    # Wire graph
    graph.add_node("select_next", select_next)
    for aid in participants:
        graph.add_conditional_edges(f"speak__{aid}", _cond_factory(aid))
        graph.add_edge(f"tools__{aid}", f"speak__{aid}")

    graph.add_conditional_edges("select_next", _edge_from_select)
    graph.add_edge(START, "select_next")
    app = graph.compile()

    # Return tuple seeded by the first participant for streaming compatibility
    first = participants[0]
    root_node = agents_by_id[first]
    rt = runtimes[first]
    return (
        app,
        first,
        rt.provider,
        rt.name_to_entry,
        root_node.data.tools,
        root_node.data.model,
        root_node.data.structured_output,
        root_node.data.safety,
    )


__all__ = ["build", "DEFAULT_MAX_TURNS"]
