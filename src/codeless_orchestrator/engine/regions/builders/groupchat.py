from __future__ import annotations

from collections.abc import Callable
from typing import Literal, cast

from langgraph.graph import StateGraph

from ....logging import get_logger
from ....providers.base import LLMProvider
from ....providers.types import ChatMessage
from ...group_policy import choose_next_speaker, pick_round_robin
from ..builder import BuildChildFn, RegionBuildResult
from ..context import CompileContext
from ..types import GroupChatRegion
from ...state import EngineState, GroupBucket

_LOG = get_logger(__name__)
_DEFAULT_MAX_TURNS = 6


def _init_group_bucket(
    *,
    state: EngineState,
    participants: list[str],
    mode: Literal["moderator", "round_robin"],
    max_turns: int,
) -> GroupBucket:
    bucket = cast(GroupBucket, {
        "participants": list(participants),
        "speaker": None,
        "turn": 0,
        "max_turns": max_turns,
        "mode": mode,
        "end": False,
    })
    state["group"] = bucket
    return bucket


def _select_factory(
    *,
    region: GroupChatRegion,
    participants: list[str],
    moderator_provider: LLMProvider | None,
    max_turns: int,
) -> Callable[[EngineState], EngineState]:
    def _select(state: EngineState) -> EngineState:
        bucket = state.get("group")
        if bucket is None or not isinstance(bucket, dict):
            bucket = _init_group_bucket(
                state=state,
                participants=participants,
                mode=region.mode,
                max_turns=max_turns,
            )
        gb = cast(GroupBucket, bucket)

        if gb.get("end"):
            state["group"] = gb
            return state

        turn = gb.get("turn", 0)
        if turn >= max_turns:
            gb["end"] = True
            state["group"] = gb
            return state

        current = gb.get("speaker")
        nxt: str | None = None
        end_conversation = False
        if region.mode == "round_robin":
            nxt = pick_round_robin(participants, current)
        else:
            if moderator_provider is None:
                nxt = pick_round_robin(participants, current)
            else:
                nxt, end_conversation = choose_next_speaker(
                    messages=state.get("messages", []),
                    participants=participants,
                    current_speaker=current,
                    moderator_provider=moderator_provider,
                )
        if end_conversation or nxt not in participants:
            if nxt not in participants and nxt is not None:
                _LOG.warning("Moderator returned invalid speaker '%s'; ending conversation", nxt)
            gb["end"] = True
            state["group"] = gb
            return state

        gb["speaker"] = nxt
        state["group"] = gb
        return state

    return _select


def _select_conditional_factory(
    *,
    participant_keys: dict[str, str],
    completion_target: str,
) -> Callable[[EngineState], str]:
    def _edge(state: EngineState) -> str:
        bucket = state.get("group")
        if not bucket or bucket.get("end"):
            return completion_target
        speaker = bucket.get("speaker")
        if isinstance(speaker, str) and speaker in participant_keys:
            return participant_keys[speaker]
        return completion_target

    return _edge


def _agent_conditional_factory(
    *,
    tools_key: str,
    select_key: str,
) -> Callable[[EngineState], str]:
    def _should_continue(state: EngineState) -> str:
        tcs = state.get("tool_calls") or []
        return tools_key if tcs else select_key

    return _should_continue


def build_groupchat(
    *,
    region: GroupChatRegion,
    graph: StateGraph,
    ctx: CompileContext,
    build_child: BuildChildFn,
) -> RegionBuildResult:
    participants = list(region.participants or region.agents)
    if not participants:
        raise ValueError("GroupChatRegion requires at least one participant")

    entry_key = ctx.region_entry_key(region.id)
    exit_key = ctx.region_exit_key(region.id)
    ctx.ensure_passthrough_node(graph, entry_key)
    ctx.ensure_passthrough_node(graph, exit_key)

    select_key = f"group__{region.id}__select"
    participant_keys: dict[str, str] = {}
    tool_keys: dict[str, str] = {}

    for aid in participants:
        artifacts = ctx.agent_artifacts(aid)
        agent_key = ctx.agent_node_key(region.id, aid)
        tools_key = ctx.tools_node_key(region.id, aid)
        participant_keys[aid] = agent_key
        tool_keys[aid] = tools_key
        label = artifacts.node.label or artifacts.node.id

        def _speaker_factory(fn: Callable[[EngineState], EngineState], speaker_label: str) -> Callable[[EngineState], EngineState]:
            def _wrapped(state: EngineState) -> EngineState:
                before = len(state.get("messages", []))
                new_state = fn(state)
                messages = new_state.get("messages", [])
                if len(messages) > before:
                    last = messages[-1]
                    if isinstance(last, ChatMessage) and isinstance(last.content, str) and last.content:
                        prefix = f"[Agent: {speaker_label}] "
                        if not last.content.startswith(prefix):
                            messages[-1] = ChatMessage(
                                role=last.role,
                                content=prefix + last.content,
                                tool_calls=last.tool_calls,
                            )
                return new_state

            return _wrapped

        graph.add_node(agent_key, _speaker_factory(artifacts.agent_fn, label))
        graph.add_node(tools_key, artifacts.tools_fn)
        graph.add_edge(tools_key, agent_key)

    moderator_provider: LLMProvider | None = None
    if region.mode == "moderator":
        moderator_id = region.moderator_id
        candidate_ids = list(participants)
        if moderator_id is not None:
            candidate_ids = [moderator_id] + [aid for aid in candidate_ids if aid != moderator_id]
        for aid in candidate_ids:
            try:
                node = ctx.get_agent(aid)
            except Exception:
                continue
            if moderator_id is None and "moderator" not in (node.label or "").lower():
                continue
            try:
                moderator_provider = ctx.agent_artifacts(aid).provider
                break
            except Exception:
                continue

    max_turns = _DEFAULT_MAX_TURNS

    # Build optional downstream region before finalizing completion targets
    next_res: RegionBuildResult | None = None
    if region.next_region is not None:
        next_res = build_child(region.next_region, True)

    completion_target = next_res.entry_key if next_res is not None else exit_key
    graph.add_node(
        select_key,
        _select_factory(
            region=region,
            participants=participants,
            moderator_provider=moderator_provider,
            max_turns=max_turns,
        ),
    )
    graph.add_edge(entry_key, select_key)
    graph.add_conditional_edges(
        select_key,
        _select_conditional_factory(
            participant_keys=participant_keys,
            completion_target=completion_target,
        ),
    )

    for aid in participants:
        agent_key = participant_keys[aid]
        tools_key = tool_keys[aid]
        graph.add_conditional_edges(
            agent_key,
            _agent_conditional_factory(tools_key=tools_key, select_key=select_key),
        )

    if next_res is not None:
        graph.add_edge(next_res.exit_key, exit_key)

    notes = list(region.notes)
    if next_res is not None:
        notes.extend(next_res.notes)

    return RegionBuildResult(region=region, entry_key=entry_key, exit_key=exit_key, notes=notes)


__all__ = ["build_groupchat"]
