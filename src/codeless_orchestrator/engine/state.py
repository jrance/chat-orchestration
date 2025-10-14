from __future__ import annotations

from typing import Any, Literal, TypedDict

from ..providers.types import ChatMessage, ToolCall


class GroupBucket(TypedDict, total=False):
    participants: list[str]
    speaker: str | None
    turn: int
    max_turns: int
    mode: Literal["moderator", "round_robin"]
    end: bool


class EngineState(TypedDict):
    """Typed state for the orchestration engine.

    - messages: full running transcript including system, user, assistant, and tool messages
    - metadata: freeform labels and run info
    - tool_calls: list of tool calls produced by the last assistant message (if any)
    - steps: number of LLM turns completed so far
    - agent_cursor: optional id of the agent currently running (for multi-agent graphs)
    - concurrent: transient bucket for fan-out/fan-in stages
    """

    messages: list[ChatMessage]
    metadata: dict[str, Any]
    tool_calls: list[ToolCall] | None
    steps: int
    agent_cursor: str | None
    concurrent: dict[str, Any] | None
    group: GroupBucket | None


__all__ = ["EngineState", "GroupBucket"]
