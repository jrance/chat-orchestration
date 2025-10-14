from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

# Roles for chat messages
Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class ContentPartText:
    type: Literal["text"]
    text: str


# Currently only text is supported; ready to extend for multimodal later
ContentPart = ContentPartText


@dataclass(frozen=True)
class ToolCall:
    id: str | None
    name: str
    # Raw streamed JSON string for arguments; parse at higher layer
    arguments_json: str


ToolParameters = dict[str, Any]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str | None
    parameters: ToolParameters


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str | list[ContentPart]
    # For role == "tool" messages coming from tools
    tool_call_id: str | None = None
    # For assistant messages that include tool calls
    tool_calls: list[ToolCall] | None = None


@dataclass(frozen=True)
class ChatRequest:
    model: str
    messages: list[ChatMessage]
    tools: list[ToolSpec] | None = None
    tool_choice: Literal["auto", "none"] | dict[str, Any] | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stop: list[str] | None = None
    seed: int | None = None
    json_mode_enabled: bool = False
    response_format_schema: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    timeout: float | None = None


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class ChatResponse:
    message: ChatMessage
    finish_reason: str | None = None
    usage: Usage | None = None
    raw: Any | None = None


@dataclass(frozen=True)
class ChatChunk:
    # Event type taxonomy for streaming
    type: Literal[
        "role",
        "content.delta",
        "tool_call.start",
        "tool_call.delta",
        "tool_call.end",
        "message.end",
    ]
    # Role set on first chunk when available
    role: Role | None = None
    # Text chunk for normal content
    text_delta: str | None = None
    # Tool call fields for function/tool calling
    tool_call_index: int | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    arguments_delta: str | None = None
    # Finalization / token accounting
    finish_reason: str | None = None
    usage: Usage | None = None
