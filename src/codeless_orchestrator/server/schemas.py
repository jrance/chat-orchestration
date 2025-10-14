from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from ..engine.types import EngineRunResult
from ..providers.types import ChatMessage, ToolCall, Usage


# -------------------- Chat Message IO --------------------
Role = Literal["system", "user", "assistant", "tool"]


class ToolCallModel(BaseModel):
    id: Optional[str] = None
    name: str
    arguments_json: str = Field(default="", alias="argumentsJson")


class ChatMessageModel(BaseModel):
    role: Role
    content: Any
    tool_call_id: Optional[str] = Field(default=None, alias="toolCallId")
    tool_calls: Optional[list[ToolCallModel]] = Field(default=None, alias="toolCalls")

    model_config = ConfigDict(populate_by_name=True)

    def to_internal(self) -> ChatMessage:
        tcs: list[ToolCall] | None = None
        if self.tool_calls:
            tcs = [ToolCall(id=tc.id, name=tc.name, arguments_json=tc.arguments_json) for tc in self.tool_calls]
        return ChatMessage(
            role=self.role, content=self.content, tool_call_id=self.tool_call_id, tool_calls=tcs
        )

    @staticmethod
    def from_internal(msg: ChatMessage) -> "ChatMessageModel":
        tcs: list[ToolCallModel] | None = None
        if msg.tool_calls is not None:
            tcs = [ToolCallModel.model_validate(tc.__dict__) for tc in msg.tool_calls]
        return ChatMessageModel(
            role=msg.role, content=msg.content, tool_call_id=msg.tool_call_id, tool_calls=tcs
        )


# -------------------- Validate --------------------
class ValidateResponse(BaseModel):
    valid: bool
    errors: list[dict[str, Any]]
    warnings: list[dict[str, Any]]
    normalized: Optional[dict[str, Any]] = None


# -------------------- Compile --------------------
class CompileResponse(BaseModel):
    ok: bool = True
    orchestration: Literal["single", "sequential", "handoff", "concurrent", "groupchat"]
    agents: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    capabilities: dict[str, Any]
    notes: list[str] = Field(default_factory=list)


# -------------------- Execute --------------------
class ExecuteInputModel(BaseModel):
    messages: Optional[list[ChatMessageModel]] = None
    text: Optional[str] = None
    vars: Optional[dict[str, Any]] = None


class ExecuteOptionsModel(BaseModel):
    request_id: Optional[str] = Field(default=None, alias="requestId")


class ExecuteRequest(BaseModel):
    config: dict[str, Any]
    input: ExecuteInputModel
    options: Optional[ExecuteOptionsModel] = None


class UsageModel(BaseModel):
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

    @staticmethod
    def from_internal(u: Usage | None) -> Optional["UsageModel"]:
        if u is None:
            return None
        return UsageModel(
            prompt_tokens=u.prompt_tokens,
            completion_tokens=u.completion_tokens,
            total_tokens=u.total_tokens,
        )


class ExecuteResponse(BaseModel):
    messages: list[ChatMessageModel]
    finish_reason: Optional[str] = None
    usage: Optional[UsageModel] = None
    steps: int = 0
    request_id: str
    metrics: Optional[dict[str, Any]] = None

    @staticmethod
    def from_result(res: EngineRunResult, *, request_id: str) -> "ExecuteResponse":
        return ExecuteResponse(
            messages=[ChatMessageModel.from_internal(m) for m in res.messages],
            finish_reason=res.finish_reason,
            usage=UsageModel.from_internal(res.usage),
            steps=res.steps,
            request_id=request_id,
        )


# -------------------- SSE Stream Event (for docs only) --------------------
class StreamEvent(BaseModel):
    type: Literal[
        "role",
        "content.delta",
        "tool_call.start",
        "tool_call.delta",
        "tool_call.end",
        "message.end",
        "error",
        "end",
    ]
    role: Optional[Role] = None
    text_delta: Optional[str] = None
    tool_call_index: Optional[int] = None
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None
    arguments_delta: Optional[str] = None
    finish_reason: Optional[str] = None
    usage: Optional[UsageModel] = None
    request_id: Optional[str] = None
    error: Optional[dict[str, Any]] = None
