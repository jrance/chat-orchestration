from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

from pydantic import BaseModel, ConfigDict, Field

from ..config.schema import GraphConfig, ModelParams
from ..providers.base import LLMProvider
from ..providers.types import ChatChunk, ChatMessage, Usage
from ..tools.registry import ToolEntry, ToolRegistry


class CompiledSingleAgent(NamedTuple):
    """Compiled artifact for executing a single agent.codeless graph."""

    agent_id: str
    graph: Any  # LangGraph app
    provider: LLMProvider
    tools_attached: dict[str, ToolEntry]  # key: tool name -> entry
    tool_config: Any  # ToolsConfig (from config.schema)
    model_params: ModelParams
    structured_output: Any | None
    safety: Any | None


class EngineRunInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    messages: list[ChatMessage] | str
    metadata: dict[str, Any] = Field(default_factory=dict)
    vars: dict[str, Any] | None = None


class EngineRunResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    messages: list[ChatMessage]
    finish_reason: str | None
    usage: Usage | None
    steps: int


# Provider resolver type alias
ProviderResolver = Callable[[ModelParams], LLMProvider]


__all__ = [
    "CompiledSingleAgent",
    "EngineRunInput",
    "EngineRunResult",
    "ProviderResolver",
    "GraphConfig",
    "ModelParams",
    "ChatMessage",
    "ChatChunk",
    "LLMProvider",
    "ToolEntry",
    "ToolRegistry",
    "Usage",
]
