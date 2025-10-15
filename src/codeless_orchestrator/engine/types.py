from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..config.schema import GraphConfig, ModelParams
from ..providers.base import LLMProvider
from ..providers.types import ChatChunk, ChatMessage, Usage
from ..providers.types import ToolSpec
from ..tools.registry import ToolEntry, ToolRegistry
from .regions.types import RegionTree


class ToolAttachmentBinding(BaseModel):
    """Resolved binding between an agent's tool attachment and a registry entry.

    Supports both node-id attachments (preferred) and direct registry toolId strings
    (legacy). For node-id attachments, a unique function name is generated to
    disambiguate multiple attachments of the same tool type.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    node_id: str | None
    label: str
    tool_id: str
    function_name: str
    overrides: dict[str, Any]
    tool_spec: ToolSpec
    entry: ToolEntry


@dataclass(slots=True)
class CompiledSingleAgent:
    """Compiled artifact for executing a graph produced by the resolver."""

    agent_id: str
    graph: Any  # LangGraph app
    provider: LLMProvider
    tools_attached: dict[str, ToolEntry]
    tool_bindings_by_fn: dict[str, ToolAttachmentBinding] | None
    tool_config: Any  # ToolsConfig (from config.schema)
    model_params: ModelParams
    structured_output: Any | None
    safety: Any | None
    region_tree: RegionTree | None = None
    region_notes: list[str] = field(default_factory=list)


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
    "ToolAttachmentBinding",
    "ToolEntry",
    "ToolRegistry",
    "Usage",
]
