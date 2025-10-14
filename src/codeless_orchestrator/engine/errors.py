from __future__ import annotations

from dataclasses import dataclass


class EngineError(Exception):
    """Base exception for engine-related errors."""


class GraphCompileError(EngineError):
    """Raised when the provided graph config cannot be compiled."""


@dataclass
class ToolExecutionError(EngineError):
    """Raised when a tool call cannot be executed."""

    tool_name: str
    reason: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"Tool '{self.tool_name}' failed: {self.reason}"


class OrchestrationUnsupportedError(EngineError):
    """Raised when the agent graph implies an orchestration pattern we don't support yet."""

