from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, is_dataclass
from typing import Any, Protocol, cast, runtime_checkable

from ..providers.types import ToolSpec


class ToolError(Exception):
    """Base exception for tool-related errors."""


class ToolTimeoutError(ToolError):
    """Raised when a tool invocation exceeds the allowed timeout."""


class ToolValidationError(ToolError):
    """Raised when tool arguments are invalid or cannot be parsed."""


@runtime_checkable
class Tool(Protocol):
    """Typed protocol representing a tool implementation.

    Tools expose a name, description, and JSON Schema parameters. Implementations
    must provide an `invoke` method that accepts a dictionary of arguments and returns
    JSON-serializable data (dict, list, or str).
    """

    name: str
    description: str | None
    parameters: dict[str, Any]

    def invoke(self, args: dict[str, Any]) -> Any:
        ...


class BaseTool(ABC):
    """Light abstract base with common helpers.

    Subclasses should set `name`, `description`, and `parameters` and implement `invoke`.
    """

    name: str
    description: str | None = None
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {},
    }

    @abstractmethod
    def invoke(self, args: dict[str, Any]) -> Any:
        raise NotImplementedError

    def as_tool_spec(self) -> ToolSpec:
        """Convert this tool to a provider-facing ToolSpec."""
        return ToolSpec(name=self.name, description=self.description, parameters=self.parameters)

    @staticmethod
    def normalize_result(result: Any) -> dict | list | str:
        """Normalize an arbitrary result into the allowed JSON-serializable shapes.

        - dict, list, or str are accepted directly.
        - dataclasses are converted to dict via asdict.
        - bytes are decoded as UTF-8 strings.
        - Otherwise, raise ToolError to keep outputs deterministic.
        """
        if isinstance(result, (dict, list, str)):
            return result
        if is_dataclass(result) and not isinstance(result, type):
            return asdict(cast(Any, result))
        if isinstance(result, bytes):
            try:
                return result.decode("utf-8")
            except Exception as exc:  # pragma: no cover - extremely unlikely
                raise ToolError(f"Failed to decode bytes result: {exc}") from exc
        raise ToolError("Tool result must be dict, list, or str (after normalization)")
