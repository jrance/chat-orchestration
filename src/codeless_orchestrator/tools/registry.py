from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from ..providers.types import ToolSpec
from .base import BaseTool


class ToolEntry(BaseModel):
    """A registry entry describing a tool implementation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tool_id: str
    version: str | None = None
    impl: BaseTool

    def to_tool_spec(self) -> ToolSpec:
        return self.impl.as_tool_spec()


class ToolRegistry:
    """In-memory registry for tool discovery and conversion to provider specs."""

    def __init__(self) -> None:
        self._entries: dict[str, ToolEntry] = {}

    def register(self, entry: ToolEntry) -> None:
        self._entries[entry.tool_id] = entry

    def get(self, tool_id: str) -> ToolEntry:
        try:
            return self._entries[tool_id]
        except KeyError:
            raise KeyError(f"Unknown tool_id: {tool_id}") from None

    def has(self, tool_id: str) -> bool:
        return tool_id in self._entries

    def to_provider_tool_specs(self, attached: list[str]) -> list[ToolSpec]:
        """Return provider ToolSpecs for the set of attached tool IDs.

        Raises KeyError if any attached tool_id is not registered to fail fast.
        """
        specs: list[ToolSpec] = []
        for tool_id in attached:
            entry = self.get(tool_id)
            specs.append(entry.to_tool_spec())
        return specs


# Default global registry and builtin registration helper
default_registry = ToolRegistry()


def register_default_tools(registry: ToolRegistry | None = None) -> None:
    """Register builtin tools in the provided registry (or default)."""
    reg = registry or default_registry
    try:
        from .builtin.web_search import WebSearchTool
    except Exception:  # pragma: no cover - import issues should not break process
        return

    reg.register(
        ToolEntry(tool_id="tool:web-search", version=None, impl=WebSearchTool())
    )
