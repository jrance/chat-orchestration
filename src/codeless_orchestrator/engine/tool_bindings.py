from __future__ import annotations

import re
from typing import Any

from ..config.schema import AgentNode, GraphConfig, ToolNode
from ..tools.registry import ToolEntry, ToolRegistry
from ..providers.types import ToolSpec
from .errors import GraphCompileError
from .types import ToolAttachmentBinding


_ALLOWED_NAME_RE = re.compile(r"[^a-z0-9_]+")


def slugify_function_name(base: str, node_id: str | None) -> str:
    """Return a safe, <=64-char function name with optional node-id suffix.

    - Lowercase
    - Hyphens and non [a-z0-9_] -> underscore
    - Collapse duplicate underscores and trim
    - If node_id provided, append '__' + last 8 hex characters (hyphens removed)
    """
    s = (base or "").lower().replace("-", "_")
    s = _ALLOWED_NAME_RE.sub("_", s)
    s = re.sub(r"_+", "_", s).strip("_") or "tool"
    if node_id:
        short = re.sub(r"[^a-zA-Z0-9]", "", node_id)[-8:] or "bind"
        s = f"{s}__{short.lower()}"
    # Guard length
    return s[:64]


def resolve_attached_tools(
    agent_node: AgentNode, cfg: GraphConfig, registry: ToolRegistry
) -> list[ToolAttachmentBinding]:
    """Resolve attachments on an agent node to concrete registry entries and provider specs.

    Supports both forms in `attached`:
    - Tool node IDs from the graph (preferred from UI)
    - Direct registry tool IDs (legacy)
    """
    node_by_id: dict[str, Any] = {n.id: n for n in cfg.nodes}
    bindings: list[ToolAttachmentBinding] = []

    for attached_id in list(agent_node.data.tools.attached or []):
        node_id: str | None = None
        label: str
        tool_id: str
        overrides: dict[str, Any]

        # Prefer legacy direct toolId form when value looks like a registry id
        if isinstance(attached_id, str) and attached_id.startswith("tool:"):
            tool_id = attached_id
            overrides = {}
            label = tool_id
            for maybe in cfg.nodes:
                if isinstance(maybe, ToolNode) and maybe.data.tool_id == tool_id:
                    overrides = dict(maybe.data.parameter_overrides or {})
                    label = maybe.label
                    break
        else:
            n = node_by_id.get(attached_id)
            if isinstance(n, ToolNode):
                node_id = n.id
                label = n.label
                tool_id = n.data.tool_id
                overrides = dict(n.data.parameter_overrides or {})
            else:
                # Treat as direct registry tool id (back-compat). If a ToolNode with
                # matching tool_id exists in the graph, use its overrides/label.
                tool_id = attached_id
                overrides = {}
                label = tool_id
                # Prefer the first matching node for overrides if present
                for maybe in cfg.nodes:
                    if isinstance(maybe, ToolNode) and maybe.data.tool_id == tool_id:
                        overrides = dict(maybe.data.parameter_overrides or {})
                        label = maybe.label
                        break

        # Lookup registry entry with clearer error message
        if not registry.has(tool_id):
            raise GraphCompileError(
                f"Unknown toolId '{tool_id}' for tool attachment '{attached_id}'"
            )
        entry: ToolEntry = registry.get(tool_id)

        function_name = slugify_function_name(entry.impl.name, node_id)
        tool_spec = ToolSpec(
            name=function_name,
            description=entry.impl.description,
            parameters=entry.impl.parameters,
        )

        bindings.append(
            ToolAttachmentBinding(
                node_id=node_id,
                label=label,
                tool_id=tool_id,
                function_name=function_name,
                overrides=overrides,
                tool_spec=tool_spec,
                entry=entry,
            )
        )

    return bindings


__all__ = ["resolve_attached_tools", "slugify_function_name"]
