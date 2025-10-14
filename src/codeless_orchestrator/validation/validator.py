from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from ..config.schema import AgentNode, GraphConfig, ToolNode
from .models import ValidationIssue, ValidationReport


def _duplicates(items: Iterable[str]) -> list[str]:
    cnt = Counter(items)
    return [k for k, v in cnt.items() if v > 1]


def validate_graph(cfg: GraphConfig) -> ValidationReport:
    issues: list[ValidationIssue] = []

    # Duplicate node ids
    dups = _duplicates(n.id for n in cfg.nodes)
    for nid in dups:
        issues.append(
            ValidationIssue(
                code="node.duplicate_id",
                message=f"Duplicate node id {nid}",
                path=["nodes", nid],
                severity="ERROR",
            )
        )

    # Duplicate edge ids
    edge_dups = _duplicates(e.id for e in cfg.edges)
    for eid in edge_dups:
        issues.append(
            ValidationIssue(
                code="edge.duplicate_id",
                message=f"Duplicate edge id {eid}",
                path=["edges", eid],
                severity="ERROR",
            )
        )

    node_by_id = {n.id: n for n in cfg.nodes}

    # Edge endpoints exist
    for e in cfg.edges:
        if e.from_ not in node_by_id:
            issues.append(
                ValidationIssue(
                    code="edge.unknown_node",
                    message=f"Edge.from refers to unknown node {e.from_}",
                    path=["edges", e.id, "from"],
                    severity="ERROR",
                )
            )
        if e.to not in node_by_id:
            issues.append(
                ValidationIssue(
                    code="edge.unknown_node",
                    message=f"Edge.to refers to unknown node {e.to}",
                    path=["edges", e.id, "to"],
                    severity="ERROR",
                )
            )

    # Tool attachments must exist and be of kind tool (when they reference nodes)
    tool_ids = {n.id for n in cfg.nodes if isinstance(n, ToolNode)}
    for n in cfg.nodes:
        if isinstance(n, AgentNode):
            for t in n.data.tools.attached:
                # Allow direct registry tool IDs (e.g., 'tool:web-search') in attachments
                if t not in node_by_id:
                    if isinstance(t, str) and t.startswith("tool:"):
                        continue
                    issues.append(
                        ValidationIssue(
                            code="tools.unknown_attachment",
                            message=f"Agent attaches unknown tool {t}",
                            path=["nodes", n.id, "tools", "attached"],
                            severity="ERROR",
                        )
                    )
                elif t not in tool_ids:
                    node = node_by_id.get(t)
                    node_label = getattr(node, "label", "") if node is not None else ""
                    issues.append(
                        ValidationIssue(
                            code="tools.attachment_not_tool",
                            message=f"Attached id {t} is not a tool (label: '{node_label}')",
                            path=["nodes", n.id, "tools", "attached"],
                            severity="ERROR",
                        )
                    )

            # Optional warnings on tool policy values
            if n.data.tools.max_calls_per_turn < 0:
                issues.append(
                    ValidationIssue(
                        code="tools.max_calls_per_turn.invalid",
                        message="max_calls_per_turn is negative",
                        path=["nodes", n.id, "tools", "max_calls_per_turn"],
                        severity="WARNING",
                    )
                )
            if n.data.tools.parallelism < 1:
                issues.append(
                    ValidationIssue(
                        code="tools.parallelism.invalid",
                        message="parallelism should be >= 1",
                        path=["nodes", n.id, "tools", "parallelism"],
                        severity="WARNING",
                    )
                )

            # Optional: warn if attached tool has no edges referencing it
            attached = set(n.data.tools.attached)
            referenced = {e.from_ for e in cfg.edges} | {e.to for e in cfg.edges}
            for tid in attached:
                if tid in tool_ids and tid not in referenced:
                    issues.append(
                        ValidationIssue(
                            code="tools.attachment_unused",
                            message=f"Attached tool {tid} is never referenced by edges",
                            path=["nodes", n.id, "tools", "attached"],
                            severity="WARNING",
                        )
                    )

    return ValidationReport(issues=issues)


__all__ = ["validate_graph"]
