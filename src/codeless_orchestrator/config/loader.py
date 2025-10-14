from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ..validation.models import ValidationIssue, ValidationReport
from ..validation.validator import validate_graph
from .schema import GraphConfig


def _summarize_pydantic_error(err: ValidationError) -> str:
    parts: list[str] = []
    for e in err.errors():
        loc = ".".join(str(x) for x in e.get("loc", []))
        msg = e.get("msg", "validation error")
        parts.append(f"{loc}: {msg}")
    return "; ".join(parts) if parts else str(err)


def load_graph(
    data: dict[str, Any] | str | Path, *, validate: bool = True
) -> tuple[GraphConfig, ValidationReport]:
    """Load a graph config from a dict or JSON file path.

    - Parses using Pydantic with aliases to accept camelCase/snake_case.
    - Runs cross-reference validation and returns a ValidationReport.
    - If ``validate`` is True and errors exist, raises ValueError.
    """

    raw: dict[str, Any]
    if isinstance(data, (str, Path)):
        with open(Path(data), encoding="utf-8") as f:
            raw = json.load(f)
    else:
        raw = data

    try:
        cfg = GraphConfig.model_validate(raw)
    except ValidationError as e:  # normalize to ValueError as requested
        raise ValueError(_summarize_pydantic_error(e)) from e

    report = validate_graph(cfg)
    if validate and report.has_errors():
        msg = "; ".join(
            f"{iss.code}: {iss.message}"
            for iss in report.issues
            if iss.severity == "ERROR"
        )
        raise ValueError(msg)

    return cfg, report


def to_json_schema() -> dict[str, Any]:
    """Return the JSON Schema for the GraphConfig model."""
    return GraphConfig.model_json_schema()


__all__ = ["load_graph", "to_json_schema", "GraphConfig", "ValidationReport", "ValidationIssue"]
