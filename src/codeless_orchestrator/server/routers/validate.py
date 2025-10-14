from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Depends, Query

from ...config.loader import load_graph
from ..schemas import ValidateResponse

router = APIRouter(prefix="", tags=["validate"])


@router.post("/validate", response_model=ValidateResponse)
def validate_config(
    config: dict[str, Any] = Body(..., description="Graph configuration JSON"),
    return_normalized: bool = Query(False, alias="returnNormalized"),
) -> ValidateResponse:
    try:
        cfg, report = load_graph(config, validate=False)
    except Exception as e:
        # If parsing itself fails, surface as a single error
        return ValidateResponse(
            valid=False,
            errors=[{"code": "parse.error", "message": str(e), "path": [], "severity": "ERROR"}],
            warnings=[],
            normalized=None,
        )

    errors = [
        {"code": i.code, "message": i.message, "path": i.path, "severity": i.severity}
        for i in report.issues
        if i.severity == "ERROR"
    ]
    warnings = [
        {"code": i.code, "message": i.message, "path": i.path, "severity": i.severity}
        for i in report.issues
        if i.severity == "WARNING"
    ]

    normalized = cfg.model_dump(by_alias=True) if return_normalized else None
    return ValidateResponse(valid=len(errors) == 0, errors=errors, warnings=warnings, normalized=normalized)

