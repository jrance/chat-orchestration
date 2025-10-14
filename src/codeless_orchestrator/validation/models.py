from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Severity = Literal["ERROR", "WARNING"]


@dataclass(slots=True, frozen=True)
class ValidationIssue:
    code: str
    message: str
    path: list[str]
    severity: Severity


@dataclass(slots=True)
class ValidationReport:
    issues: list[ValidationIssue]

    def has_errors(self) -> bool:  # pragma: no cover - trivial
        return any(i.severity == "ERROR" for i in self.issues)


__all__ = ["Severity", "ValidationIssue", "ValidationReport"]
