from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProviderError(Exception):
    message: str
    status_code: int | None = None
    code: str | None = None

    def __str__(self) -> str:  # pragma: no cover - trivial
        prefix = f"[{self.status_code}] " if self.status_code is not None else ""
        code = f" ({self.code})" if self.code else ""
        return f"{prefix}{self.message}{code}"
