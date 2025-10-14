from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SafetyPolicy:
    name: str
    injection_threshold: float
    redact_pii_default: bool


def get_policy(policy_ref: str | None) -> SafetyPolicy:
    # Single default policy; hook for extensions
    name = policy_ref or "enterprise-v3"
    if name == "enterprise-v3":
        return SafetyPolicy(name=name, injection_threshold=0.7, redact_pii_default=True)
    # Fallback safe defaults
    return SafetyPolicy(name=name, injection_threshold=0.7, redact_pii_default=True)


__all__ = ["SafetyPolicy", "get_policy"]

