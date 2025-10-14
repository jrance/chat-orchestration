from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config.schema import AgentNodeData, SafetyConfig
from ..providers.types import ChatMessage
from .policies import SafetyPolicy, get_policy
from .safety import make_refusal_reply, redact_message, redact_text, scan_injection_score


@dataclass
class FilterResult:
    blocked: bool
    refusal_message: ChatMessage | None
    redacted_messages: list[ChatMessage]
    warnings: list[str]


def _should_redact(safety_cfg: SafetyConfig, policy: SafetyPolicy) -> bool:
    return bool(getattr(safety_cfg, "pii_redaction", policy.redact_pii_default))


def pre_provider_filter(
    messages: list[ChatMessage],
    *,
    safety_cfg: SafetyConfig,
    policy: SafetyPolicy | None = None,
) -> FilterResult:
    pol = policy or get_policy(getattr(safety_cfg, "policy_ref", None))
    warnings: list[str] = []

    # Injection defense: score recent user messages (simple: last one)
    if getattr(safety_cfg, "prompt_injection_defense", False) and messages:
        last = messages[-1]
        text = last.content if isinstance(last.content, str) else ""
        score = scan_injection_score(text)
        threshold = pol.injection_threshold
        if score >= threshold:
            if getattr(safety_cfg, "on_block", "Refuse") == "Warn":
                warnings.append("injection_detected")
            else:
                # Refuse
                refusal = ChatMessage(role="assistant", content=make_refusal_reply())
                return FilterResult(
                    blocked=True,
                    refusal_message=refusal,
                    redacted_messages=[refusal],
                    warnings=warnings,
                )

    # PII redaction for provider view only
    if _should_redact(safety_cfg, pol):
        red = [redact_message(m) for m in messages]
    else:
        red = list(messages)

    return FilterResult(blocked=False, refusal_message=None, redacted_messages=red, warnings=warnings)


def post_provider_filter(
    assistant_message: ChatMessage,
    *,
    safety_cfg: SafetyConfig,
    policy: SafetyPolicy | None = None,
) -> ChatMessage:
    pol = policy or get_policy(getattr(safety_cfg, "policy_ref", None))
    if _should_redact(safety_cfg, pol) and isinstance(assistant_message.content, str):
        return ChatMessage(
            role=assistant_message.role,
            content=redact_text(assistant_message.content),
            tool_calls=assistant_message.tool_calls,
            tool_call_id=assistant_message.tool_call_id,
        )
    return assistant_message


__all__ = ["FilterResult", "pre_provider_filter", "post_provider_filter"]

