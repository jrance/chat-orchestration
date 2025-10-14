from __future__ import annotations

from codeless_orchestrator.providers.types import ChatMessage
from codeless_orchestrator.runtime.policies import get_policy
from codeless_orchestrator.runtime.filters import pre_provider_filter, post_provider_filter
from codeless_orchestrator.runtime.safety import redact_text


def _safety_cfg(enabled: bool = True):
    return type(
        "S", (),
        {
            "policy_ref": "enterprise-v3",
            "pii_redaction": enabled,
            "prompt_injection_defense": False,
            "on_block": "Refuse",
        },
    )()


def test_redact_email_phone_cc():
    text = "Contact me at john.doe@example.com or +1 (555) 123-4567. Card 4111 1111 1111 1111"
    msgs = [ChatMessage(role="user", content=text)]
    pol = get_policy("enterprise-v3")
    pre = pre_provider_filter(msgs, safety_cfg=_safety_cfg(True), policy=pol)
    assert "[REDACTED:EMAIL]" in pre.redacted_messages[0].content
    assert "[REDACTED:PHONE]" in pre.redacted_messages[0].content
    # Luhn redaction
    assert "[REDACTED:CC]" in pre.redacted_messages[0].content
    # Post redaction for assistant
    assistant = post_provider_filter(ChatMessage(role="assistant", content=text), safety_cfg=_safety_cfg(True), policy=pol)
    assert "[REDACTED:EMAIL]" in assistant.content


def test_no_redaction_when_disabled():
    txt = "john@example.com"
    pre = pre_provider_filter([ChatMessage(role="user", content=txt)], safety_cfg=_safety_cfg(False), policy=get_policy("enterprise-v3"))
    assert pre.redacted_messages[0].content == txt
    out = post_provider_filter(ChatMessage(role="assistant", content=txt), safety_cfg=_safety_cfg(False), policy=get_policy("enterprise-v3"))
    assert out.content == txt

