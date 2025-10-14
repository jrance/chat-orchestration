from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ..providers.types import ChatMessage


# PII regexes (conservative)
RE_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
RE_PHONE = re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}\b")
RE_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
RE_IPV4 = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b")


def _luhn_check(num: str) -> bool:
    s = num.replace(" ", "").replace("-", "")
    if not s.isdigit() or len(s) < 12 or len(s) > 19:
        return False
    total = 0
    reverse = list(map(int, s[::-1]))
    for i, d in enumerate(reverse):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


RE_CC_CANDIDATE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")


def redact_text(text: str) -> str:
    if not text:
        return text
    out = text
    out = RE_EMAIL.sub("[REDACTED:EMAIL]", out)
    out = RE_PHONE.sub("[REDACTED:PHONE]", out)
    out = RE_SSN.sub("[REDACTED:SSN]", out)
    out = RE_IPV4.sub("[REDACTED:IP]", out)
    # Credit card via Luhn check
    def _repl_cc(m: re.Match[str]) -> str:
        s = m.group(0)
        return "[REDACTED:CC]" if _luhn_check(s) else s

    out = RE_CC_CANDIDATE.sub(_repl_cc, out)
    return out


# Prompt-injection heuristics
INJECTION_KEYWORDS = [
    "ignore previous instructions",
    "as system",
    "developer mode",
    "bypass",
    "disregard",
    "print system prompt",
    "reveal chain of thought",
    "begin system prompt",
    "end system prompt",
]

RE_IMPERATIVES = re.compile(r"\b(ignore|bypass|disregard|override|reveal|print)\b", re.I)


def scan_injection_score(text: str) -> float:
    if not text:
        return 0.0
    t = text.lower()
    score = 0.0
    for kw in INJECTION_KEYWORDS:
        if kw in t:
            score += 0.4
    if RE_IMPERATIVES.search(t):
        score += 0.2
    return min(score, 1.0)


def make_refusal_reply() -> str:
    return (
        "I can’t comply with that request. For your safety, I won’t share or "
        "act on instructions that attempt to override system policies."
    )


def redact_message(message: ChatMessage) -> ChatMessage:
    if isinstance(message.content, str):
        return ChatMessage(
            role=message.role,
            content=redact_text(message.content),
            tool_call_id=message.tool_call_id,
            tool_calls=message.tool_calls,
        )
    return message


__all__ = [
    "redact_text",
    "scan_injection_score",
    "make_refusal_reply",
    "redact_message",
]

