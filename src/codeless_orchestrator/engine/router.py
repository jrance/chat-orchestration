from __future__ import annotations

import json
from collections.abc import Iterable

from ..logging import get_logger
from ..providers.base import LLMProvider
from ..providers.types import ChatMessage, ChatRequest

_log = get_logger(__name__)


def _build_prompt(current_agent: str, candidates: list[str]) -> list[ChatMessage]:
    candidates_json = json.dumps(candidates)
    system = (
        "You are a router. Choose the best next agent id from the candidates.\n"
        "Return ONLY the id in JSON as {\"next_agent_id\": \"<id>\"}.\n"
        "If unsure, choose the first candidate."
    )
    user = (
        f"Current agent id: {current_agent}\n"
        f"Candidates (ordered by priority): {candidates_json}\n"
        "Choose exactly one id from the list."
    )
    return [ChatMessage(role="system", content=system), ChatMessage(role="user", content=user)]


def _parse_choice(text_or_json: str, candidates: list[str]) -> str | None:
    # Try JSON
    try:
        obj = json.loads(text_or_json)
        if isinstance(obj, dict):
            val = obj.get("next_agent_id")
            if isinstance(val, str) and val in candidates:
                return val
    except Exception:
        pass
    # Fallback: search for a candidate id as substring
    lower = text_or_json.strip().lower()
    for c in candidates:
        if c.lower() in lower:
            return c
    return None


def choose_next_agent(
    *,
    messages: list[ChatMessage],
    current_agent: str,
    candidates: Iterable[str],
    provider: LLMProvider,
) -> str:
    """Choose the next agent id using the given provider.

    Robust to non-JSON responses; falls back deterministically to the first
    candidate and logs a warning.
    """
    cand_list = list(candidates)
    if not cand_list:
        raise ValueError("No candidates provided for routing")

    # Build a low-temperature classification prompt
    prompt_msgs = _build_prompt(current_agent, cand_list)

    # Prefer JSON mode if supported; we cannot detect provider support, so we just set the flag
    req = ChatRequest(
        model="",
        messages=prompt_msgs + messages[-5:],  # include a short tail of context
        tools=None,
        tool_choice="none",
        temperature=0.0,
        top_p=1.0,
        max_tokens=64,
        stop=None,
        seed=None,
        json_mode_enabled=True,
        response_format_schema=None,
        metadata=None,
        timeout=10.0,
    )
    try:
        resp = provider.chat(req)
        content = resp.message.content
        text = content if isinstance(content, str) else ""
        choice = _parse_choice(text, cand_list)
        if choice is None:
            _log.warning(
                "Router parse failed; defaulting to first candidate",
            )
            return cand_list[0]
        return choice
    except Exception as e:  # pragma: no cover - provider error paths
        _log.warning("Router provider failed (%s); defaulting to first candidate", e)
        return cand_list[0]


__all__ = ["choose_next_agent"]
