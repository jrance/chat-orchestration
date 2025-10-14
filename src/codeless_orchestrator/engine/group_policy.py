from __future__ import annotations

import json
from collections.abc import Iterable

from ..logging import get_logger
from ..providers.base import LLMProvider
from ..providers.types import ChatMessage, ChatRequest

_log = get_logger(__name__)


def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```") and t.endswith("```"):
        # remove first and last fence lines
        lines = t.splitlines()
        if len(lines) >= 2:
            # drop first and last line
            return "\n".join(lines[1:-1]).strip()
    return t


def _build_moderator_prompt(participants: list[str]) -> list[ChatMessage]:
    plist = json.dumps(participants)
    system = (
        "You are a moderator routing the next speaker in a group chat.\n"
        "Reply in strict JSON. Either choose a next speaker id from the provided list, or end.\n"
        "Valid shapes: {\"next\": \"<id>\"} OR {\"end\": true}.\n"
        "Do not include any other text."
    )
    user = (
        f"Eligible next speakers (ordered): {plist}\n"
        "Pick exactly one id or end."
    )
    return [ChatMessage(role="system", content=system), ChatMessage(role="user", content=user)]


def _parse_moderator_choice(text_or_json: str, participants: list[str]) -> tuple[str | None, bool]:
    # Remove code fences and try JSON
    raw = _strip_code_fences(text_or_json)
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            if obj.get("end") is True:
                return None, True
            nxt = obj.get("next")
            if isinstance(nxt, str):
                if nxt in participants:
                    return nxt, False
                # Explicit invalid id
                _log.warning("Moderator returned invalid id '%s'; defaulting to first", nxt)
                return participants[0], False
    except Exception:
        pass
    # Fallback: search for any participant id mention
    lower = raw.lower()
    for pid in participants:
        if pid.lower() in lower:
            return pid, False
    return None, False


def choose_next_speaker(
    messages: list[ChatMessage],
    participants: Iterable[str],
    current_speaker: str | None,
    moderator_provider: LLMProvider,
) -> tuple[str | None, bool]:
    """Ask the moderator model to select the next speaker.

    Returns a tuple (next_id | None, end: bool). On parsing/provider failure
    or invalid id, falls back deterministically to the first participant.
    """
    plist = list(participants)
    if not plist:
        return None, True

    prompt = _build_moderator_prompt(plist)
    req = ChatRequest(
        model="",
        messages=prompt + messages[-6:],
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
        resp = moderator_provider.chat(req)
        content = resp.message.content
        text = content if isinstance(content, str) else ""
        choice, end = _parse_moderator_choice(text, plist)
        if end:
            return None, True
        if choice is None or choice not in plist:
            _log.warning("Moderator parse failed; defaulting to first participant")
            return plist[0], False
        return choice, False
    except Exception as e:  # pragma: no cover - provider errors
        _log.warning("Moderator provider failed (%s); defaulting to first participant", e)
        return plist[0], False


def pick_round_robin(participants: list[str], current_speaker: str | None) -> str | None:
    if not participants:
        return None
    if current_speaker is None:
        return participants[0]
    try:
        idx = participants.index(current_speaker)
    except ValueError:
        return participants[0]
    nxt = (idx + 1) % len(participants)
    return participants[nxt]


__all__ = ["choose_next_speaker", "pick_round_robin"]
