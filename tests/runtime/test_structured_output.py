from __future__ import annotations

import json
from typing import Any
from collections.abc import Iterator

from codeless_orchestrator.providers.base import LLMProvider
from codeless_orchestrator.providers.types import ChatChunk, ChatMessage, ChatRequest, ChatResponse, Usage
from codeless_orchestrator.runtime.structured_output import (
    ensure_structured_output,
    post_process_text,
)


class StubProvider(LLMProvider):
    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[ChatRequest] = []

    def chat(self, req: ChatRequest) -> ChatResponse:
        self.calls.append(req)
        text = self.replies.pop(0) if self.replies else "{}"
        return ChatResponse(message=ChatMessage(role="assistant", content=text), finish_reason="stop", usage=Usage())

    def stream(self, req: ChatRequest) -> Iterator[ChatChunk]:  # pragma: no cover - not used
        yield from ()


def _policy(schema: dict[str, Any] | None = None, on="RetryAndRepair", max_rep=2):
    return type(
        "P", (),
        {
            "enabled": True,
            "schema_": schema or {},
            "on_violation": on,
            "max_repair_attempts": max_rep,
            "post_process": type("PP", (), {"normalize_whitespace": True, "ensure_markdown": True})(),
        },
    )()


def test_valid_json_no_repair() -> None:
    provider = StubProvider([json.dumps({"answer": "yes"})])
    req = ChatRequest(model="gpt", messages=[ChatMessage(role="user", content="hi")])
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"], "additionalProperties": False}
    msg, usage, finish, attempts = ensure_structured_output(provider, req, policy_cfg=_policy(schema), post_cfg=None)
    assert attempts == 1
    assert isinstance(msg.content, str)
    # Compact
    assert msg.content == "{" + '"answer":"yes"' + "}"


def test_invalid_then_repair_succeeds() -> None:
    bad = json.dumps({})
    good = json.dumps({"answer": "ok"})
    provider = StubProvider([bad, good])
    req = ChatRequest(model="gpt", messages=[ChatMessage(role="user", content="q")])
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"], "additionalProperties": False}
    msg, _, _, attempts = ensure_structured_output(provider, req, policy_cfg=_policy(schema), post_cfg=None)
    assert attempts == 2
    assert msg.content == "{" + '"answer":"ok"' + "}"


def test_on_violation_refuse_raises() -> None:
    provider = StubProvider(["not json"])
    req = ChatRequest(model="gpt", messages=[ChatMessage(role="user", content="q")])
    try:
        ensure_structured_output(provider, req, policy_cfg=_policy({}, on="Refuse", max_rep=0), post_cfg=None)
        assert False, "expected exception"
    except ValueError:
        pass


def test_post_process_text() -> None:
    txt = "Hello\n\n   world  !"
    out = post_process_text(txt, ensure_markdown=True, normalize_ws=True)
    assert out.startswith("```") and "Hello world !" in out
