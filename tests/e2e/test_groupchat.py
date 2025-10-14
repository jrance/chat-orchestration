from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from codeless_orchestrator.providers.base import LLMProvider
from codeless_orchestrator.providers.types import ChatMessage, ChatRequest, ChatResponse
from codeless_orchestrator.server import deps
from codeless_orchestrator.server.app import app


class ModeratorProvider(LLMProvider):
    def __init__(self, decisions: list[dict[str, Any]]) -> None:
        self.decisions = list(decisions)

    def chat(self, req: ChatRequest) -> ChatResponse:
        # Moderator is called to pick next speaker; return JSON string
        if not self.decisions:
            payload = {"end": True}
        else:
            payload = self.decisions.pop(0)
        return ChatResponse(message=ChatMessage(role="assistant", content=json.dumps(payload)), finish_reason="stop")
    def stream(self, req: ChatRequest):  # pragma: no cover - not used in this test
        yield from ()


class ProviderX(LLMProvider):
    def chat(self, req: ChatRequest) -> ChatResponse:
        return ChatResponse(message=ChatMessage(role="assistant", content="X says hi"), finish_reason="stop")
    def stream(self, req: ChatRequest):  # pragma: no cover - not used
        yield from ()


class ProviderY(LLMProvider):
    def chat(self, req: ChatRequest) -> ChatResponse:
        return ChatResponse(message=ChatMessage(role="assistant", content="Y says hi"), finish_reason="stop")
    def stream(self, req: ChatRequest):  # pragma: no cover - not used
        yield from ()


def _load_cfg() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    return json.loads((root / "examples" / "configs" / "groupchat_moderator.json").read_text(encoding="utf-8"))


def test_groupchat_moderator_turn_order() -> None:
    cfg = _load_cfg()

    moderator = ModeratorProvider(decisions=[{"next": "Y"}, {"next": "X"}, {"end": True}])

    def resolver(mp: Any) -> LLMProvider:
        mid = getattr(mp, "model_id")
        if mid == "router-model":
            return moderator
        if mid == "model-x":
            return ProviderX()
        return ProviderY()

    client = TestClient(app)
    app.dependency_overrides[deps.get_provider_resolver] = lambda: resolver
    try:
        resp = client.post("/execute", json={"config": cfg, "input": {"text": "go"}})
        assert resp.status_code == 200
        data = resp.json()
        texts = [m.get("content") for m in data.get("messages", []) if m.get("role") == "assistant"]
        # Messages include attribution prefix like "[Agent: Agent Y] Y says hi"
        assert any(isinstance(t, str) and t.startswith("[Agent: Agent Y]") for t in texts)
        assert any(isinstance(t, str) and t.startswith("[Agent: Agent X]") for t in texts)
    finally:
        app.dependency_overrides.pop(deps.get_provider_resolver, None)


def test_groupchat_moderator_max_turns_fence(caplog: pytest.LogCaptureFixture) -> None:
    cfg = _load_cfg()

    # Generate many "next" without end; engine should stop at default max turns
    moderator = ModeratorProvider(decisions=[{"next": "Y"}] * 10)

    def resolver(mp: Any) -> LLMProvider:
        mid = getattr(mp, "model_id")
        if mid == "router-model":
            return moderator
        if mid == "model-x":
            return ProviderX()
        return ProviderY()

    client = TestClient(app)
    app.dependency_overrides[deps.get_provider_resolver] = lambda: resolver
    try:
        with caplog.at_level("WARNING"):
            resp = client.post("/execute", json={"config": cfg, "input": {"text": "go"}})
        assert resp.status_code == 200
        data = resp.json()
        texts = [m.get("content") for m in data.get("messages", []) if m.get("role") == "assistant" and isinstance(m.get("content"), str)]
        # Default cap is 6 turns; assistant-visible messages should not exceed that by much
        assert len(texts) <= 6
    finally:
        app.dependency_overrides.pop(deps.get_provider_resolver, None)
