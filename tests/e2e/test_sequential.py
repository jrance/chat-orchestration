from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from codeless_orchestrator.providers.base import LLMProvider
from codeless_orchestrator.providers.types import ChatMessage, ChatRequest, ChatResponse
from codeless_orchestrator.server import deps
from codeless_orchestrator.server.app import app


class ProviderA(LLMProvider):
    def chat(self, req: ChatRequest) -> ChatResponse:
        return ChatResponse(message=ChatMessage(role="assistant", content="A ok"), finish_reason="stop")

    def stream(self, req: ChatRequest):  # pragma: no cover - not used
        yield from ()


class ProviderB(LLMProvider):
    def chat(self, req: ChatRequest) -> ChatResponse:
        return ChatResponse(message=ChatMessage(role="assistant", content="B ok"), finish_reason="stop")

    def stream(self, req: ChatRequest):  # pragma: no cover - not used
        yield from ()


def test_sequential_two_agents() -> None:
    root = Path(__file__).resolve().parents[2]
    cfg_path = root / "examples" / "configs" / "sequential_two_agents.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    def resolver(mp: Any) -> LLMProvider:
        mid = getattr(mp, "model_id")
        return ProviderA() if mid == "model-a" else ProviderB()

    client = TestClient(app)
    app.dependency_overrides[deps.get_provider_resolver] = lambda: resolver
    try:
        body = {"config": cfg, "input": {"text": "go"}}
        resp = client.post("/execute", json=body)
        assert resp.status_code == 200
        data = resp.json()
        steps = data.get("steps")
        assert isinstance(steps, int) and steps >= 2
        texts = [m.get("content") for m in data.get("messages", []) if m.get("role") == "assistant"]
        assert any(t == "A ok" for t in texts)
        assert any(t == "B ok" for t in texts)
    finally:
        app.dependency_overrides.pop(deps.get_provider_resolver, None)

