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


class RouterProvider(LLMProvider):
    def __init__(self, route_to: str) -> None:
        self.route_to = route_to

    def chat(self, req: ChatRequest) -> ChatResponse:
        # Router calls are made with json_mode_enabled=True; respond with JSON string
        if req.json_mode_enabled:
            return ChatResponse(message=ChatMessage(role="assistant", content=json.dumps({"next_agent_id": self.route_to})), finish_reason="stop")
        # Normal agent A content (not used)
        return ChatResponse(message=ChatMessage(role="assistant", content="A spoke"), finish_reason="stop")
    def stream(self, req: ChatRequest):  # pragma: no cover - not used
        yield from ()


class ProviderB(LLMProvider):
    def chat(self, req: ChatRequest) -> ChatResponse:
        return ChatResponse(message=ChatMessage(role="assistant", content="B ok"), finish_reason="stop")
    def stream(self, req: ChatRequest):  # pragma: no cover
        yield from ()


class ProviderC(LLMProvider):
    def chat(self, req: ChatRequest) -> ChatResponse:
        return ChatResponse(message=ChatMessage(role="assistant", content="C ok"), finish_reason="stop")
    def stream(self, req: ChatRequest):  # pragma: no cover
        yield from ()


def _load_cfg() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    return json.loads((root / "examples" / "configs" / "handoff_branch.json").read_text(encoding="utf-8"))


def test_handoff_router_selects_C() -> None:
    cfg = _load_cfg()

    def resolver(mp: Any) -> LLMProvider:
        mid = getattr(mp, "model_id")
        if mid == "model-a":
            return RouterProvider("C")
        if mid == "model-b":
            return ProviderB()
        return ProviderC()

    client = TestClient(app)
    app.dependency_overrides[deps.get_provider_resolver] = lambda: resolver
    try:
        body = {"config": cfg, "input": {"text": "go"}}
        resp = client.post("/execute", json=body)
        assert resp.status_code == 200
        data = resp.json()
        texts = [m.get("content") for m in data.get("messages", []) if m.get("role") == "assistant"]
        assert any(t == "C ok" for t in texts)
        assert not any(t == "B ok" for t in texts)
    finally:
        app.dependency_overrides.pop(deps.get_provider_resolver, None)


def test_handoff_router_invalid_falls_back_first(caplog: pytest.LogCaptureFixture) -> None:
    cfg = _load_cfg()

    def resolver(mp: Any) -> LLMProvider:
        mid = getattr(mp, "model_id")
        if mid == "model-a":
            return RouterProvider("Z")  # invalid id
        if mid == "model-b":
            return ProviderB()
        return ProviderC()

    client = TestClient(app)
    app.dependency_overrides[deps.get_provider_resolver] = lambda: resolver
    try:
        with caplog.at_level("WARNING"):
            resp = client.post("/execute", json={"config": cfg, "input": {"text": "go"}})
        assert resp.status_code == 200
        data = resp.json()
        texts = [m.get("content") for m in data.get("messages", []) if m.get("role") == "assistant"]
        # Should fall back to first candidate (B)
        assert any(t == "B ok" for t in texts)
        assert any("Router returned invalid id" in r.message or "Router parse failed" in r.message for r in caplog.records)
    finally:
        app.dependency_overrides.pop(deps.get_provider_resolver, None)
