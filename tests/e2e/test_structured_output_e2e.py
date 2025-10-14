from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from codeless_orchestrator.providers.base import LLMProvider
from codeless_orchestrator.providers.types import ChatMessage, ChatRequest, ChatResponse
from codeless_orchestrator.server import deps
from codeless_orchestrator.server.app import app


class FlakyJSONProvider(LLMProvider):
    def __init__(self) -> None:
        self.turn = 0

    def chat(self, req: ChatRequest) -> ChatResponse:
        self.turn += 1
        if self.turn == 1:
            # Invalid JSON first
            return ChatResponse(message=ChatMessage(role="assistant", content="not-json"), finish_reason="stop")
        # Then valid JSON
        return ChatResponse(message=ChatMessage(role="assistant", content=json.dumps({"answer": "ok"})), finish_reason="stop")
    def stream(self, req: ChatRequest):  # pragma: no cover - not used
        yield from ()


def test_structured_output_repair_nonstream() -> None:
    root = Path(__file__).resolve().parents[2]
    cfg_path = root / "examples" / "configs" / "structured_output.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    client = TestClient(app)
    app.dependency_overrides[deps.get_provider_resolver] = lambda: (lambda _: FlakyJSONProvider())
    try:
        resp = client.post("/execute", json={"config": cfg, "input": {"text": "q"}})
        assert resp.status_code == 200
        data = resp.json()
        last = data["messages"][-1]
        assert last["role"] == "assistant"
        # Compact JSON string content
        assert last["content"] == "{" + '"answer":"ok"' + "}"
    finally:
        app.dependency_overrides.pop(deps.get_provider_resolver, None)

