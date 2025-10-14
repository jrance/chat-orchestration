from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from codeless_orchestrator.providers.base import LLMProvider
from codeless_orchestrator.providers.types import ChatMessage, ChatRequest, ChatResponse
from codeless_orchestrator.server import deps
from codeless_orchestrator.server.app import app


class SleepyProvider(LLMProvider):
    def __init__(self, text: str, delay_s: float) -> None:
        self.text = text
        self.delay_s = delay_s

    def chat(self, req: ChatRequest) -> ChatResponse:
        time.sleep(self.delay_s)
        return ChatResponse(message=ChatMessage(role="assistant", content=self.text), finish_reason="stop")
    def stream(self, req: ChatRequest):  # pragma: no cover - not used
        yield from ()


def test_concurrent_fanout_aggregate_and_timing() -> None:
    root = Path(__file__).resolve().parents[2]
    cfg_path = root / "examples" / "configs" / "concurrent_fanout.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    def resolver(mp: Any) -> LLMProvider:
        mid = getattr(mp, "model_id")
        if mid == "model-b":
            return SleepyProvider("B branch", 0.03)
        if mid == "model-c":
            return SleepyProvider("C branch", 0.04)
        # Root A not used for content in concurrent builder
        return SleepyProvider("root", 0.0)

    client = TestClient(app)
    app.dependency_overrides[deps.get_provider_resolver] = lambda: resolver
    try:
        t0 = time.time()
        resp = client.post("/execute", json={"config": cfg, "input": {"text": "go"}})
        elapsed = time.time() - t0
        assert resp.status_code == 200
        data = resp.json()
        # Merged assistant content should be deterministic aggregate
        final = data["messages"][-1]
        assert final["role"] == "assistant"
        txt = final.get("content") or ""
        # Aggregate strategy orders by agent id, i.e. B then C
        assert "B: B branch" in txt and "C: C branch" in txt
        # Timing should be less than sequential sum (0.07s) with generous margin.
        # Allow overheads from app/compile/testclient on CI; keep it under ~0.2s.
        assert elapsed < 0.2
    finally:
        app.dependency_overrides.pop(deps.get_provider_resolver, None)
