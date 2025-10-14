from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from fastapi.testclient import TestClient

from codeless_orchestrator.providers.base import LLMProvider
from codeless_orchestrator.providers.types import ChatChunk, ChatMessage, ChatRequest, ChatResponse, Usage
from codeless_orchestrator.server import deps
from codeless_orchestrator.server.app import app


class ProviderWithUsage(LLMProvider):
    def chat(self, req: ChatRequest) -> ChatResponse:
        return ChatResponse(
            message=ChatMessage(role="assistant", content="Hello"),
            finish_reason="stop",
            usage=Usage(prompt_tokens=2, completion_tokens=2, total_tokens=4),
        )

    def stream(self, req: ChatRequest) -> Iterator[ChatChunk]:
        yield ChatChunk(type="role", role="assistant")
        yield ChatChunk(type="content.delta", text_delta="Hello")
        yield ChatChunk(type="message.end", finish_reason="stop")


def _cfg_single() -> dict[str, Any]:
    # Load any single-agent config; structured or not both fine
    root = Path(__file__).resolve().parents[2]
    cfg_path = root / "examples" / "configs" / "single_agent_with_tool.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    # Remove attached tools to force simple stream path
    for n in cfg.get("nodes", []):
        if isinstance(n, dict) and n.get("kind") == "agent.codeless":
            if "tools" in n.get("data", {}):
                n["data"]["tools"]["attached"] = []
    return cfg


def test_nonstream_includes_metrics() -> None:
    cfg = _cfg_single()
    client = TestClient(app)
    app.dependency_overrides[deps.get_provider_resolver] = lambda: (lambda _: ProviderWithUsage())
    try:
        resp = client.post("/execute", json={"config": cfg, "input": {"text": "hello"}})
        assert resp.status_code == 200
        data = resp.json()
        m = data.get("metrics")
        assert isinstance(m, dict)
        assert m.get("token_usage", {}).get("total") == 4
        assert isinstance(m.get("steps"), int)
    finally:
        app.dependency_overrides.pop(deps.get_provider_resolver, None)


def test_stream_message_end_carries_metrics() -> None:
    cfg = _cfg_single()
    client = TestClient(app)
    app.dependency_overrides[deps.get_provider_resolver] = lambda: (lambda _: ProviderWithUsage())
    try:
        events: list[dict] = []
        with client.stream("POST", "/execute/stream", json={"config": cfg, "input": {"text": "hi"}}) as resp:
            assert resp.status_code == 200
            for line in resp.iter_lines():
                if not line:
                    continue
                s = line.decode("utf-8") if isinstance(line, (bytes, bytearray)) else str(line)
                if not s.startswith("data: "):
                    continue
                payload = s[len("data: ") :]
                obj = json.loads(payload)
                events.append(obj)
                if obj.get("type") == "end":
                    break
        mend = next(e for e in events if e.get("type") == "message.end")
        assert mend.get("metrics") is not None
        assert all("request_id" in e for e in events)
    finally:
        app.dependency_overrides.pop(deps.get_provider_resolver, None)

