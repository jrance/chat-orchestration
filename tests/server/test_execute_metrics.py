from __future__ import annotations

import json
from typing import Any, Iterator

from fastapi.testclient import TestClient

from codeless_orchestrator.providers.base import LLMProvider
from codeless_orchestrator.providers.types import ChatChunk, ChatMessage, ChatRequest, ChatResponse, Usage
from codeless_orchestrator.server import deps
from codeless_orchestrator.server.app import app


class StubProviderWithUsage(LLMProvider):
    def chat(self, req: ChatRequest) -> ChatResponse:
        return ChatResponse(
            message=ChatMessage(role="assistant", content="Hello"),
            finish_reason="stop",
            usage=Usage(prompt_tokens=3, completion_tokens=2, total_tokens=5),
        )

    def stream(self, req: ChatRequest) -> Iterator[ChatChunk]:  # pragma: no cover - not used
        yield from ()


def _cfg() -> dict[str, Any]:
    return {
        "meta": {
            "id": "g1",
            "name": "Sample",
            "version": "1",
            "metadata": {"showToolNodesOnCanvas": True},
        },
        "nodes": [
            {
                "id": "agent-1",
                "kind": "agent.codeless",
                "label": "Agent",
                "data": {
                    "systemInstructions": "",
                    "styleGuide": "",
                    "model": {"provider": "openai", "modelId": "gpt-4o-mini"},
                    "context": {
                        "historyWindow": {"mode": "LastN", "n": 5},
                        "injectOrgPreamble": True,
                        "vars": {},
                    },
                    "tools": {
                        "policy": "None",
                        "attached": [],
                        "timeoutMs": 10000,
                        "maxCallsPerTurn": 0,
                        "parallelism": 1,
                        "redactPII": True,
                    },
                    "structuredOutput": {"enabled": False, "schema": {}},
                    "safety": {"policyRef": "default"},
                    "telemetry": {"labels": {"env": "test"}, "emitUsage": True},
                },
            }
        ],
        "edges": [],
    }


def test_execute_includes_metrics_when_emitUsage() -> None:
    client = TestClient(app)
    app.dependency_overrides[deps.get_provider_resolver] = lambda: (lambda _: StubProviderWithUsage())
    try:
        body = {"config": _cfg(), "input": {"text": "hello"}}
        resp = client.post("/execute", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("metrics") is not None
        m = data["metrics"]
        assert m.get("token_usage") is not None
        assert m["token_usage"].get("total") == 5
        assert isinstance(m.get("steps"), int)
    finally:
        app.dependency_overrides.pop(deps.get_provider_resolver, None)


class StubStreamProvider(LLMProvider):
    def chat(self, req: ChatRequest) -> ChatResponse:  # pragma: no cover - not used
        raise RuntimeError("not used")

    def stream(self, req: ChatRequest) -> Iterator[ChatChunk]:
        yield ChatChunk(type="role", role="assistant")
        yield ChatChunk(type="content.delta", text_delta="Hello")
        yield ChatChunk(type="message.end", finish_reason="stop")


def test_stream_message_end_has_metrics() -> None:
    client = TestClient(app)
    app.dependency_overrides[deps.get_provider_resolver] = lambda: (lambda _: StubStreamProvider())
    try:
        body = {"config": _cfg(), "input": {"text": "hello"}}
        events: list[dict] = []
        with client.stream("POST", "/execute/stream", json=body) as resp:
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
        # The message.end event should contain metrics
        end_msg = next((e for e in events if e.get("type") == "message.end"), None)
        assert end_msg is not None
        assert end_msg.get("metrics") is not None
        # All events include request_id
        assert all("request_id" in e for e in events)
    finally:
        app.dependency_overrides.pop(deps.get_provider_resolver, None)

