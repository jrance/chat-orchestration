from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from codeless_orchestrator.providers.base import LLMProvider
from codeless_orchestrator.providers.types import ChatMessage, ChatRequest, ChatResponse
from codeless_orchestrator.server import deps
from codeless_orchestrator.server.app import app


class StubProvider(LLMProvider):
    def chat(self, req: ChatRequest) -> ChatResponse:
        return ChatResponse(message=ChatMessage(role="assistant", content="Hello"), finish_reason="stop")

    def stream(self, req: ChatRequest):  # pragma: no cover - not used here
        yield from ()


def _valid_config() -> dict:
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
                    "telemetry": {"labels": {}, "emitUsage": True},
                },
            }
        ],
        "edges": [],
    }


def test_execute_nonstream_basic() -> None:
    client = TestClient(app)

    # Override provider resolver to use stub
    def resolver(_: Any) -> LLMProvider:
        return StubProvider()

    app.dependency_overrides[deps.get_provider_resolver] = lambda: resolver

    try:
        body = {
            "config": _valid_config(),
            "input": {"text": "hello"},
            "options": {"requestId": "req-1"},
        }
        resp = client.post("/execute", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert data["request_id"] == "req-1"
        msgs = data["messages"]
        assert any(m.get("role") == "assistant" and m.get("content") == "Hello" for m in msgs)
        assert data["steps"] >= 1
    finally:
        app.dependency_overrides.pop(deps.get_provider_resolver, None)

