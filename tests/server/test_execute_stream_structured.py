from __future__ import annotations

import json
from typing import Any, Iterator

from fastapi.testclient import TestClient

from codeless_orchestrator.providers.base import LLMProvider
from codeless_orchestrator.providers.types import ChatChunk, ChatMessage, ChatRequest, ChatResponse
from codeless_orchestrator.server import deps
from codeless_orchestrator.server.app import app


class StubProvider(LLMProvider):
    def chat(self, req: ChatRequest) -> ChatResponse:
        return ChatResponse(message=ChatMessage(role="assistant", content=json.dumps({"answer": "ok"})), finish_reason="stop")

    def stream(self, req: ChatRequest) -> Iterator[ChatChunk]:  # pragma: no cover - not used
        yield from ()


def _cfg() -> dict[str, Any]:
    return {
        "meta": {"id": "m1", "name": "t", "version": "0", "metadata": {"showToolNodesOnCanvas": True}},
        "nodes": [
            {
                "id": "a1",
                "kind": "agent.codeless",
                "label": "Agent",
                "data": {
                    "systemInstructions": "",
                    "styleGuide": "",
                    "model": {"provider": "openai", "modelId": "gpt-4o-mini"},
                    "context": {"historyWindow": {"mode": "LastN", "n": 3}, "injectOrgPreamble": False, "vars": {}},
                    "tools": {"policy": "None", "attached": [], "timeoutMs": 10000, "maxCallsPerTurn": 0, "parallelism": 1, "redactPII": True},
                    "structuredOutput": {"enabled": True, "schema": {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"], "additionalProperties": False}},
                    "safety": {"policyRef": "default"},
                    "telemetry": {"labels": {}, "emitUsage": True},
                },
            }
        ],
        "edges": [],
    }


def test_execute_stream_structured_ok() -> None:
    client = TestClient(app)
    app.dependency_overrides[deps.get_provider_resolver] = lambda: (lambda _: StubProvider())
    try:
        body = {"config": _cfg(), "input": {"text": "q"}}
        with client.stream("POST", "/execute/stream", json=body) as r:
            assert r.status_code == 200
            buf = ""
            for chunk in r.iter_text():
                buf += chunk or ""
            # role and content should appear, then message.end and end
            assert "\"type\":\"role\"" in buf
            assert "\"type\":\"content.delta\"" in buf
            # Compact JSON will be escaped inside text_delta string
            assert '"text_delta":"{' in buf and 'answer' in buf
            assert '"type":"message.end"' in buf or '"type":"end"' in buf
    finally:
        app.dependency_overrides.pop(deps.get_provider_resolver, None)
