from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from codeless_orchestrator.providers.base import LLMProvider
from codeless_orchestrator.providers.types import ChatMessage, ChatRequest, ChatResponse
from codeless_orchestrator.server import deps
from codeless_orchestrator.server.app import app


class CaptureProvider(LLMProvider):
    def __init__(self) -> None:
        self.last_req: ChatRequest | None = None

    def chat(self, req: ChatRequest) -> ChatResponse:
        self.last_req = req
        return ChatResponse(message=ChatMessage(role="assistant", content="ok"), finish_reason="stop")

    def stream(self, req: ChatRequest):  # pragma: no cover - not used
        yield from ()


def _config() -> dict[str, Any]:
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
                    "systemInstructions": "Hello {dept}",
                    "styleGuide": "Use {tone} tone.",
                    "model": {"provider": "openai", "modelId": "gpt-4o-mini"},
                    "context": {
                        "historyWindow": {"mode": "LastN", "n": 3},
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


def test_execute_nonstream_with_vars_and_preamble() -> None:
    client = TestClient(app)

    provider = CaptureProvider()

    def resolver(_: Any) -> LLMProvider:
        return provider

    app.dependency_overrides[deps.get_provider_resolver] = lambda: resolver
    app.dependency_overrides[deps.get_org_preamble] = lambda: "Injected Preamble"

    try:
        body = {
            "config": _config(),
            "input": {"text": "hi", "vars": {"dept": "Legal", "tone": "formal"}},
        }
        resp = client.post("/execute", json=body)
        assert resp.status_code == 200
        # ensure provider saw rendered system
        assert provider.last_req is not None
        first = provider.last_req.messages[0]
        assert first.role == "system"
        text = first.content if isinstance(first.content, str) else ""
        assert "Injected Preamble" in text
        assert "Hello Legal" in text
        assert "Use formal tone." in text
    finally:
        app.dependency_overrides.pop(deps.get_provider_resolver, None)
        app.dependency_overrides.pop(deps.get_org_preamble, None)

