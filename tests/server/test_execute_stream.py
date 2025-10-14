from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from fastapi.testclient import TestClient

from codeless_orchestrator.providers.base import LLMProvider
from codeless_orchestrator.providers.types import ChatChunk, ChatRequest
from codeless_orchestrator.server import deps
from codeless_orchestrator.server.app import app


class StubStreamProvider(LLMProvider):
    def chat(self, req: ChatRequest):  # pragma: no cover - not used
        raise RuntimeError("not used")

    def stream(self, req: ChatRequest) -> Iterator[ChatChunk]:
        yield ChatChunk(type="role", role="assistant")
        yield ChatChunk(type="content.delta", text_delta="Hel")
        yield ChatChunk(type="content.delta", text_delta="lo")
        yield ChatChunk(type="message.end", finish_reason="stop")


def _valid_config() -> dict:
    # Single agent with no tools (streaming-allowed path)
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


def test_execute_stream_sends_events_in_order() -> None:
    client = TestClient(app)

    def resolver(_: Any) -> LLMProvider:
        return StubStreamProvider()

    app.dependency_overrides[deps.get_provider_resolver] = lambda: resolver

    try:
        body = {"config": _valid_config(), "input": {"text": "hello"}, "options": {}}
        events: list[dict] = []
        with client.stream("POST", "/execute/stream", json=body) as resp:
            for line in resp.iter_lines():
                if not line:
                    continue
                s = line.decode("utf-8") if isinstance(line, (bytes, bytearray)) else str(line)
                if not s.startswith("data: "):
                    continue
                payload = s[len("data: ") :]
                events.append(__import__("json").loads(payload))
                # Break after end to avoid waiting
                if events and events[-1].get("type") == "end":
                    break

        types = [e.get("type") for e in events]
        assert types[:4] == ["role", "content.delta", "content.delta", "message.end"], types
        assert types[-1] == "end"
    finally:
        app.dependency_overrides.pop(deps.get_provider_resolver, None)
