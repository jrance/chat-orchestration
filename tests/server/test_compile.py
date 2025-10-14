from __future__ import annotations

from fastapi.testclient import TestClient

from codeless_orchestrator.server.app import app


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
                "id": "tool:web-search",
                "kind": "tool",
                "label": "Web Search",
                "data": {
                    "name": "web_search",
                    "toolId": "tool:web-search",
                    "version": "1.0",
                    "parameterOverrides": {},
                },
            },
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
                        "policy": "Auto",
                        "timeoutMs": 10000,
                        "maxCallsPerTurn": 1,
                        "parallelism": 2,
                        "redactPII": True,
                        "attached": ["tool:web-search"],
                    },
                    "structuredOutput": {"enabled": False, "schema": {}},
                    "safety": {"policyRef": "default"},
                    "telemetry": {"labels": {}, "emitUsage": True},
                },
            },
        ],
        "edges": [
            {"id": "e1", "from": "agent-1", "to": "tool:web-search"}
        ],
    }


def test_compile_metadata_and_capabilities() -> None:
    client = TestClient(app)
    resp = client.post("/compile", json=_valid_config())
    assert resp.status_code == 200
    data = resp.json()

    assert data["ok"] is True
    assert data["orchestration"] == "single"
    # Agent metadata present
    agents = data["agents"]
    assert len(agents) == 1
    assert agents[0]["id"] == "agent-1"
    assert agents[0]["provider"] == "openai"
    # Tools projection present with function names
    tools = data["tools"]
    assert len(tools) == 1
    assert tools[0]["toolId"] == "tool:web-search"
    assert isinstance(tools[0].get("functionName"), str)
    # Capabilities
    caps = data["capabilities"]
    assert caps.get("streaming") is True
    assert caps.get("tools") is True
