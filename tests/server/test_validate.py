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
                    "systemInstructions": "You are helpful.",
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


def test_validate_ok() -> None:
    client = TestClient(app)
    resp = client.post("/validate", json=_valid_config())
    assert resp.status_code == 200
    data = resp.json()
    assert data["valid"] is True
    assert data["errors"] == []


def test_validate_unknown_node_edge() -> None:
    client = TestClient(app)
    bad = _valid_config()
    # Introduce an edge to an unknown node
    bad["edges"].append({"id": "e2", "from": "agent-1", "to": "missing"})
    resp = client.post("/validate", json=bad)
    assert resp.status_code == 200
    data = resp.json()
    assert data["valid"] is False
    # One of the issues should be edge.unknown_node
    codes = [i["code"] for i in data["errors"]]
    assert "edge.unknown_node" in codes

