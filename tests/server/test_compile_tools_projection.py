from __future__ import annotations

from fastapi.testclient import TestClient

from codeless_orchestrator.server.app import app


def _config_with_tool_node() -> dict:
    return {
        "meta": {"id": "g1", "name": "Sample", "version": "1", "metadata": {"showToolNodesOnCanvas": True}},
        "nodes": [
            {
                "id": "node-93d1bbce",
                "kind": "tool",
                "label": "Web Search",
                "data": {
                    "name": "web_search",
                    "toolId": "tool:web-search",
                    "version": "1.0",
                    "parameterOverrides": {"max_results": 3},
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
                    "context": {"historyWindow": {"mode": "LastN", "n": 5}, "injectOrgPreamble": True, "vars": {}},
                    "tools": {"policy": "Auto", "timeoutMs": 10000, "maxCallsPerTurn": 1, "parallelism": 2, "redactPII": True, "attached": ["node-93d1bbce"]},
                    "structuredOutput": {"enabled": False, "schema": {}},
                    "safety": {"policyRef": "default"},
                    "telemetry": {"labels": {}, "emitUsage": True},
                },
            },
        ],
        "edges": [],
    }


def test_compile_projects_resolved_tools() -> None:
    client = TestClient(app)
    resp = client.post("/compile", json=_config_with_tool_node())
    assert resp.status_code == 200
    body = resp.json()
    tools = body.get("tools") or []
    assert len(tools) == 1
    t0 = tools[0]
    assert t0.get("nodeId") == "node-93d1bbce"
    assert t0.get("toolId") == "tool:web-search"
    fname = t0.get("functionName")
    assert isinstance(fname, str) and fname.startswith("web_search__")
