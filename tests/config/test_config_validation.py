from __future__ import annotations

from codeless_orchestrator.config.loader import load_graph


def _minimal_meta() -> dict:
    return {
        "id": "g",
        "name": "G",
        "version": "1",
        "metadata": {"show_tool_nodes_on_canvas": True},
    }


def _agent_node(node_id: str = "a1") -> dict:
    return {
        "id": node_id,
        "kind": "agent.codeless",
        "label": "A",
        "data": {
            "system_instructions": "You are helpful.",
            "style_guide": "",
            "model": {
                "provider": "openai",
                "model_id": "gpt-4o-mini",
                "temperature": 0.7,
                "top_p": 0.9,
                "json_mode_enabled": False,
            },
            "context": {"history_window": {"mode": "LastN", "n": 3}},
            "tools": {
                "policy": "Auto",
                "timeout_ms": 1000,
                "max_calls_per_turn": 0,
                "parallelism": 1,
                "redact_pii": True,
                "attached": [],
            },
            "structured_output": {"enabled": False, "schema": {}},
            "safety": {"policy_ref": "p:v1"},
            "telemetry": {"labels": {}},
        },
    }


def _tool_node(node_id: str = "t1") -> dict:
    return {
        "id": node_id,
        "kind": "tool",
        "label": "T",
        "data": {
            "name": "duckduckgo",
            "tool_id": "duckduckgo-search",
            "version": "1.0",
        },
    }


def test_edge_refers_to_unknown_node() -> None:
    data = {
        "meta": _minimal_meta(),
        "nodes": [_agent_node("a1")],
        "edges": [{"id": "e1", "from": "a1", "to": "missing"}],
    }
    _, report = load_graph(data, validate=False)
    errs = [i for i in report.issues if i.severity == "ERROR"]
    assert any(i.code == "edge.unknown_node" for i in errs)


def test_attached_tool_must_exist_and_be_tool() -> None:
    # Missing tool id
    data_missing = {
        "meta": _minimal_meta(),
        "nodes": [_agent_node("a1")],
        "edges": [],
    }
    # attach non-existent
    data_missing["nodes"][0]["data"]["tools"]["attached"] = ["t-missing"]
    _, report_missing = load_graph(data_missing, validate=False)
    assert any(i.code == "tools.unknown_attachment" for i in report_missing.issues)

    # Attach a non-tool (another agent)
    data_not_tool = {
        "meta": _minimal_meta(),
        "nodes": [_agent_node("a1"), _agent_node("a2")],
        "edges": [],
    }
    data_not_tool["nodes"][0]["data"]["tools"]["attached"] = ["a2"]
    _, report_not_tool = load_graph(data_not_tool, validate=False)
    assert any(i.code == "tools.attachment_not_tool" for i in report_not_tool.issues)


def test_duplicate_node_ids() -> None:
    data = {
        "meta": _minimal_meta(),
        "nodes": [_agent_node("dup"), _tool_node("dup")],
        "edges": [],
    }
    _, report = load_graph(data, validate=False)
    assert any(i.code == "node.duplicate_id" for i in report.issues)


def test_warnings() -> None:
    # Trigger warnings by invalid but non-fatal values
    data = {
        "meta": _minimal_meta(),
        "nodes": [_agent_node("a1"), _tool_node("t1")],
        "edges": [],
    }
    data["nodes"][0]["data"]["tools"]["max_calls_per_turn"] = -1
    data["nodes"][0]["data"]["tools"]["parallelism"] = 0
    _, report = load_graph(data, validate=False)
    assert any(i.severity == "WARNING" for i in report.issues)

