from __future__ import annotations

import pytest

from codeless_orchestrator.config.loader import load_graph


def _valid_graph_json_camel() -> dict:
    return {
        "meta": {
            "id": "g1",
            "name": "Sample",
            "version": "1",
            "metadata": {"showToolNodesOnCanvas": True},
        },
        "nodes": [
            {
                "id": "tool1",
                "kind": "tool",
                "label": "Web Search",
                "data": {
                    "name": "duckduckgo",
                    "toolId": "duckduckgo-search",
                    "version": "1.0",
                    "parameterOverrides": {},
                },
            },
            {
                "id": "agent1",
                "kind": "agent.codeless",
                "label": "Main Agent",
                "data": {
                    "systemInstructions": "You are helpful.",
                    "styleGuide": "",
                    "model": {
                        "provider": "openai",
                        "modelId": "gpt-4o-mini",
                        "temperature": 0.7,
                        "topP": 1.0,
                        "jsonModeEnabled": False,
                    },
                    "context": {
                        "historyWindow": {"mode": "LastN", "n": 5},
                        "injectOrgPreamble": True,
                        "vars": {},
                    },
                    "tools": {
                        "policy": "Auto",
                        "timeoutMs": 10000,
                        "maxCallsPerTurn": 2,
                        "parallelism": 4,
                        "redactPII": True,
                        "attached": ["tool1"],
                    },
                    "structuredOutput": {
                        "enabled": False,
                        "schema": {},
                        "onViolation": "RetryAndRepair",
                        "maxRepairAttempts": 2,
                        "postProcess": {
                            "normalizeWhitespace": True,
                            "ensureMarkdown": True,
                        },
                    },
                    "safety": {
                        "policyRef": "policy:v1",
                        "onBlock": "Refuse",
                        "piiRedaction": True,
                        "promptInjectionDefense": True,
                    },
                    "telemetry": {"labels": {"env": "test"}, "emitUsage": True},
                },
            },
        ],
        "edges": [
            {"id": "e1", "from": "agent1", "to": "tool1"}
        ],
    }


def _valid_graph_json_snake() -> dict:
    return {
        "meta": {
            "id": "g1",
            "name": "Sample",
            "version": "1",
            "metadata": {"show_tool_nodes_on_canvas": True},
        },
        "nodes": [
            {
                "id": "tool1",
                "kind": "tool",
                "label": "Web Search",
                "data": {
                    "name": "duckduckgo",
                    "tool_id": "duckduckgo-search",
                    "version": "1.0",
                    "parameter_overrides": {},
                },
            },
            {
                "id": "agent1",
                "kind": "agent.codeless",
                "label": "Main Agent",
                "data": {
                    "system_instructions": "You are helpful.",
                    "style_guide": "",
                    "model": {
                        "provider": "openai",
                        "model_id": "gpt-4o-mini",
                        "temperature": 0.7,
                        "top_p": 1.0,
                        "json_mode_enabled": False,
                    },
                    "context": {
                        "history_window": {"mode": "LastN", "n": 5},
                        "inject_org_preamble": True,
                        "vars": {},
                    },
                    "tools": {
                        "policy": "Auto",
                        "timeout_ms": 10000,
                        "max_calls_per_turn": 2,
                        "parallelism": 4,
                        "redact_pii": True,
                        "attached": ["tool1"],
                    },
                    "structured_output": {
                        "enabled": False,
                        "schema": {},
                        "on_violation": "RetryAndRepair",
                        "max_repair_attempts": 2,
                        "post_process": {
                            "normalize_whitespace": True,
                            "ensure_markdown": True,
                        },
                    },
                    "safety": {
                        "policy_ref": "policy:v1",
                        "on_block": "Refuse",
                        "pii_redaction": True,
                        "prompt_injection_defense": True,
                    },
                    "telemetry": {"labels": {"env": "test"}, "emit_usage": True},
                },
            },
        ],
        "edges": [
            {"id": "e1", "from": "agent1", "to": "tool1"}
        ],
    }


def test_load_valid_single_agent_with_tool() -> None:
    cfg, report = load_graph(_valid_graph_json_camel())

    assert report.issues == [], "Expected no validation issues"

    # Assert normalized fields
    agent = next(n for n in cfg.nodes if getattr(n, "kind", None) == "agent.codeless")
    assert agent.data.model.top_p == 1.0
    assert agent.data.model.json_mode_enabled is False
    assert agent.data.context.history_window.mode == "LastN"


def test_aliases_are_accepted() -> None:
    cfg_snake, report_snake = load_graph(_valid_graph_json_snake())
    assert not report_snake.issues

    cfg_camel, report_camel = load_graph(_valid_graph_json_camel())
    assert not report_camel.issues

    # Both parse identically for select normalized fields
    a_snake = next(n for n in cfg_snake.nodes if getattr(n, "kind", None) == "agent.codeless")
    a_camel = next(n for n in cfg_camel.nodes if getattr(n, "kind", None) == "agent.codeless")
    assert a_snake.data.model.top_p == a_camel.data.model.top_p == 1.0
    assert a_snake.data.tools.timeout_ms == a_camel.data.tools.timeout_ms == 10000


def _invalid_base() -> dict:
    base = _valid_graph_json_camel()
    # Keep structure, override in tests
    return base


def test_invalid_model_params_temperature() -> None:
    data = _invalid_base()
    agent = next(n for n in data["nodes"] if n["kind"] == "agent.codeless")
    agent["data"]["model"]["temperature"] = -1
    with pytest.raises(ValueError) as ei:
        load_graph(data)
    assert "temperature" in str(ei.value)


def test_invalid_model_params_top_p() -> None:
    data = _invalid_base()
    agent = next(n for n in data["nodes"] if n["kind"] == "agent.codeless")
    agent["data"]["model"]["topP"] = 2.5
    with pytest.raises(ValueError) as ei:
        load_graph(data)
    assert "top_p" in str(ei.value) or "topP" in str(ei.value)

