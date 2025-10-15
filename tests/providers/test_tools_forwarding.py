from __future__ import annotations

from typing import Any

from codeless_orchestrator.config.loader import load_graph
from codeless_orchestrator.engine.executor import OrchestrationEngine
from codeless_orchestrator.providers.openai import OpenAIProvider
from codeless_orchestrator.providers.types import ChatMessage
from codeless_orchestrator.tools.registry import default_registry, register_default_tools


class _DummyChoice:
    def __init__(self) -> None:
        self.message = {"role": "assistant", "content": "ok"}
        self.finish_reason = "stop"


class _DummyResponse:
    def __init__(self) -> None:
        self.choices = [_DummyChoice()]
        self.usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}


class _FakeChatCompletions:
    def __init__(self) -> None:
        self.last_kwargs: dict[str, Any] | None = None

    def create(self, **kwargs: Any) -> Any:  # matches OpenAI SDK surface used
        self.last_kwargs = dict(kwargs)
        return _DummyResponse()


class _FakeOpenAIClient:
    def __init__(self) -> None:
        self.chat = type("_Chat", (), {"completions": _FakeChatCompletions()})()


def _config() -> dict[str, Any]:
    # Matches the user's provided orchestration config with a tool node attached
    return {
        "config": {
            "meta": {
                "id": "f6ed760b-9fa2-40f9-8a1a-7d492292da3b",
                "name": "Untitled Graph",
                "version": "1.0.0",
                "metadata": {"showToolNodesOnCanvas": True},
            },
            "nodes": [
                {
                    "id": "0b523619-2a84-4b5d-8ab4-77ad8a969b13",
                    "kind": "agent.codeless",
                    "label": "Policy Agent",
                    "data": {
                        "systemInstructions": "Answer the user's question",
                        "styleGuide": "",
                        "model": {
                            "provider": "openai",
                            "modelId": "gpt-4o",
                            "temperature": 0.3,
                            "topP": 1,
                            "maxTokens": 800,
                            "seed": None,
                            "stop": [],
                            "jsonModeEnabled": False,
                        },
                        "context": {
                            "historyWindow": {"mode": "LastN", "n": 10},
                            "injectOrgPreamble": True,
                            "vars": {},
                        },
                        "tools": {
                            "policy": "Auto",
                            "timeoutMs": 10000,
                            "maxCallsPerTurn": 0,
                            "parallelism": 10,
                            "redactPII": True,
                            "attached": ["93d1bbce-abd8-4db9-a90d-80a8279d5e40"],
                        },
                        "structuredOutput": {
                            "enabled": False,
                            "schema": {},
                            "onViolation": "RetryAndRepair",
                            "maxRepairAttempts": 2,
                            "postProcess": {"normalizeWhitespace": True, "ensureMarkdown": True},
                        },
                        "safety": {
                            "policyRef": "enterprise-v3",
                            "onBlock": "Refuse",
                            "piiRedaction": True,
                            "promptInjectionDefense": True,
                        },
                        "telemetry": {"labels": {}, "emitUsage": True},
                    },
                },
                {
                    "id": "93d1bbce-abd8-4db9-a90d-80a8279d5e40",
                    "kind": "tool",
                    "label": "Policy Search",
                    "data": {
                        "name": "Tool",
                        "toolId": "tool:web-search",
                        "version": "1.4.3",
                        "parameterOverrides": {},
                    },
                },
            ],
            "edges": [
                {
                    "id": "f5821851-232e-4d1d-9b4a-5d840b9b1e7a",
                    "from": "0b523619-2a84-4b5d-8ab4-77ad8a969b13",
                    "to": "93d1bbce-abd8-4db9-a90d-80a8279d5e40",
                }
            ],
        },
        "input": {"messages": [{"role": "user", "content": "test"}]},
    }


def test_openai_provider_forwards_tools_and_choice() -> None:
    # Ensure builtin tools registered in the default registry
    register_default_tools(default_registry)

    cfg_json = _config()["config"]
    cfg, _report = load_graph(cfg_json, validate=True)

    fake_client = _FakeOpenAIClient()
    provider = OpenAIProvider(client=fake_client)

    # Compile with our OpenAI provider stub and default registry
    engine = OrchestrationEngine()
    compiled = engine.compile(
        cfg,
        provider_resolver=lambda _mp: provider,
        registry=default_registry,
    )

    # Execute a single non-stream turn; provider.chat -> client.chat.completions.create
    engine.execute(compiled, _config()["input"])  # type: ignore[arg-type]

    kwargs = fake_client.chat.completions.last_kwargs  # type: ignore[attr-defined]
    assert kwargs is not None, "OpenAI client was not invoked"

    # Tools should be included with function specs
    tools = kwargs.get("tools")
    assert isinstance(tools, list) and len(tools) >= 1
    f0 = tools[0]
    assert f0.get("type") == "function"
    func = f0.get("function") or {}
    assert isinstance(func, dict)
    assert func.get("name", "").startswith("web_search__")
    params = func.get("parameters") or {}
    props = params.get("properties", {})
    assert "q" in props and "max_results" in props

    # With policy Auto, provider should forward tool_choice='auto'
    assert kwargs.get("tool_choice") == "auto"
