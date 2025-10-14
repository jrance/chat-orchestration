from __future__ import annotations

import pytest

from codeless_orchestrator.config.loader import load_graph
from codeless_orchestrator.config.schema import (
    AgentContext,
    AgentNode,
    AgentNodeData,
    GraphConfig,
    Meta,
    MetaMetadata,
    ModelParams,
    ToolsConfig,
)
from codeless_orchestrator.engine.compiler import compile_single_agent
from codeless_orchestrator.providers.base import LLMProvider
from codeless_orchestrator.providers.types import ChatMessage, ChatRequest, ChatResponse
from codeless_orchestrator.tools.registry import ToolRegistry


class _Stub(LLMProvider):  # pragma: no cover - trivial
    def chat(self, req: ChatRequest) -> ChatResponse:
        return ChatResponse(message=ChatMessage(role="assistant", content=""))

    def stream(self, req: ChatRequest):
        if False:
            yield


def _agent(attached: list[str]) -> GraphConfig:
    agent = AgentNode(
        id="a1",
        label="A",
        data=AgentNodeData(
            system_instructions="",
            style_guide="",
            model=ModelParams(provider="openai", model_id="gpt-4o-mini"),
            context=AgentContext(history_window={"mode": "LastN", "n": 3}, inject_org_preamble=True, vars={}),
            tools=ToolsConfig(attached=attached, policy="Auto"),
            structured_output={},
            safety={"policy_ref": "default"},
            telemetry={"labels": {}, "emit_usage": True},
        ),
    )
    return GraphConfig(meta=Meta(id="m1", name="t", version="0", metadata=MetaMetadata()), nodes=[agent], edges=[])


def test_validator_unknown_attached_node() -> None:
    cfg, report = load_graph(_agent(["missing-node"]).model_dump(), validate=False)
    errs = [i for i in report.issues if i.severity == "ERROR"]
    assert any(i.code == "tools.unknown_attachment" for i in errs)


def test_validator_attached_id_not_a_tool() -> None:
    # Attach an agent by id
    a1 = AgentNode(
        id="a1",
        label="A",
        data=AgentNodeData(
            system_instructions="",
            style_guide="",
            model=ModelParams(provider="openai", model_id="gpt-4o-mini"),
            context=AgentContext(history_window={"mode": "LastN", "n": 3}, inject_org_preamble=True, vars={}),
            tools=ToolsConfig(attached=["a2"], policy="Auto"),
            structured_output={},
            safety={"policy_ref": "default"},
            telemetry={"labels": {}, "emit_usage": True},
        ),
    )
    a2 = AgentNode(
        id="a2",
        label="B",
        data=a1.data,
    )
    cfg = GraphConfig(meta=Meta(id="m1", name="t", version="0", metadata=MetaMetadata()), nodes=[a1, a2], edges=[])
    _, report = load_graph(cfg.model_dump(), validate=False)
    errs = [i for i in report.issues if i.severity == "ERROR"]
    assert any(i.code == "tools.attachment_not_tool" for i in errs)


def test_compile_error_for_unknown_tool_id_string() -> None:
    # Validator should not block 'tool:' forms; compiler should error with clear message
    cfg = _agent(["tool:missing-tool"])
    with pytest.raises(Exception) as excinfo:
        _ = compile_single_agent(cfg, provider_resolver=lambda _mp: _Stub(), registry=ToolRegistry())
    assert "Unknown toolId 'tool:missing-tool'" in str(excinfo.value)

