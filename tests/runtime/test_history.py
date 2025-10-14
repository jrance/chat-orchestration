from __future__ import annotations

from codeless_orchestrator.config.schema import (
    AgentContext,
    AgentNode,
    AgentNodeData,
    HistoryWindow,
    ModelParams,
    ToolsConfig,
)
from codeless_orchestrator.providers.types import ChatMessage
from codeless_orchestrator.runtime.history import assemble_prompt, window_messages, make_system_message


def test_lastn_excludes_system() -> None:
    msgs = [
        ChatMessage(role="system", content="sys"),
        ChatMessage(role="user", content="u1"),
        ChatMessage(role="assistant", content="a1"),
        ChatMessage(role="tool", content="t1"),
        ChatMessage(role="user", content="u2"),
    ]
    out = window_messages(msgs, mode="LastN", n=2)
    # Should include only the last two of user/assistant/tool roles, in order
    assert [m.role for m in out] == ["tool", "user"]
    assert [m.content for m in out] == ["t1", "u2"]


def test_system_assembly_with_preamble_and_vars() -> None:
    agent = AgentNode(
        id="a1",
        label="Agent",
        data=AgentNodeData(
            system_instructions="Hello {dept}",
            style_guide="Use {tone} tone.",
            model=ModelParams(provider="openai", model_id="gpt-4o-mini"),
            context=AgentContext(
                history_window=HistoryWindow(mode="LastN", n=2),
                inject_org_preamble=True,
                vars={},
            ),
            tools=ToolsConfig(),
            structured_output={},
            safety={"policy_ref": "default"},
            telemetry={"labels": {}, "emit_usage": True},
        ),
    )
    org_preamble = "Org Preamble"
    vars_req = {"dept": "Legal", "tone": "formal"}
    # No history
    messages = []
    assembled = assemble_prompt(agent, messages, org_preamble, vars_req)
    assert len(assembled) >= 1
    sys = assembled[0]
    assert sys.role == "system"
    text = sys.content if isinstance(sys.content, str) else ""
    assert "Org Preamble" in text
    assert "Hello Legal" in text
    assert "Use formal tone." in text

