from __future__ import annotations

from codeless_orchestrator.providers.transform_gemini import (
    to_gemini_generation_config,
    to_gemini_history,
    to_gemini_tool_config,
    to_gemini_tools,
)
from codeless_orchestrator.providers.types import ChatMessage, ChatRequest, ToolCall, ToolSpec


def test_history_with_tool_linking() -> None:
    msgs = [
        ChatMessage(role="user", content="Search for x"),
        ChatMessage(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id="call_1", name="web_search", arguments_json='{"q":"x"}')],
        ),
        ChatMessage(role="tool", content='{"result":42}', tool_call_id="call_1"),
    ]

    out = to_gemini_history(msgs)
    # user message
    assert out[0]["role"] == "user"
    assert out[0]["parts"][0]["text"] == "Search for x"

    # assistant tool call is represented as function_call part
    parts1 = out[1]["parts"]
    assert "function_call" in parts1[0]
    fc = parts1[0]["function_call"]
    assert fc["name"] == "web_search"
    assert fc["args"] == {"q": "x"}

    # tool result is mapped to function_response with the linked name
    parts2 = out[2]["parts"]
    fr = parts2[0]["function_response"]
    assert fr["name"] == "web_search"
    assert fr["response"] == {"result": 42}


def test_tools_transform() -> None:
    tools = [
        ToolSpec(
            name="web_search",
            description="Search the web",
            parameters={
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
            },
        )
    ]

    out = to_gemini_tools(tools)
    assert isinstance(out, list)
    assert out and "function_declarations" in out[0]
    fns = out[0]["function_declarations"]
    assert fns[0]["name"] == "web_search"
    assert fns[0]["parameters"]["properties"]["q"]["type"] == "string"


def test_generation_config_json_mode() -> None:
    req = ChatRequest(
        model="gemini-1.5-pro",
        messages=[ChatMessage(role="user", content="{}")],
        json_mode_enabled=True,
    )
    cfg = to_gemini_generation_config(req)
    assert cfg.get("response_mime_type") == "application/json"


def test_tool_choice_modes() -> None:
    assert to_gemini_tool_config("auto") == {"function_calling_config": {"mode": "AUTO"}}
    assert to_gemini_tool_config("none") == {"function_calling_config": {"mode": "NONE"}}
    sel = {"type": "function", "function": {"name": "web_search"}}
    out = to_gemini_tool_config(sel)
    assert out == {
        "function_calling_config": {"mode": "ANY", "allowed_function_names": ["web_search"]}
    }
