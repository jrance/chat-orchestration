from __future__ import annotations

from codeless_orchestrator.providers.transform_openai import (
    from_openai_message,
    to_openai_messages,
    to_openai_tools,
)
from codeless_orchestrator.providers.types import ChatMessage, ToolCall, ToolSpec


def test_to_openai_messages_tool_and_text() -> None:
    msgs = [
        ChatMessage(role="user", content="Hello"),
        ChatMessage(role="tool", content="{\"result\":42}", tool_call_id="call_1"),
    ]

    out = to_openai_messages(msgs)
    assert out == [
        {"role": "user", "content": "Hello"},
        {"role": "tool", "content": '{"result":42}', "tool_call_id": "call_1"},
    ]


def test_to_openai_messages_with_assistant_tool_calls() -> None:
    """Test that assistant messages with tool_calls are properly converted."""
    msgs = [
        ChatMessage(role="user", content="who won the game?"),
        ChatMessage(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(
                    id="call_123",
                    name="web_search",
                    arguments_json='{"q":"panthers vs cowboys"}',
                )
            ],
        ),
        ChatMessage(
            role="tool",
            content='{"result": "Panthers won"}',
            tool_call_id="call_123",
        ),
    ]

    out = to_openai_messages(msgs)
    assert len(out) == 3
    assert out[0] == {"role": "user", "content": "who won the game?"}
    assert out[1] == {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "web_search",
                    "arguments": '{"q":"panthers vs cowboys"}',
                },
            }
        ],
    }
    assert out[2] == {
        "role": "tool",
        "content": '{"result": "Panthers won"}',
        "tool_call_id": "call_123",
    }


def test_to_openai_tools() -> None:
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

    out = to_openai_tools(tools)
    assert out == [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web",
                "parameters": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                    "required": ["q"],
                },
            },
        }
    ]


def test_from_openai_message_with_tool_calls() -> None:
    # Construct a dict-shaped message similar to OpenAI SDK object
    msg = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "function": {"name": "web_search", "arguments": '{"q":"x"}'},
            }
        ],
    }

    out = from_openai_message(msg)
    assert out.role == "assistant"
    assert out.tool_calls is not None
    assert len(out.tool_calls) == 1
    tc = out.tool_calls[0]
    assert tc.id == "call_1"
    assert tc.name == "web_search"
    assert tc.arguments_json == '{"q":"x"}'

