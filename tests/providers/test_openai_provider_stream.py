from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from codeless_orchestrator.providers.openai import OpenAIProvider
from codeless_orchestrator.providers.types import ChatMessage, ChatRequest


class _Delta:
    def __init__(
        self,
        role: str | None = None,
        content: str | None = None,
        tool_calls: Any = None,
    ) -> None:
        self.role = role
        self.content = content
        self.tool_calls = tool_calls


class _Function:
    def __init__(self, name: str | None = None, arguments: str | None = None) -> None:
        self.name = name
        self.arguments = arguments


class _ToolCall:
    def __init__(self, index: int, id: str | None, function: _Function) -> None:
        self.index = index
        self.id = id
        self.function = function


class _Choice:
    def __init__(self, delta: _Delta | None = None, finish_reason: str | None = None) -> None:
        self.delta = delta
        self.finish_reason = finish_reason


class _Event:
    def __init__(self, choice: _Choice) -> None:
        self.choices = [choice]


class StubStreamClient:
    def __init__(self, events: Iterable[_Event]) -> None:
        self._events = list(events)
        self.chat = type("Chat", (), {"completions": self})()
        self.last_kwargs: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> Iterable[_Event]:  # type: ignore[override]
        assert kwargs.get("stream") is True
        self.last_kwargs = kwargs
        return iter(self._events)

    def with_options(self, **_: Any) -> StubStreamClient:  # pragma: no cover - not used here
        return self


def test_openai_provider_stream_with_tool_calls() -> None:
    events: list[_Event] = [
        _Event(_Choice(delta=_Delta(role="assistant"))),
        _Event(_Choice(delta=_Delta(content="Hel"))),
        _Event(_Choice(delta=_Delta(content="lo"))),
        _Event(
            _Choice(
                delta=_Delta(
                    tool_calls=[
                        _ToolCall(
                            index=0,
                            id="call_1",
                            function=_Function(name="web_search", arguments=None),
                        )
                    ]
                )
            )
        ),
        _Event(
            _Choice(
                delta=_Delta(
                    tool_calls=[
                        _ToolCall(
                            index=0,
                            id="call_1",
                            function=_Function(arguments='{"q":"t"}'),
                        )
                    ]
                )
            )
        ),
        _Event(
            _Choice(
                delta=_Delta(
                    tool_calls=[
                        _ToolCall(index=0, id="call_1", function=_Function(arguments='est"}'))
                    ]
                )
            )
        ),
        _Event(_Choice(delta=_Delta(), finish_reason="tool_calls")),
    ]

    client = StubStreamClient(events)
    provider = OpenAIProvider(client=client)

    req = ChatRequest(model="gpt-4o-mini", messages=[ChatMessage(role="user", content="hi")])
    out = list(provider.stream(req))

    # Validate ordered deltas
    assert out[0].type == "role" and out[0].role == "assistant"
    assert out[1].type == "content.delta" and out[1].text_delta == "Hel"
    assert out[2].type == "content.delta" and out[2].text_delta == "lo"
    assert out[3].type == "tool_call.start"
    assert out[3].tool_call_index == 0
    assert out[3].tool_call_id == "call_1"
    assert out[3].tool_name == "web_search"
    assert out[4].type == "tool_call.delta" and out[4].arguments_delta == '{"q":"t"}'
    assert out[5].type == "tool_call.delta" and out[5].arguments_delta == 'est"}'
    assert out[-1].type == "message.end" and out[-1].finish_reason == "tool_calls"


def test_openai_provider_stream_pure_text_stop() -> None:
    events: list[_Event] = [
        _Event(_Choice(delta=_Delta(role="assistant"))),
        _Event(_Choice(delta=_Delta(content="Hello "))),
        _Event(_Choice(delta=_Delta(content="world"))),
        _Event(_Choice(delta=_Delta(), finish_reason="stop")),
    ]

    client = StubStreamClient(events)
    provider = OpenAIProvider(client=client)

    req = ChatRequest(model="gpt-4o-mini", messages=[ChatMessage(role="user", content="hi")])
    out = list(provider.stream(req))

    assert [c.type for c in out] == ["role", "content.delta", "content.delta", "message.end"]
    assert out[-1].finish_reason == "stop"
