from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from codeless_orchestrator.providers.gemini import GeminiProvider
from codeless_orchestrator.providers.types import ChatMessage, ChatRequest


class _StreamEvent:
    def __init__(
        self,
        *,
        role: str | None = None,
        text: str | None = None,
        function_calls: list[Any] | None = None,
        finish_reason: str | None = None,
    ) -> None:
        self.role = role
        self.text = text
        self.function_calls = function_calls
        self.finish_reason = finish_reason
        self.finishReason = finish_reason


class _FnCall:
    def __init__(self, index: int, name: str | None, id: str | None, arguments: str | None) -> None:
        self.index = index
        self.name = name
        self.id = id
        self.arguments = arguments


class StubStreamModel:
    def __init__(self) -> None:
        self.last_kwargs: dict[str, Any] = {}

    def generate_content(self, **kwargs: Any) -> Iterable[_StreamEvent]:  # type: ignore[override]
        assert kwargs.get("stream") is True
        self.last_kwargs = kwargs
        events: list[_StreamEvent] = [
            _StreamEvent(role="model"),
            _StreamEvent(text="Hel"),
            _StreamEvent(text="lo"),
            _StreamEvent(
                function_calls=[
                    _FnCall(index=0, name="web_search", id=None, arguments=None)
                ]
            ),
            _StreamEvent(
                function_calls=[
                    _FnCall(index=0, name=None, id=None, arguments='{"q":"te"}')
                ]
            ),
            _StreamEvent(finish_reason="STOP"),
        ]
        return iter(events)


def test_gemini_provider_stream_sequence() -> None:
    model = StubStreamModel()
    provider = GeminiProvider(model_client=model)

    req = ChatRequest(model="gemini-1.5-pro", messages=[ChatMessage(role="user", content="hi")])
    out = list(provider.stream(req))

    # Expected ordered events
    assert out[0].type == "role" and out[0].role == "assistant"
    assert out[1].type == "content.delta" and out[1].text_delta == "Hel"
    assert out[2].type == "content.delta" and out[2].text_delta == "lo"
    assert out[3].type == "tool_call.start"
    assert out[3].tool_call_index == 0
    assert out[3].tool_name == "web_search"
    assert out[4].type == "tool_call.delta" and out[4].arguments_delta == '{"q":"te"}'
    assert out[-1].type == "message.end" and out[-1].finish_reason == "stop"
