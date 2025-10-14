from __future__ import annotations

from typing import Any

from codeless_orchestrator.providers.openai import OpenAIProvider
from codeless_orchestrator.providers.types import ChatMessage, ChatRequest


class StubClient:
    class _Completions:
        def __init__(self, parent: StubClient) -> None:
            self._parent = parent

        def create(self, *args: Any, **kwargs: Any):  # mimics sdk
            self._parent.last_kwargs = kwargs
            # Simulate normal OpenAI response object
            class Choice:
                def __init__(self):
                    self.message = type("M", (), {"role": "assistant", "content": "{}"})()
                    self.finish_reason = "stop"

            class Resp:
                def __init__(self):
                    self.choices = [Choice()]

            # If schema provided in response_format, simulate success
            return Resp()

    def __init__(self, fail_first: bool = False) -> None:
        self.chat = type("Chat", (), {"completions": StubClient._Completions(self)})()
        self.last_kwargs: dict[str, Any] | None = None
        self._fail_first = fail_first


def test_response_format_with_schema() -> None:
    stub = StubClient()
    provider = OpenAIProvider(client=stub)
    req = ChatRequest(
        model="gpt",
        messages=[ChatMessage(role="user", content="hi")],
        json_mode_enabled=True,
        response_format_schema={"type": "object", "properties": {}, "additionalProperties": True},
    )
    provider.chat(req)
    kwargs = stub.last_kwargs or {}
    rf = kwargs.get("response_format", {})
    assert rf.get("type") in ("json_schema", "json_object")
    if rf.get("type") == "json_schema":
        assert "json_schema" in rf

