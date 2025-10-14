from __future__ import annotations

import abc
from collections.abc import Iterator

from .types import ChatChunk, ChatRequest, ChatResponse


class LLMProvider(abc.ABC):
    @abc.abstractmethod
    def chat(self, req: ChatRequest) -> ChatResponse:  # pragma: no cover - interface
        """Non-streaming chat completion request."""

    @abc.abstractmethod
    def stream(self, req: ChatRequest) -> Iterator[ChatChunk]:  # pragma: no cover - interface
        """Streaming chat completion request yielding structured deltas."""
