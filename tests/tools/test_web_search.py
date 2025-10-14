from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from codeless_orchestrator.tools.builtin.web_search import WebSearchTool


class StubDDGSClient:
    def text(
        self,
        *,
        q: str,
        max_results: int = 5,
        region: str | None = None,
        safesearch: str | None = None,
        time: str | None = None,
        backend: str | None = None,
    ) -> Iterable[dict[str, Any]]:
        # Emit deterministic rows based on max_results
        out = []
        for i in range(max_results):
            out.append(
                {
                    "title": f"{q} result {i}",
                    "href": f"https://example.com/{q}/{i}",
                    "body": f"Body {i}",
                }
            )
        return out


def test_web_search_invocation_and_shape() -> None:
    tool = WebSearchTool(client=StubDDGSClient())
    args = {"q": "openai", "max_results": 3}
    results = tool.invoke(args)
    assert isinstance(results, list)
    assert len(results) == 3
    first = results[0]
    assert set(["title", "href", "body", "source"]).issubset(first.keys())
    assert first["source"] == "duckduckgo"


def test_parameters_schema() -> None:
    schema = WebSearchTool().parameters
    assert schema["type"] == "object"
    assert "q" in schema["properties"]
    assert "q" in schema["required"]
