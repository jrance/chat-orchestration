from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from duckduckgo_search import DDGS

from ..base import BaseTool


class _DefaultDDGSClient:
    """Wrapper to abstract DDGS for simple DI in tests."""

    def __init__(self) -> None:
        self._client = DDGS()

    # Some versions of duckduckgo_search expect `keywords`, others allow `q`.
    # We defensively support both for compatibility.
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
        kwargs: dict[str, Any] = {
            "max_results": max_results,
        }
        if region is not None:
            kwargs["region"] = region
        if safesearch is not None:
            kwargs["safesearch"] = safesearch
        if time is not None:
            # duckduckgo_search uses `timelimit` in some versions
            kwargs["timelimit"] = time
        if backend is not None:
            kwargs["backend"] = backend
        # Prefer `keywords` which is widely supported across versions.
        return self._client.text(keywords=q, **kwargs)


class WebSearchTool(BaseTool):
    name = "web_search"
    description = "Search the web for recent information."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "q": {"type": "string", "description": "Search query."},
            "max_results": {
                "type": "integer",
                "minimum": 1,
                "maximum": 50,
                "default": 5,
                "description": "Number of results to return (1-50).",
            },
            "region": {"type": "string", "description": "Region code, e.g., 'wt-wt'."},
            "safesearch": {
                "type": "string",
                "enum": ["off", "moderate", "strict"],
                "description": "Safe search level.",
            },
            "time": {
                "type": "string",
                "enum": ["d", "w", "m", "y"],
                "description": "Time window: day/week/month/year.",
            },
            "backend": {
                "type": "string",
                "enum": ["auto", "api", "html"],
                "description": "Backend implementation selection.",
            },
        },
        "required": ["q"],
        "additionalProperties": False,
    }

    def __init__(self, client: Any | None = None) -> None:
        self._client = client or _DefaultDDGSClient()

    def invoke(self, args: dict[str, Any]) -> list[dict[str, Any]]:
        q: str = args["q"]
        max_results: int = int(args.get("max_results", 5))
        region = args.get("region")
        safesearch = args.get("safesearch")
        time = args.get("time")
        backend = args.get("backend")

        iterable = self._client.text(
            q=q,
            max_results=max_results,
            region=region,
            safesearch=safesearch,
            time=time,
            backend=backend,
        )

        # Normalize shape for stability across client versions
        out: list[dict[str, Any]] = []
        for item in iterable:
            title = item.get("title") if isinstance(item, dict) else None
            href = item.get("href") if isinstance(item, dict) else None
            body = item.get("body") if isinstance(item, dict) else None
            if title is None and href is None and body is None:
                # Skip unexpected rows
                continue
            out.append(
                {
                    "title": title or "",
                    "href": href or "",
                    "body": body or "",
                    "source": "duckduckgo",
                }
            )
        return out
