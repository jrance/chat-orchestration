from __future__ import annotations

from collections.abc import Iterable
import os
from typing import Any

from codeless_orchestrator.tools.builtin.web_search import WebSearchTool
from codeless_orchestrator.tools.builtin.web_search import _DefaultDDGSClient


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


def test_legacy_values_are_normalized() -> None:
    recorded: dict[str, Any] = {}

    class RecordingClient:
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
            recorded["safesearch"] = safesearch
            recorded["backend"] = backend
            return []

    tool = WebSearchTool(client=RecordingClient())
    tool.invoke({"q": "openai", "safesearch": "strict", "backend": "api"})

    assert recorded["safesearch"] == "on"
    assert recorded["backend"] == "auto"


def test_default_ddgs_client_uses_proxy_settings(monkeypatch, tmp_path) -> None:
    recorded: dict[str, Any] = {}

    class RecordingInnerClient:
        def __init__(self) -> None:
            self.ca_cert_file: str | None = None

    class RecordingDDGS:
        def __init__(self, **kwargs: Any) -> None:
            recorded.update(kwargs)
            self.client = RecordingInnerClient()

        def text(self, **_: Any) -> list[dict[str, Any]]:
            return []

    proxy_url = "http://proxy.example.com:8080"
    ca_path = tmp_path / "proxy-ca.pem"
    ca_path.write_text("dummy")

    monkeypatch.setenv("PROXY_ENABLED", "1")
    monkeypatch.setenv("PROXY_URL", proxy_url)
    monkeypatch.setenv("PROXY_CA_BUNDLE", str(ca_path))
    monkeypatch.delenv("PROXY_VERIFY", raising=False)
    monkeypatch.setattr("codeless_orchestrator.tools.builtin.web_search.DDGS", RecordingDDGS)
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising=False)
    monkeypatch.delenv("CURL_CA_BUNDLE", raising=False)

    client = _DefaultDDGSClient()

    assert recorded["proxy"] == proxy_url
    assert "verify" not in recorded
    assert client.text(q="openai") == []
    assert client._client.client.ca_cert_file == str(ca_path)
    assert os.environ["SSL_CERT_FILE"] == str(ca_path)
    assert os.environ["REQUESTS_CA_BUNDLE"] == str(ca_path)
    assert os.environ["CURL_CA_BUNDLE"] == str(ca_path)
