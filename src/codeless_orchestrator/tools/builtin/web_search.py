from __future__ import annotations

from collections.abc import Iterable
import os
from importlib import import_module
from pathlib import Path
import warnings
from typing import Any

from codeless_orchestrator.providers.settings import (
    get_httpx_proxies,
    get_httpx_verify,
)

try:
    _ddgs_module = import_module("ddgs")
except ModuleNotFoundError:  # pragma: no cover - fallback for older installations
    from duckduckgo_search import DDGS  # type: ignore[no-redef]
else:  # pragma: no cover - prefer the actively maintained package
    DDGS = getattr(_ddgs_module, "DDGS")

from ..base import BaseTool


class _DefaultDDGSClient:
    """Wrapper to abstract DDGS for simple DI in tests."""

    def __init__(self) -> None:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("ignore")
            proxy = get_httpx_proxies()
            verify = get_httpx_verify()
            
            # Apply CA bundle to environment variables
            # This ensures any HTTP client created internally will use these settings
            if isinstance(verify, str):
                self._apply_ca_bundle_env(verify)
            
            # CRITICAL FIX for ddgs 9.6.1+ with primp and custom CA bundles:
            # The new ddgs architecture creates primp.Client instances inside
            # HttpClient objects for each search engine. We need to:
            # 1. Create a custom primp.Client with ca_cert_file FIRST
            # 2. Patch the HttpClient class to use our custom client
            # 3. Then create the DDGS instance
            
            custom_http_client = None
            if isinstance(verify, str):
                # Create a custom HttpClient with CA bundle support if available
                custom_http_client = self._create_http_client_with_ca(proxy, verify)
            
            # Build DDGS kwargs
            kwargs: dict[str, Any] = {}
            if proxy is not None:
                kwargs["proxy"] = proxy
            
            # Handle verify parameter for DDGS
            # - If verify is a bool, pass it through.
            # - If verify is a string (CA bundle path), rely on env vars and do NOT pass 'verify'.
            if isinstance(verify, bool):
                kwargs["verify"] = verify
            
            # Create DDGS instance
            self._client = DDGS(**kwargs)
            
            # If we created a custom HttpClient, we need to patch the engines
            if custom_http_client is not None:
                self._patch_engines_http_client(custom_http_client)
            # Additionally, if an inner client exists and exposes 'ca_cert_file', set it.
            if isinstance(verify, str):
                self._apply_ca_to_inner_client(verify)

    def _patch_engines_http_client(self, http_client: Any) -> None:
        """Patch all engine instances to use our custom HttpClient.
        
        In ddgs 9.6.1+, each search engine has its own HttpClient instance.
        We need to replace them with our custom client that has ca_cert_file.
        """
        try:
            # Access the engines cache if it exists
            engines_cache = getattr(self._client, "_engines_cache", {})
            for engine in engines_cache.values():
                if hasattr(engine, "http_client"):
                    engine.http_client = http_client
        except Exception:
            # If patching fails, fall back to environment variables
            pass

    def _apply_ca_to_inner_client(self, ca_bundle_path: str) -> None:
        """If the DDGS instance exposes an inner client with 'ca_cert_file', set it.

        Some ddgs implementations expose an attribute like 'client' pointing to a
        lower-level HTTP client. In tests we rely on this to propagate the CA file.
        """
        try:
            inner = getattr(self._client, "client", None)
            if inner is not None and hasattr(inner, "ca_cert_file"):
                resolved = Path(ca_bundle_path).expanduser()
                if resolved.exists():
                    setattr(inner, "ca_cert_file", str(resolved))
        except Exception:
            # Best-effort; environment variables already applied
            pass

    # ddgs 9.6.1+ uses `query` parameter, older versions use `keywords`.
    # We support both for compatibility.
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
        kwargs: dict[str, Any] = {}
        if max_results is not None:
            kwargs["max_results"] = max_results
        if region is not None:
            kwargs["region"] = region
        if safesearch is not None:
            kwargs["safesearch"] = safesearch
        if time is not None:
            kwargs["timelimit"] = time
        if backend is not None:
            kwargs["backend"] = backend
        
        # Try new API first (query parameter), fall back to old API (keywords)
        try:
            return self._client.text(query=q, **kwargs)
        except TypeError:
            # Old API compatibility
            return self._client.text(keywords=q, **kwargs)

    @staticmethod
    def _apply_ca_bundle_env(path: str) -> None:
        resolved = Path(path).expanduser()
        if not resolved.exists():
            return
        str_path = str(resolved)
        os.environ.setdefault("SSL_CERT_FILE", str_path)
        os.environ.setdefault("REQUESTS_CA_BUNDLE", str_path)
        os.environ.setdefault("CURL_CA_BUNDLE", str_path)

    @staticmethod
    def _create_http_client_with_ca(proxy: str | None, ca_bundle_path: str) -> Any | None:
        """Create an HttpClient with a custom primp.Client that has ca_cert_file.
        
        For ddgs 9.6.1+, we need to create an HttpClient object that internally
        uses a primp.Client configured with the CA bundle.
        
        IMPORTANT: This must be called BEFORE creating a DDGS instance due to
        a bug in primp where creating a Client without ca_cert_file breaks all
        subsequent Client creations.
        """
        resolved = Path(ca_bundle_path).expanduser()
        if not resolved.exists():
            return None
        
        try:
            import primp
            # First, try to import the new HttpClient class from ddgs
            try:
                from ddgs.http_client import HttpClient
                
                # Create a custom primp.Client with ca_cert_file FIRST
                # (before HttpClient creates its own)
                custom_primp_client = primp.Client(
                    proxy=proxy,
                    timeout=10,
                    verify=True,
                    ca_cert_file=ca_bundle_path,
                    impersonate="chrome_131",  # Use a recent stable version
                    impersonate_os="windows",
                )
                
                # Create HttpClient instance
                http_client = HttpClient(proxy=proxy, timeout=10, verify=True)
                
                # Replace its internal primp.Client with our custom one
                http_client.client = custom_primp_client
                
                return http_client
                
            except ImportError:
                # Fall back to old API - create primp.Client directly
                return primp.Client(
                    proxy=proxy,
                    verify=True,
                    ca_cert_file=ca_bundle_path,
                    timeout=10,
                )
            
        except Exception:
            # If anything fails, return None and fall back to env vars
            return None


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
                "enum": ["off", "moderate", "strict", "on"],
                "description": "Safe search level. 'strict' inputs are mapped to 'on' for newer DDGS versions.",
            },
            "time": {
                "type": "string",
                "enum": ["d", "w", "m", "y"],
                "description": "Time window: day/week/month/year.",
            },
            "backend": {
                "type": "string",
                "enum": ["auto", "html", "lite", "bing", "api"],
                "description": "Backend implementation selection. Legacy 'api' inputs degrade to 'auto'.",
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
        safesearch = self._normalize_safesearch(args.get("safesearch"))
        time = args.get("time")
        backend = self._normalize_backend(args.get("backend"))

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

    @staticmethod
    def _normalize_safesearch(value: Any | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            return value
        lowered = value.lower()
        if lowered == "strict":
            return "on"
        if lowered in {"on", "moderate", "off"}:
            return lowered
        return value

    @staticmethod
    def _normalize_backend(value: Any | None) -> str | None:
        if value is None or not isinstance(value, str):
            return value
        lowered = value.lower()
        if lowered in {"api", "ecosia"}:
            return "auto"
        if lowered in {"auto", "html", "lite", "bing"}:
            return lowered
        return value
