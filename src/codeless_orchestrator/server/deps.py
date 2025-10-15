from __future__ import annotations

import threading
from typing import Callable
import os
from functools import lru_cache

from ..engine.executor import OrchestrationEngine
from ..engine.types import ModelParams, ProviderResolver
from ..logging import get_logger as _get_logger
from ..providers.gemini import GeminiProvider
from ..providers.openai import OpenAIProvider
from ..tools.registry import ToolRegistry, default_registry, register_default_tools
from ..runtime.telemetry import InMemorySink, OTLPSink, Telemetry, TelemetrySink
from fastapi import Request

_engine_singleton: OrchestrationEngine | None = None
_engine_lock = threading.Lock()

_registry_singleton: ToolRegistry | None = None
_registry_lock = threading.Lock()

# Eagerly register builtin tools at import time so routes don't pay import cost
# during first request (helps timing-sensitive tests).
try:  # pragma: no cover - import-time optimization
    register_default_tools(default_registry)
except Exception:
    # Best-effort only; get_registry() will retry lazily.
    pass


def get_engine() -> OrchestrationEngine:
    global _engine_singleton
    if _engine_singleton is None:
        with _engine_lock:
            if _engine_singleton is None:
                _engine_singleton = OrchestrationEngine()
    return _engine_singleton


def get_registry() -> ToolRegistry:
    """Return a ToolRegistry with builtin tools registered once."""
    global _registry_singleton
    if _registry_singleton is None:
        with _registry_lock:
            if _registry_singleton is None:
                # Use the module-level default registry to preserve global tool state
                register_default_tools(default_registry)
                _registry_singleton = default_registry
    return _registry_singleton


def get_provider_resolver() -> ProviderResolver:
    def _resolver(mp: ModelParams):
        if mp.provider == "openai":
            return OpenAIProvider()
        if mp.provider == "gemini":
            return GeminiProvider()
        # Let engine/compiler raise with a nicer error later
        return OpenAIProvider()

    return _resolver


def get_logger():  # pragma: no cover - thin wrapper
    return _get_logger("codeless_orchestrator.server")


@lru_cache(maxsize=1)
def get_org_preamble() -> str | None:
    """Read organization preamble from env or file once.

    Priority:
    - If ORG_PREAMBLE_FILE is set and the file exists, read UTF-8 content.
    - Else if ORG_PREAMBLE is set, use its value.
    - Else return None.
    """
    path = os.environ.get("ORG_PREAMBLE_FILE")
    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read().strip()
                return text or None
        except Exception:
            # Fall through to ORG_PREAMBLE env if file can't be read
            pass
    val = os.environ.get("ORG_PREAMBLE")
    if val is not None:
        v = val.strip()
        return v or None
    return None


# ---------------- Telemetry DI ----------------
def get_telemetry_sink() -> TelemetrySink:
    """Return a telemetry sink based on env flags.

    - If TELEMETRY_ENABLE=1 and OTEL_EXPORTER_OTLP_ENDPOINT set, use OTLPSink
    - Else, use InMemorySink
    """
    enabled = os.environ.get("TELEMETRY_ENABLE") == "1"
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if enabled and endpoint:
        try:  # pragma: no cover - only with extra installed
            return OTLPSink()
        except Exception:
            # Fallback on any failure
            return InMemorySink()
    return InMemorySink()


def build_telemetry(request: Request, labels: dict[str, str] | None = None) -> Telemetry:
    """Construct a request-scoped Telemetry aggregator using the configured sink."""
    # Try to propagate request id from middleware if present
    req_id = None
    try:
        req_id = getattr(request.state, "request_id", None)
    except Exception:
        req_id = None
    sink = get_telemetry_sink()
    return Telemetry(sink=sink, request_id=req_id, labels=labels)
