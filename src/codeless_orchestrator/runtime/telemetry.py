from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass
from statistics import median
from typing import Any, Dict, Final, List, Optional


class TelemetrySink:
    """Interface for telemetry event sinks."""

    def on_event(self, event: dict) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def flush(self) -> None:  # pragma: no cover - interface
        pass


class InMemorySink(TelemetrySink):
    """Simple in-memory sink for tests and response embedding."""

    def __init__(self) -> None:
        self.events: list[dict] = []

    def on_event(self, event: dict) -> None:
        # Store a shallow copy to prevent accidental external mutation
        self.events.append(dict(event))

    def flush(self) -> None:  # pragma: no cover - no-op
        return


class OTLPSink(TelemetrySink):
    """Basic OTLP exporter using OpenTelemetry if installed and enabled.

    We intentionally keep this minimal; production users can extend.
    """

    def __init__(self) -> None:
        # Lazy import guarded by extras and env in deps
        try:  # pragma: no cover - exercised only with extra installed
            from opentelemetry import trace  # type: ignore
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (  # type: ignore
                OTLPSpanExporter,
            )
            from opentelemetry.sdk.resources import Resource  # type: ignore
            from opentelemetry.sdk.trace import TracerProvider  # type: ignore
            from opentelemetry.sdk.trace.export import BatchSpanProcessor  # type: ignore

            endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
            service_name = os.getenv("OTEL_SERVICE_NAME", "codeless-orchestrator")
            resource = Resource.create({"service.name": service_name})
            provider = TracerProvider(resource=resource)
            processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
            provider.add_span_processor(processor)
            trace.set_tracer_provider(provider)
            self._tracer = trace.get_tracer(__name__)
        except Exception as e:  # pragma: no cover - optional path
            # Fallback to no-op if OTLP not configured; users opt-in via deps
            self._tracer = None

    def on_event(self, event: dict) -> None:  # pragma: no cover - optional path
        if not self._tracer:
            return
        # Represent a run as a span with attributes
        name = event.get("provider", "run")
        with self._tracer.start_as_current_span(str(name)) as span:
            try:
                for k, v in event.items():
                    # Flatten nested common fields for quick dashboards
                    if isinstance(v, dict):
                        for sk, sv in v.items():
                            span.set_attribute(f"{k}.{sk}", sv)
                    else:
                        span.set_attribute(k, v)
            except Exception:
                # Best-effort only
                pass

    def flush(self) -> None:  # pragma: no cover - no-op
        return


_MAX_TOOL_SAMPLES: Final[int] = 200


@dataclass
class _TimerToken:
    name: str
    start: float


class Telemetry:
    """Run-scoped telemetry aggregator."""

    def __init__(
        self,
        *,
        sink: TelemetrySink,
        request_id: str | None = None,
        labels: dict[str, str] | None = None,
    ) -> None:
        self.run_id: str = str(uuid.uuid4())
        self.request_id: str | None = request_id
        self.labels: dict[str, str] = dict(labels or {})
        self._sink = sink
        self._t0 = time.perf_counter()

        # Aggregates
        self._counters: dict[str, int] = {}
        self._errors: list[dict[str, str]] = []
        self._provider: str | None = None
        self._model_id: str | None = None
        self._token_usage: dict[str, int | None] | None = None

        # Per-turn
        self._turns: list[dict[str, Any]] = []

        # Tool metrics: name -> stats
        self._tool_stats: dict[str, dict[str, Any]] = {}

    # ---------------------- Timers ----------------------
    def start_timer(self, name: str) -> _TimerToken:
        return _TimerToken(name=name, start=time.perf_counter())

    def stop_timer(self, token: _TimerToken) -> float:
        dur_ms = (time.perf_counter() - token.start) * 1000.0
        # Record generic observation under timers.<name>
        key = f"timers.{token.name}"
        self.observe(key, dur_ms)
        return dur_ms

    # ---------------------- Counters/observations ----------------------
    def inc(self, name: str, *, by: int = 1) -> None:
        self._counters[name] = int(self._counters.get(name, 0)) + int(by)

    def observe(self, name: str, value: float) -> None:
        # Generic numeric observation; we store as a simple list
        # Keep memory bounded: store only a modest number of samples per key
        buf: list[float] = self._tool_stats.setdefault("__generic__", {}).setdefault(name, [])
        buf.append(float(value))
        if len(buf) > _MAX_TOOL_SAMPLES:
            del buf[0 : len(buf) - _MAX_TOOL_SAMPLES]

    # ---------------------- Usage/errors ----------------------
    def usage(
        self,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        total_tokens: int | None = None,
    ) -> None:
        self._token_usage = {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "total": total_tokens,
        }

    def record_error(self, kind: str, message: str) -> None:
        self._errors.append({"kind": kind, "message": message})
        self.inc("errors", by=1)

    # ---------------------- Provider/model ----------------------
    def set_provider_model(self, provider: str | None, model_id: str | None) -> None:
        if provider and not self._provider:
            self._provider = provider
        if model_id and not self._model_id:
            self._model_id = model_id

    # ---------------------- Turns ----------------------
    def record_turn(
        self,
        *,
        duration_ms: float,
        finish_reason: str | None,
        had_tool_calls: bool,
    ) -> None:
        idx = len(self._turns)
        self._turns.append(
            {
                "idx": idx,
                "duration_ms": max(0.0, float(duration_ms)),
                "finish_reason": finish_reason,
                "had_tool_calls": had_tool_calls,
            }
        )

    # ---------------------- Tools ----------------------
    def record_tool_call(self, name: str, *, duration_ms: float | None = None, error: bool = False) -> None:
        st = self._tool_stats.setdefault(name, {"calls": 0, "errors": 0, "latencies_ms": []})
        st["calls"] = int(st.get("calls", 0)) + 1
        if error:
            st["errors"] = int(st.get("errors", 0)) + 1
            self.inc("tool_errors", by=1)
        if duration_ms is not None:
            buf: list[float] = st.setdefault("latencies_ms", [])
            buf.append(float(duration_ms))
            if len(buf) > _MAX_TOOL_SAMPLES:
                del buf[0 : len(buf) - _MAX_TOOL_SAMPLES]
        self.inc("tool_calls", by=1)

    # ---------------------- Snapshot/export ----------------------
    def _aggregate_tool_metrics(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, st in self._tool_stats.items():
            if name == "__generic__":
                continue
            lats: list[float] = list(st.get("latencies_ms", []) or [])
            lats_sorted = sorted(lats)
            p50 = median(lats_sorted) if lats_sorted else None
            # p95 estimation
            p95 = None
            if lats_sorted:
                idx = max(0, min(len(lats_sorted) - 1, int(round(0.95 * (len(lats_sorted) - 1)))))
                p95 = lats_sorted[idx]
            out[name] = {
                "calls": int(st.get("calls", 0)),
                "errors": int(st.get("errors", 0)),
                "p50_ms": p50,
                "p95_ms": p95,
                "max_ms": max(lats_sorted) if lats_sorted else None,
            }
        return out

    def snapshot(self) -> dict[str, Any]:
        total_ms = (time.perf_counter() - self._t0) * 1000.0
        data: dict[str, Any] = {
            "run_id": self.run_id,
            "request_id": self.request_id,
            "steps": len(self._turns),
            "duration_ms_total": total_ms,
            "provider": self._provider,
            "model_id": self._model_id,
        }
        if self._token_usage is not None:
            data["token_usage"] = self._token_usage
        if self._turns:
            data["turns"] = list(self._turns)
        if self._errors:
            data["errors"] = list(self._errors)
        tools = self._aggregate_tool_metrics()
        if tools:
            data["tools"] = tools
        if self.labels:
            data["labels"] = dict(self.labels)

        # Emit to sink for long-lived collectors
        try:
            self._sink.on_event(data)
        except Exception:
            # Never fail the request because of telemetry
            pass
        return data


__all__ = [
    "TelemetrySink",
    "InMemorySink",
    "OTLPSink",
    "Telemetry",
]

