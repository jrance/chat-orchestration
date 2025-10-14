from __future__ import annotations

from codeless_orchestrator.runtime.telemetry import InMemorySink, Telemetry


def test_basic_counters_and_timers() -> None:
    sink = InMemorySink()
    t = Telemetry(sink=sink, request_id="req-123", labels={"env": "test"})

    tok = t.start_timer("compile")
    # do nothing
    dur = t.stop_timer(tok)
    assert dur >= 0

    t.inc("steps", by=2)
    t.usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    snap = t.snapshot()

    assert snap["request_id"] == "req-123"
    assert snap["duration_ms_total"] >= 0
    assert snap["token_usage"]["total"] == 15


def test_tool_metrics_aggregation() -> None:
    sink = InMemorySink()
    t = Telemetry(sink=sink)

    # Simulate 3 tool calls with different durations and one error
    t.record_tool_call("web_search", duration_ms=10.0)
    t.record_tool_call("web_search", duration_ms=30.0)
    t.record_tool_call("web_search", duration_ms=50.0, error=True)
    snap = t.snapshot()

    tools = snap.get("tools") or {}
    ws = tools.get("web_search") or {}
    assert ws.get("calls") == 3
    assert ws.get("errors") == 1
    assert ws.get("max_ms") >= 50.0
    assert ws.get("p50_ms") is not None


def test_request_id_propagation_marker() -> None:
    # Telemetry stores request_id; server attaches it to events
    sink = InMemorySink()
    t = Telemetry(sink=sink, request_id="req-xyz")
    snap = t.snapshot()
    assert snap.get("request_id") == "req-xyz"

