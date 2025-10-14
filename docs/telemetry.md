Telemetry & Observability

What’s Captured

- Run correlation: `run_id`, `request_id`
- Durations: total runtime and timer samples for `compile` and `execute`
- Provider/model: `provider`, `model_id`
- Token usage: `{ prompt, completion, total }`
- Turns: `[{ idx, duration_ms, finish_reason, had_tool_calls }]`
- Tools: per-tool `{ calls, errors, p50_ms, p95_ms, max_ms }`
- Errors: list of `{ kind: provider|tool|engine, message }`
- Labels: user-provided labels from config

Enable/Disable

- Responses include a `metrics` object when the agent config has `telemetry.emitUsage = true` (default true).
- For streaming (SSE), the final `message.end` event includes `metrics` under the same flag. All events include `request_id`.

OTLP Export (Optional)

- Install extras: `pip install -e .[otel]`
- Set env:
  - `TELEMETRY_ENABLE=1`
  - `OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317`
  - Optional: `OTEL_SERVICE_NAME=codeless-orchestrator`
- The server uses an in-memory sink by default; when enabled and configured, an OTLP sink exports run spans with flattened attributes suitable for basic dashboards.

Common Dashboards

- Latency histograms: per-turn `duration_ms` and per-tool `p50/p95/max`
- Error rates: provider and tool error counts
- Token usage trends: `token_usage.total`

Local Run

- Install dev: `pip install -e .[dev]`
- Start: `uvicorn codeless_orchestrator.server.app:app --reload`
- Try:
  - POST `/execute` with `telemetry.emitUsage=true` in config → response includes `metrics`
  - POST `/execute/stream` and confirm final `message.end` includes `metrics` and all events include `request_id`

