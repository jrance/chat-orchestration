Run Locally

- Install: `pip install -e .[dev]`
- Start: `uvicorn codeless_orchestrator.server.app:app --reload`

Environment

- `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_ORG` (optional)
- `GOOGLE_API_KEY` (or `GEMINI_API_KEY`)
- `LOG_LEVEL` for server logs
- Telemetry / OTLP (optional):
  - `TELEMETRY_ENABLE=1` to enable OTLP export (defaults to disabled)
  - `OTEL_EXPORTER_OTLP_ENDPOINT` (e.g. `http://otel-collector:4317`)
  - `OTEL_SERVICE_NAME` (defaults to `codeless-orchestrator`)

Features

- CORS enabled (all origins) and `X-Request-ID` propagation
- Health: `GET /healthz` returns `{ "status": "ok" }`
- SSE streaming via `/execute/stream` with fallback if `sse-starlette` is not installed

Request IDs

- The server sets and propagates `X-Request-ID`. You can pass your own via header or request options. The header is echoed and included on all SSE events.

Telemetry

- See `docs/telemetry.md` for captured fields and OTLP setup.
- When `telemetry.emitUsage` is true in the agent config, non-stream responses include a `metrics` object and the final `message.end` SSE event includes `metrics`.

Notes

- The server validates configs on `/validate` without failing the request; errors are returned in the payload.
- `/compile` and `/execute` call the engine/compiler and will return 4xx/5xx on invalid configs or provider failures.
- For SSE, heartbeats are sent when `sse-starlette` is available; the manual fallback streams `data:` frames without explicit pings.
