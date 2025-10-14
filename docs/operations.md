Operations Guide

Startup and Serving

- Entrypoint prints Python version, then executes startup.
- Gunicorn runs Uvicorn workers serving `codeless_orchestrator.server.app:app`.
- Reverse proxy recommended for TLS and connection management.

Health and Readiness

- Liveness/Readiness: `GET /healthz`.
- Readiness should gate traffic until the app is responsive.

Graceful Shutdown

- SIGTERM triggers Gunicorn graceful shutdown; inflight requests finish.
- Use `--graceful-timeout 30` and `--keep-alive 75` (already configured).

Scaling

- Vertical: increase `WEB_CONCURRENCY` roughly to CPU cores (or `cores * 2` for mixed workloads). Validate with load tests.
- Horizontal: scale pods/replicas; enable HPA on CPU or custom metrics.
- Ensure reverse proxy and clients handle long-lived SSE connections.

Reverse Proxy Tips (Nginx)

- `proxy_set_header Host $host;`
- `proxy_http_version 1.1;`
- `proxy_set_header Connection "";`
- `proxy_buffering off;`
- `proxy_read_timeout 3600s;`
- `send_timeout 3600s;`

Secrets and Key Rotation

- Deliver API keys via env vars or secret managers.
- Rotate by updating the secret and restarting the deployment.

Resource Footprint

- Baseline: ~100–200 MiB RAM per worker depending on traffic and providers.
- Tune worker count and memory limits to avoid OOM under load.

Observability

- Set `TELEMETRY_ENABLE=1` and `OTEL_EXPORTER_OTLP_ENDPOINT` to export spans.
- Set `OTEL_SERVICE_NAME` (default: `codeless-orchestrator`).

