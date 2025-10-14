Deployment Guide

- Image: Built from `python:3.13-slim`, minimal, non-root runtime.
- App: FastAPI served by Gunicorn with Uvicorn workers.
- Health: `GET /healthz` returns `{ "status": "ok" }`.

Quick Start

- Build: `docker build -t codeless-orchestrator:local .`
- Run: `docker run --rm -p 8000:8000 -e OPENAI_API_KEY=... codeless-orchestrator:local`
- Health: `curl http://localhost:8000/healthz`
- Compose: `docker compose -f deploy/docker-compose.yml up --build`

Runtime Environment Variables

- `PORT` (default `8000`): HTTP port to bind.
- `WEB_CONCURRENCY` (default `2`): Gunicorn workers. Typically set to CPU cores.
- `WORKER_TIMEOUT` (default `0`): Worker timeout in seconds. Use `0` for SSE/streaming safety.
- `LOG_LEVEL` (default `info`): `debug|info|warning|error`.
- `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_ORG`: OpenAI credentials.
- `GOOGLE_API_KEY`: Google Gemini API key.
- `ORG_PREAMBLE`, `ORG_PREAMBLE_FILE`: Optional org preamble string or file path.
- `TELEMETRY_ENABLE` (0/1), `OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_SERVICE_NAME`: OTLP telemetry.
- `CORS_ORIGINS`: Comma-separated origins for CORS (optional).

Gunicorn/Uvicorn Settings

- Workers: `WEB_CONCURRENCY` (e.g., `2`–`4` for typical small instances).
- Keep-alive: 75 seconds to match common reverse proxies.
- Timeout: `WORKER_TIMEOUT=0` recommended for SSE/streaming.

Reverse Proxy (Nginx) for SSE

- Use the following snippet to support streaming/SSE:

  - `proxy_set_header Host $host;`
  - `proxy_http_version 1.1;`
  - `proxy_set_header Connection "";`
  - `proxy_buffering off;`
  - `proxy_read_timeout 3600s;`
  - `send_timeout 3600s;`

Kubernetes

- Apply manifests from `deploy/k8s/`:
  - `deployment.yaml`: sets non-root, liveness/readiness probes, resources.
  - `service.yaml`: ClusterIP service on port 80 → 8000.
  - `hpa.yaml`: optional autoscaling by CPU.

Telemetry (OTLP)

- Enable by setting `TELEMETRY_ENABLE=1` and `OTEL_EXPORTER_OTLP_ENDPOINT`.
- Install OTLP extras are already included in the image build stage.

Security Posture

- Non-root user (`uid 10001`), minimal OS packages, no secrets baked in.
- Configure secrets via environment variables or orchestration secret stores.

