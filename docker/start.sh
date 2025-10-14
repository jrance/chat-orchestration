#!/usr/bin/env sh
set -e

HOST="0.0.0.0"
APP="codeless_orchestrator.server.app:app"

exec gunicorn "$APP" \
  -k uvicorn.workers.UvicornWorker \
  -w "${WEB_CONCURRENCY:-2}" \
  -b "${HOST}:${PORT:-8000}" \
  --timeout "${WORKER_TIMEOUT:-0}" \
  --graceful-timeout 30 \
  --keep-alive 75 \
  --log-level "${LOG_LEVEL:-info}" \
  --access-logfile '-' --error-logfile '-'

