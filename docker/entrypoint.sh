#!/usr/bin/env sh
set -e

echo "Starting codeless-orchestrator…"
echo "Python: $(python -V)"

exec "$@"

