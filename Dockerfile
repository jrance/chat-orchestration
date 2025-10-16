FROM python:3.13-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

FROM base AS builder
COPY pyproject.toml README.md /app/
COPY src /app/src

# Optional: Install Rust/Cargo for websearch support (requires ddgs -> primp)
# Uncomment the following lines to enable web search functionality:
# RUN apt-get update \
#     && apt-get install -y --no-install-recommends curl build-essential \
#     && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
#     && . $HOME/.cargo/env \
#     && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv \
    && . /opt/venv/bin/activate \
    && pip install --upgrade pip \
    && pip install -e .[sse] \
    && pip install gunicorn

# Optional: Install web search support (requires Rust/Cargo - see commented lines above)
# RUN . /opt/venv/bin/activate && . $HOME/.cargo/env && pip install -e .[websearch]

# Optional OTLP extras (install only if you plan to use telemetry export)
RUN . /opt/venv/bin/activate && pip install -e .[otel]

FROM base AS runtime
ENV PATH="/opt/venv/bin:$PATH" \
    PORT=8000 \
    WEB_CONCURRENCY=2 \
    WORKER_TIMEOUT=0 \
    LOG_LEVEL=info

# Create non-root user
RUN useradd -m -u 10001 appuser

COPY --from=builder /opt/venv /opt/venv
COPY docker/entrypoint.sh docker/start.sh /app/docker/
RUN chmod +x /app/docker/*.sh && chown -R appuser:appuser /app

USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --retries=3 CMD curl -fsS http://127.0.0.1:${PORT}/healthz || exit 1

ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["/app/docker/start.sh"]

