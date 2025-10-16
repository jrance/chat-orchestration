# Codeless Orchestrator

Codeless agent orchestration engine built on LangGraph and FastAPI. This repository is the bootstrap foundation: clean packaging, quality gates, tests, and CI — ready for iterative feature work.

## Overview

This project aims to provide a production-ready, codeless orchestration layer for LLM-powered agents. It leverages LangGraph for graph-based control flow and FastAPI for a robust HTTP interface. In this initial PR, we focus on packaging, structure, and quality tooling - no business logic yet.

## Features (planned)

- LangGraph-based orchestration primitives
- Provider abstraction (OpenAI, Google, others)
- Tooling integration (web search, function tools)
- FastAPI service with streaming support
- Observability and tracing hooks

## Install

Python 3.10–3.13 is supported.

```
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Unix
# source .venv/bin/activate

pip install -e .[dev]
```

### Optional Dependencies

The project includes several optional feature sets that can be installed as needed:

- **Web Search**: `pip install -e .[websearch]` - Adds DuckDuckGo web search capabilities ⚠️ **Requires Rust/Cargo**
- **JSON Performance**: `pip install -e .[json]` - Uses orjson for faster JSON processing ⚠️ **Requires Rust/Cargo**
- **Server-Sent Events**: `pip install -e .[sse]` - Enables SSE streaming endpoints (Pure Python ✓)
- **OpenTelemetry**: `pip install -e .[otel]` - Adds distributed tracing support (Pure Python ✓)

**For corporate laptops without Rust/Cargo:**
```bash
# Install without Rust dependencies
pip install -e .[dev,sse,otel]
```

**To install all optional dependencies (requires Rust/Cargo):**
```bash
pip install -e .[websearch,json,sse,otel,dev]
```

**Note**: The `websearch` and `json` extras require Rust/Cargo to be installed:
- `websearch` uses `ddgs` which depends on `primp` (Rust-based HTTP client)
- `json` uses `orjson` (Rust-based JSON serializer)

On corporate laptops where Cargo may not be available, you can skip these optional dependencies:
- Web search tool will be gracefully disabled
- The server will fall back to standard Python JSON serialization

## Dev Quickstart

```
ruff check
mypy src
pytest --maxfail=1 -q --cov=src --cov-report=term-missing
```

## Examples

- See `docs/examples.md` for ready-to-run configs and curl/SSE demos.
- Example configs are in `examples/configs/` and can be posted directly to the server.
- Minimal SSE client: `examples/scripts/sse_client.py`.

Run locally:
- Start server: `uvicorn codeless_orchestrator.server.app:app --reload`
- Execute (non-stream): see `examples/requests/execute.http`
- Execute (SSE): see `examples/requests/stream.http`

## Config Schema

- See docs/config-schema.md for a high-level overview of the codeless graph configuration, field normalization (camelCase ↔ snake_case), and validation behavior.
- Example usage:

```
from codeless_orchestrator.config.loader import load_graph

cfg, report = load_graph("path/to/graph.json")
if report.has_errors():
    raise SystemExit("Invalid graph configuration")
```

## Project Roadmap

This PR establishes the foundation. Upcoming PRs will introduce the core engine, provider integrations, API endpoints, and streaming. See `docs/architecture.md` for the high-level structure and `docs/decisions/` for ADRs.

## Tools

A modular tools framework is available. See `docs/tools.md` for how to define tools, register them, enforce overrides and timeouts, and expose provider tool specs.

## License

Licensed under Apache-2.0. See `LICENSE` (to be added).

## Commands To Run Locally

- Create venv: `python -m venv .venv && .\\.venv\\Scripts\\activate` (Windows) or `source .venv/bin/activate` (Unix)
- Install dev deps: `pip install -e .[dev]`
- Lint: `ruff check`
- Type-check: `mypy src`
- Test: `pytest --maxfail=1 -q --cov=src`
