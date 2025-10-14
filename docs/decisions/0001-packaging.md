# ADR 0001: Packaging, Pins, and Python Versions

## Status
Accepted

## Context

We need a predictable, low-overhead build and packaging setup to enable quick iteration while keeping quality gates (lint, type checking, tests) strict and reproducible.

## Decision

- Build backend: `hatchling` for minimal configuration and `src/` layout.
- Project metadata in `pyproject.toml` with `name = "codeless-orchestrator"` and `requires-python = ">=3.10"`.
- Runtime dependencies pinned with safe upper bounds (`<`) to reduce breakage from upstream changes.
- Optional extras:
  - `json`: `orjson` for improved JSON performance in later streaming work.
  - `sse`: `sse-starlette` for SSE helpers.
- Dev extras include `ruff`, `mypy`, `pytest`, `pytest-asyncio`, and `coverage`.
- CI runs across Python 3.10–3.13 for broad compatibility.

## Consequences

- Reproducible installs and CI behavior.
- Easy local development (`pip install -e .[dev]`).
- Clear path to extend tooling and tests without changing the build system.

