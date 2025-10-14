# Contributing

Thanks for your interest in contributing! This repo is set up for clean, incremental changes with high signal-to-noise in PRs.

## Development Environment

- Python 3.10–3.13
- Install dev dependencies: `pip install -e .[dev]`

## Code Style

- Lint: `ruff check`
- Type check: `mypy src`
- Tests: `pytest --maxfail=1 -q --cov=src`

## Branch & PR Guidelines

- Prefer small, focused PRs with descriptive titles.
- Include rationale in the description, link ADRs when relevant.
- Ensure CI passes (lint, type, tests) before requesting review.

## Commit Messages

- Use imperative mood: “Add X”, “Fix Y”.
- Reference issues where applicable.

## ADRs

Architectural decisions live in `docs/decisions/`. Add a new sequentially numbered file when making notable decisions.

