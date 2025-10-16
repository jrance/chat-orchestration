# Making Web Search and JSON Extras Optional

## Problem

Both `primp` (v0.15.0) and `orjson` packages require Cargo (Rust compiler) to install. This causes installation failures on corporate laptops where Rust/Cargo may not be available:

- `primp` is a dependency of `ddgs` (DuckDuckGo Search) - used for web search
- `orjson` is used for high-performance JSON serialization

## Solution

Both web search and high-performance JSON functionality have been made **optional**. The project can now be installed without requiring any Rust dependencies.

## Changes Made

### 1. `pyproject.toml`
- Moved `ddgs>=9.6.1,<10` from required dependencies to optional dependencies
- Created a new optional dependency group: `websearch = ["ddgs>=9.6.1,<10"]`

### 2. `README.md`
- Added an "Optional Dependencies" section explaining all available extras
- Documented that `websearch` requires Rust/Cargo
- Explained that web search will be gracefully disabled if not installed

### 3. `docs/tools.md`
- Added a note to the "Builtin: Web Search" section
- Documented that the tool requires the optional `websearch` dependency
- Mentioned graceful fallback behavior

## Installation Options

### For Corporate Laptops (No Rust/Cargo Required)
```bash
# Basic installation with development tools
pip install -e .[dev]

# With SSE and OpenTelemetry support (both are pure Python)
pip install -e .[dev,sse,otel]
```

### With Web Search Support (Requires Rust/Cargo)
```bash
pip install -e .[websearch,dev]
```

### With High-Performance JSON (Requires Rust/Cargo)
```bash
pip install -e .[json,dev]
```

### All Features (Requires Rust/Cargo)
```bash
pip install -e .[websearch,json,sse,otel,dev]
```

## How It Works

The code already had graceful fallback handling in `src/codeless_orchestrator/tools/registry.py`:

```python
def register_default_tools(registry: ToolRegistry | None = None) -> None:
    """Register builtin tools in the provided registry (or default)."""
    reg = registry or default_registry
    try:
        from .builtin.web_search import WebSearchTool
    except Exception:  # pragma: no cover - import issues should not break process
        return

    reg.register(
        ToolEntry(tool_id="tool:web-search", version=None, impl=WebSearchTool())
    )
```

If `ddgs` is not installed, the import will fail and the web search tool simply won't be registered. No errors will be raised during startup.

## For Corporate Laptop Users

If you cannot install Cargo/Rust on your corporate laptop:

1. Install the project without the `websearch` and `json` extras:
   ```bash
   pip install -e .[dev,sse,otel]
   ```

2. What you get:
   - ✅ All core functionality (LangGraph, FastAPI, agents, etc.)
   - ✅ Development tools (pytest, ruff, mypy)
   - ✅ SSE streaming support
   - ✅ OpenTelemetry tracing
   - ❌ Web search tool (`tool:web-search`) will not be available
   - ❌ High-performance JSON serialization (falls back to standard Python JSON)

3. If you need web search or orjson functionality later, you have these alternatives:
   - Request IT to install Rust/Cargo
   - Use an older version of packages that don't require Rust (not recommended)
   - Implement a custom web search tool using a different HTTP client library
   - Use standard Python JSON (performance difference is minimal for most use cases)

## Testing

To verify the installation works without web search:

```bash
# Create fresh virtual environment
python -m venv test_venv
.\test_venv\Scripts\activate  # Windows
# source test_venv/bin/activate  # Linux/Mac

# Install without websearch
pip install -e .[dev]

# Run tests (web search tests may be skipped)
pytest
```
