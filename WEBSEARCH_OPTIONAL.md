# Making Web Search Optional

## Problem

The `primp` package (v0.15.0), which is a dependency of `ddgs` (DuckDuckGo Search), requires Cargo (Rust compiler) to install. This causes installation failures on corporate laptops where Rust/Cargo may not be available.

## Solution

The web search functionality has been made **optional**. The project can now be installed without requiring `ddgs` or `primp`.

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

### Basic Installation (without web search)
```bash
pip install -e .
```

### With Development Tools
```bash
pip install -e .[dev]
```

### With Web Search Support
```bash
pip install -e .[websearch]
```
**Note**: Requires Rust/Cargo to be installed

### All Features
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

1. Install the project without the `websearch` extra:
   ```bash
   pip install -e .[dev]
   ```

2. The web search tool (`tool:web-search`) will not be available, but all other functionality will work normally.

3. If you need web search functionality, you have these alternatives:
   - Request IT to install Rust/Cargo
   - Use an older version of `duckduckgo-search` that doesn't require `primp` (not recommended)
   - Implement a custom web search tool using a different HTTP client library

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
