# Installation Troubleshooting for Corporate Laptops

## Issue: Cargo/Rust Build Errors

If you see errors like:
- `Building wheel for primp (pyproject.toml) did not run successfully`
- `Building wheel for orjson (pyproject.toml) did not run successfully`
- `error: could not find Cargo`
- `error: rustc not found`

### Root Cause
Some Python packages are written in Rust for performance:
- `orjson` - Fast JSON serialization (optional `json` extra)
- `primp` - HTTP client used by `ddgs` (optional `websearch` extra)

These require Rust/Cargo compiler to install.

### Solution for Corporate Laptops

**Option 1: Install without Rust dependencies (Recommended)**

```powershell
# Uninstall any partial installation
pip uninstall -y codeless-orchestrator orjson ddgs primp

# Install with only pure-Python dependencies
pip install -e .[dev,sse,otel]
```

This gives you:
- ✅ Full core functionality
- ✅ All dev tools
- ✅ SSE streaming
- ✅ OpenTelemetry
- ❌ No web search (tool will be auto-disabled)
- ❌ No orjson (uses standard Python JSON instead)

**Option 2: Install Rust (if you have admin rights)**

```powershell
# Install Rust via rustup
# Visit: https://rustup.rs/
# Or via Chocolatey:
choco install rust

# Then install with all features
pip install -e .[websearch,json,sse,otel,dev]
```

**Option 3: Use pre-built wheels (if available)**

Some packages provide pre-built wheels for Windows. Update pip and try:

```powershell
python -m pip install --upgrade pip wheel
pip install -e .[websearch,json,sse,otel,dev]
```

## Verification

After installation, verify what's installed:

```powershell
# Check installed packages
pip list | Select-String "codeless|orjson|ddgs|primp"

# Test import
python -c "from codeless_orchestrator.server.app import app; print('✓ Import successful')"

# Run tests
pytest tests/test_smoke_import.py
```

## What Works Without Rust

Everything except:
- `tool:web-search` (DuckDuckGo search tool)
- `orjson` performance optimization

The application will:
- Automatically skip registering the web search tool if `ddgs` isn't installed
- Fall back to FastAPI's standard `JSONResponse` if `orjson` isn't available

Both fallbacks are graceful and require no code changes.

## Quick Reference

| Extra | Requires Rust? | Purpose |
|-------|---------------|---------|
| `dev` | ❌ No | Testing, linting, type checking |
| `sse` | ❌ No | Server-Sent Events streaming |
| `otel` | ❌ No | OpenTelemetry tracing |
| `json` | ⚠️ **YES** | High-performance JSON with orjson |
| `websearch` | ⚠️ **YES** | DuckDuckGo web search tool |

## Recommended Installation Commands

```powershell
# For most corporate laptops (no admin/Rust required)
pip install -e .[dev,sse,otel]

# If you have Rust installed
pip install -e .[websearch,json,sse,otel,dev]

# Minimal installation (no extras)
pip install -e .
```
