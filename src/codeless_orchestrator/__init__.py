from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

__all__ = ["__version__"]

try:
    # The distribution name is hyphenated by design; module is underscored.
    __version__ = version("codeless-orchestrator")
except PackageNotFoundError:  # pragma: no cover - fallback for editable/uninstalled state
    __version__ = "0.0.0"

