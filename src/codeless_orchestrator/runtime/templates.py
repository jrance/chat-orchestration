from __future__ import annotations

from typing import Any


class _SafeDict(dict):
    """A dict that leaves missing keys as `{key}` during format_map.

    This allows simple template placeholders to survive if not provided,
    rather than raising a KeyError.
    """

    def __missing__(self, key: str) -> str:  # pragma: no cover - trivial
        return "{" + key + "}"


def render(text: str, vars: dict[str, Any] | None) -> str:
    """Render `text` using Python's str.format_map with a safe mapping.

    Missing keys are preserved as `{key}`. If `vars` is None or empty,
    returns the original text.
    """
    if not text:
        return ""
    mapping = _SafeDict(vars or {})
    try:
        return str(text).format_map(mapping)
    except Exception:
        # In case of non-mapping content or formatting issues, fall back to original
        return str(text)


def merge_vars(*sources: dict[str, Any] | None) -> dict[str, Any]:
    """Merge multiple var dicts with later sources overriding earlier.

    Example precedence: engine-level < config context.vars < per-request vars
    """
    merged: dict[str, Any] = {}
    for src in sources:
        if not src:
            continue
        for k, v in src.items():
            merged[k] = v
    return merged


__all__ = ["render", "merge_vars"]

