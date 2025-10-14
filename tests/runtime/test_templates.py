from __future__ import annotations

from codeless_orchestrator.runtime.templates import render, merge_vars


def test_render_missing_key_passthrough() -> None:
    text = "Hello {name}"
    out = render(text, {})
    assert out == "Hello {name}"


def test_merge_vars_precedence() -> None:
    engine = {"a": 1, "b": 2}
    config = {"b": 3, "c": 4}
    request = {"c": 5, "d": 6}
    merged = merge_vars(engine, config, request)
    # later overrides earlier
    assert merged == {"a": 1, "b": 3, "c": 5, "d": 6}

