from __future__ import annotations

import os
from typing import Any

OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
OPENAI_ORG_ENV = "OPENAI_ORG"
OPENAI_BASE_URL_ENV = "OPENAI_BASE_URL"
OPENAI_TIMEOUT_ENV = "OPENAI_TIMEOUT"
GEMINI_API_KEY_ENV = "GOOGLE_API_KEY"
GEMINI_API_KEY_FALLBACK_ENV = "GEMINI_API_KEY"
GEMINI_TIMEOUT_ENV = "GEMINI_TIMEOUT"


def _get_env_float(name: str) -> float | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def get_openai_client_kwargs(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    organization: str | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Build kwargs for OpenAI v2 client construction based on env and overrides."""
    # Pull from env if not supplied
    api_key = api_key or os.getenv(OPENAI_API_KEY_ENV)
    base_url = base_url or os.getenv(OPENAI_BASE_URL_ENV)
    organization = organization or os.getenv(OPENAI_ORG_ENV)
    if timeout is None:
        env_timeout = _get_env_float(OPENAI_TIMEOUT_ENV)
        timeout = env_timeout if env_timeout is not None else 30.0

    kwargs: dict[str, Any] = {}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    if organization:
        kwargs["organization"] = organization
    if timeout is not None:
        kwargs["timeout"] = timeout
    return kwargs


def get_gemini_api_key() -> str | None:
    """Return the Gemini API key from env, preferring GOOGLE_API_KEY then GEMINI_API_KEY."""
    return os.getenv(GEMINI_API_KEY_ENV) or os.getenv(GEMINI_API_KEY_FALLBACK_ENV)


def get_gemini_timeout() -> float | None:
    """Return default Gemini timeout in seconds from env, if set."""
    return _get_env_float(GEMINI_TIMEOUT_ENV)
