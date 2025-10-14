from __future__ import annotations

import os
from typing import Any

OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
OPENAI_ORG_ENV = "OPENAI_ORG"
OPENAI_BASE_URL_ENV = "OPENAI_BASE_URL"
OPENAI_TIMEOUT_ENV = "OPENAI_TIMEOUT"
# Backward/compat fallbacks for users who mistakenly set *_ENV names in .env
OPENAI_API_KEY_ENV_FALLBACK = "OPENAI_API_KEY_ENV"
OPENAI_ORG_ENV_FALLBACK = "OPENAI_ORG_ENV"
OPENAI_BASE_URL_ENV_FALLBACK = "OPENAI_BASE_URL_ENV"
OPENAI_TIMEOUT_ENV_FALLBACK = "OPENAI_TIMEOUT_ENV"

# Simple proxy support
PROXY_ENABLED_ENV = "PROXY_ENABLED"
PROXY_URL_ENV = "PROXY_URL"
PROXY_VERIFY_ENV = "PROXY_VERIFY"
PROXY_CA_BUNDLE_ENV = "PROXY_CA_BUNDLE"
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
    api_key = api_key or os.getenv(OPENAI_API_KEY_ENV) or os.getenv(OPENAI_API_KEY_ENV_FALLBACK)
    base_url = base_url or os.getenv(OPENAI_BASE_URL_ENV) or os.getenv(OPENAI_BASE_URL_ENV_FALLBACK)
    organization = organization or os.getenv(OPENAI_ORG_ENV) or os.getenv(OPENAI_ORG_ENV_FALLBACK)
    if timeout is None:
        env_timeout = _get_env_float(OPENAI_TIMEOUT_ENV)
        if env_timeout is None:
            env_timeout = _get_env_float(OPENAI_TIMEOUT_ENV_FALLBACK)
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


def _env_truthy(val: str | None) -> bool:
    if val is None:
        return False
    return val.strip().lower() in {"1", "true", "yes", "on"}


def is_proxy_enabled() -> bool:
    return _env_truthy(os.getenv(PROXY_ENABLED_ENV)) and bool(os.getenv(PROXY_URL_ENV))


def get_proxy_url() -> str | None:
    url = os.getenv(PROXY_URL_ENV)
    return url.strip() if url else None


def get_httpx_proxies() -> Any | None:
    """Return an httpx-compatible proxy config if enabled.

    For httpx>=0.28, use the single 'proxy' parameter with a URL string.
    """
    if not is_proxy_enabled():
        return None
    url = get_proxy_url()
    if not url:
        return None
    # httpx 0.28+ expects 'proxy' (single) rather than 'proxies'
    return url


def get_httpx_verify() -> Any | None:
    """Return a value for httpx 'verify' parameter if configured.

    - If PROXY_VERIFY is set to a falsey value, return False (disable TLS verification).
    - Else if PROXY_CA_BUNDLE is set, return that path so the proxy's CA is trusted.
    - Else return None (let httpx use its defaults).
    """
    verify_flag = os.getenv(PROXY_VERIFY_ENV)
    if verify_flag is not None and not _env_truthy(verify_flag):
        return False
    ca = os.getenv(PROXY_CA_BUNDLE_ENV)
    if ca:
        return ca
    return None


def apply_requests_proxy_env() -> None:
    """If proxy is enabled, ensure HTTP(S)_PROXY env vars are set for requests-based libs."""
    if not is_proxy_enabled():
        return
    url = get_proxy_url()
    if not url:
        return
    os.environ.setdefault("HTTP_PROXY", url)
    os.environ.setdefault("HTTPS_PROXY", url)
    # Propagate CA bundle to requests-based libraries if provided
    ca = os.getenv(PROXY_CA_BUNDLE_ENV)
    if ca:
        os.environ.setdefault("SSL_CERT_FILE", ca)
        os.environ.setdefault("REQUESTS_CA_BUNDLE", ca)
        os.environ.setdefault("CURL_CA_BUNDLE", ca)
