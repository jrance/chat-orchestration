from __future__ import annotations

import logging
import os
from typing import Final

_DEFAULT_FMT: Final[str] = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
_DEFAULT_DATEFMT: Final[str] = "%Y-%m-%d %H:%M:%S"
_ENV_LEVEL: Final[str] = "LOG_LEVEL"


_LEVELS: Final[dict[str, int]] = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET,
}


def _parse_level(level_name: str) -> int:
    return _LEVELS.get(level_name.upper(), logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger with consistent formatting.

    - Level is sourced from the ``LOG_LEVEL`` env var (default INFO).
    - Adds a StreamHandler with a concise formatter if no handlers are configured
      on this logger or its ancestors.
    """

    logger = logging.getLogger(name)
    level_name = os.getenv(_ENV_LEVEL, "INFO")
    level = _parse_level(level_name)
    logger.setLevel(level)

    # Only attach our own handler if nothing is handling log records already.
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(_DEFAULT_FMT, datefmt=_DEFAULT_DATEFMT))
        logger.addHandler(handler)

    return logger
