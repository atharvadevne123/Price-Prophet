"""Centralised logging configuration for Price-Prophet."""

from __future__ import annotations

import logging
import logging.config
import os
from typing import Any


def get_log_level() -> str:
    """Return log level from environment, defaulting to INFO."""
    return os.getenv("LOG_LEVEL", "INFO").upper()


LOGGING_CONFIG: dict[str, Any] = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s %(levelname)-8s %(name)s %(message)s",
            "datefmt": "%Y-%m-%dT%H:%M:%S",
        },
        "json": {
            "()": "logging.Formatter",
            "fmt": "time:%(asctime)s,level:%(levelname)s,logger:%(name)s,msg:%(message)s",
            "datefmt": "%Y-%m-%dT%H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "stream": "ext://sys.stdout",
        },
    },
    "root": {
        "level": get_log_level(),
        "handlers": ["console"],
    },
    "loggers": {
        "uvicorn": {"propagate": True},
        "uvicorn.access": {"propagate": True},
        "sqlalchemy.engine": {
            "level": "WARNING",
            "propagate": True,
        },
    },
}


def configure_logging() -> None:
    """Apply the logging configuration.

    Call once at application startup before any loggers are created.
    """
    LOGGING_CONFIG["root"]["level"] = get_log_level()
    logging.config.dictConfig(LOGGING_CONFIG)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, applying config if needed.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    configure_logging()
    return logging.getLogger(name)
