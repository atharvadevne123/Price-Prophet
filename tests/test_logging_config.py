"""Tests for the logging configuration module."""

from __future__ import annotations

import logging


def test_get_log_level_default():
    from app.logging_config import get_log_level

    level = get_log_level()
    assert level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def test_get_log_level_from_env(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "debug")
    from app.logging_config import get_log_level

    assert get_log_level() == "DEBUG"


def test_configure_logging_does_not_raise():
    from app.logging_config import configure_logging

    configure_logging()


def test_logging_config_version():
    from app.logging_config import LOGGING_CONFIG

    assert LOGGING_CONFIG["version"] == 1


def test_logging_config_has_standard_formatter():
    from app.logging_config import LOGGING_CONFIG

    assert "standard" in LOGGING_CONFIG["formatters"]


def test_logging_config_has_console_handler():
    from app.logging_config import LOGGING_CONFIG

    assert "console" in LOGGING_CONFIG["handlers"]


def test_get_logger_returns_logger():
    from app.logging_config import get_logger

    logger = get_logger("test.module")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test.module"


def test_json_formatter_fmt_is_string():
    from app.logging_config import LOGGING_CONFIG

    fmt = LOGGING_CONFIG["formatters"]["json"]["fmt"]
    assert isinstance(fmt, str)
    assert "level" in fmt
