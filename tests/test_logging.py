import logging

from codeless_orchestrator.logging import get_logger


def test_get_logger_returns_logger_instance():
    logger = get_logger("test")
    assert isinstance(logger, logging.Logger)


def test_logger_emits_records_with_caplog(caplog):
    caplog.set_level(logging.INFO)
    logger = get_logger("codeless.test")
    logger.info("hello world")

    assert any(
        rec.name == "codeless.test" and rec.levelno == logging.INFO and "hello world" in rec.message
        for rec in caplog.records
    ), "Expected log record not found in captured logs"

