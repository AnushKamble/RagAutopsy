"""Structured logging configuration for RAG-Autopsy."""

import logging
import sys


def setup_logging(
    level: str = "INFO",
    json_format: bool = False,
    log_file: str | None = None,
) -> logging.Logger:
    """Configure the root logger for the application."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger = logging.getLogger("rag_autopsy")
    logger.setLevel(log_level)

    # Avoid adding duplicate handlers if already configured
    if logger.handlers:
        return logger

    formatter: logging.Formatter
    if json_format:
        formatter = logging.Formatter(
            '{"time": "%(asctime)s", "name": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}'
        )
    else:
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a child logger under the rag_autopsy namespace."""
    base_logger = logging.getLogger("rag_autopsy")
    if name:
        return base_logger.getChild(name)
    return base_logger
