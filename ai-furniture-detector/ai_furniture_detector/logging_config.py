"""Logging configuration for AI Furniture Detector."""

from __future__ import annotations

import logging
from typing import Optional

from .config import settings


def configure_logging(level: Optional[str] = None, format_string: Optional[str] = None) -> None:
    """
    Configure logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Defaults to settings.log_level.
        format_string: Log format string. Defaults to settings.log_format.
    """
    level = level or settings.log_level
    format_string = format_string or settings.log_format
    if format_string.lower() == "json":
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(
        level=level,
        format=format_string,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Suppress verbose library logs
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("gradio").setLevel(logging.WARNING)
    logging.getLogger("uvicorn").setLevel(logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Logger name (typically __name__).

    Returns:
        Logger instance.
    """
    return logging.getLogger(name)
