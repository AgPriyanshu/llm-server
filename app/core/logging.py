"""Structured logging configuration for production."""

import logging
import sys
from typing import Optional


def setup_logging(level: Optional[int] = None) -> logging.Logger:
    """
    Configure structured JSON logging for production.

    Args:
        level: Logging level. Defaults to INFO.

    Returns:
        Configured logger instance.
    """
    if level is None:
        level = logging.INFO

    # JSON-formatted log output for easy parsing by log aggregators
    log_format = (
        '{"time":"%(asctime)s",'
        '"level":"%(levelname)s",'
        '"logger":"%(name)s",'
        '"message":"%(message)s"}'
    )

    logging.basicConfig(
        level=level,
        format=log_format,
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    return logging.getLogger("llm_server")


# Create a default logger instance
logger = setup_logging()

