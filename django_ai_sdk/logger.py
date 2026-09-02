from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from loguru import logger

from django_ai_sdk.utils import resolve_setting

if TYPE_CHECKING:
    from loguru import Logger

logger.remove()


if resolve_setting("AI_SDK_ENABLE_LOGS", False):
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="DEBUG",
        colorize=True,
    )

logger.add(
    "logs/django_ai_sdk.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="INFO",
    rotation="10 MB",
    retention="30 days",
    compression="zip",
    enqueue=True,
)


# Create module loggers
def get_logger(name: str) -> Logger:
    """
    Get a logger instance for a specific module.
    """
    return logger.bind(name=name)


__all__ = ["logger", "get_logger"]
