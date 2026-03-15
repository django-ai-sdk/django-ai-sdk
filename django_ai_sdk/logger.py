import sys
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from loguru import Logger


logger.remove()

# Add console handler
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="DEBUG",
    colorize=True,
)

# Add file handler
# I added this also for code agents, they can track the app,
# without the need for a restart.
logger.add(
    "logs/django_ai_sdk.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="INFO",
    rotation="10 MB",
    retention="30 days",
    compression="zip",
)


# Create module loggers
def get_logger(name: str) -> "Logger":
    """
    Get a logger instance for a specific module.
    """
    return logger.bind(name=name)


__all__ = ["logger", "get_logger"]
