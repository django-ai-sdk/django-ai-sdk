"""
Storage adapters for conversation persistence.

This module provides different storage backends for persisting conversations
and ChatMessages. Storage adapters provide callbacks to StreamWriter
for automatic storage when streaming completes.
"""

from .base import (
    BaseStorageAdapter,
    StorageAdapterRegistry,
    StorageType,
)
from .services import ThreadService

__all__ = [
    "BaseStorageAdapter",
    "StorageAdapterRegistry",
    "StorageType",
    "ThreadService",
]
