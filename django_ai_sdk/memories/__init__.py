# Memories app for knowledge base models

# Import signals to register them when Django starts
from django_ai_sdk.memories import signals  # noqa: F401
from django_ai_sdk.memories.services import MemoryService

__all__ = ["MemoryService"]
