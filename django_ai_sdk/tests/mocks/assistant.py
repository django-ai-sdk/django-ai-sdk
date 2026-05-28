"""
Mock assistant factory.

Assistants have side effects (LLM calls, registry registration, storage I/O),
so we use controlled MagicMock instances in unit tests instead of real subclasses.
"""

from unittest.mock import AsyncMock, MagicMock

from django_ai_sdk.permissions import AllowAll
from django_ai_sdk.storage.memory import MemoryStorageAdapter


def create_assistant_mock(
    assistant_id: str = "test-assistant",
    name: str = "Test Assistant",
    model: str = "gpt-4",
    permissions: list | None = None,
    storage_adapter=MemoryStorageAdapter,
    **attrs,
) -> MagicMock:
    """Create a controlled MagicMock representing a registered assistant.

    Defaults to AllowAll permissions and MemoryStorageAdapter.
    Pass extra keyword arguments to override or add attributes.
    """
    if permissions is None:
        permissions = [AllowAll]

    assistant = MagicMock()
    assistant.id = assistant_id
    assistant.name = name
    assistant.model = model
    assistant.storage_adapter = storage_adapter
    assistant.permissions = permissions
    assistant.history = AsyncMock(
        return_value=MagicMock(
            thread={"id": "thread-1", "title": "Test"}, messages=[]
        )
    )
    for k, v in attrs.items():
        setattr(assistant, k, v)
    return assistant


def create_mock_adapter_class(get_thread=None):
    """Create a mock storage adapter *class* (not instance).

    The returned mock looks like a storage adapter class with
    an async ``get_thread`` classmethod.
    """
    adapter_cls = MagicMock()
    adapter_cls.__name__ = "MockAdapter"
    adapter_cls.get_thread = AsyncMock(return_value=get_thread)
    return adapter_cls
