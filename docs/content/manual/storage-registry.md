---
title: Custom Storage Adapters
type: docs
weight: 111
---

Reference for implementing custom storage adapters.

Every storage adapter subclasses `BaseStorageAdapter` and registers itself in `StorageAdapterRegistry`. The registry powers cross-storage thread lookup, letting `Agent.get_storage_adapter()` and `ThreadService` operate across backends without knowing which one holds a thread.

## StorageType

Adapters declare their operation cost so registry lookups check the fastest backends first:

| Level | Cost | Example |
| --- | --- | --- |
| `MEMORY` | 1 | `MemoryStorageAdapter` |
| `FILE` | 2 | custom file-backed adapter |
| `DATABASE` | 3 | `DbStorageAdapter` |
| `REST_API` | 4 | custom external-service adapter |

## Implementing a Custom Adapter

Subclass `BaseStorageAdapter`, implement every abstract method, and register it. There is no automatic `__init_subclass__` registration: adapters register explicitly at module bottom.

```python
from django_ai_sdk.common import ChatMessage
from django_ai_sdk.storage.base import BaseStorageAdapter, StorageAdapterRegistry, StorageType
from django_ai_sdk.storage.schemas import ThreadInfo


class RedisStorageAdapter(BaseStorageAdapter):
    """Redis-backed storage adapter."""

    def __init__(self, thread_id: str) -> None:
        super().__init__(thread_id)
        self.redis = get_redis_client()

    # --- Class methods: thread management (no instance needed) ---

    @classmethod
    async def create_thread(cls, title, metadata=None, user=None, thread_id=None) -> str:
        ...

    @classmethod
    async def get_thread(cls, thread_id: str) -> ThreadInfo | None:
        ...

    @classmethod
    async def list_threads(cls, user=None, *, limit=None, offset=0) -> list[ThreadInfo]:
        ...

    @classmethod
    async def update_thread(cls, thread_id, title=None, metadata=None) -> bool:
        ...

    @classmethod
    async def delete_thread(cls, thread_id: str) -> bool:
        ...

    # --- Instance methods: thread-specific operations ---

    async def get_messages(self) -> list[ChatMessage]:
        ...

    async def store_chat_message(self, chat_message: ChatMessage) -> str:
        ...

    async def storage_callback(self, chat_message: ChatMessage) -> str | None:
        """Called by StreamWriter.finalize() to persist streamed replies."""
        return await self.store_chat_message(chat_message)

    async def rate_message(self, message_id, rating, feedback="", user=None) -> bool:
        ...

    async def delete_message(self, message_id: str) -> bool:
        ...

    async def restore_message(self, message_id: str) -> bool:
        ...


StorageAdapterRegistry.register(RedisStorageAdapter, StorageType.DATABASE)
```

### Abstract method signatures

| Method | Kind | Signature |
| --- | --- | --- |
| `create_thread` | class | `(title, metadata=None, user=None, thread_id=None) -> str` |
| `get_thread` | class | `(thread_id) -> ThreadInfo \| None` |
| `list_threads` | class | `(user=None, *, limit=None, offset=0) -> list[ThreadInfo]` |
| `update_thread` | class | `(thread_id, title=None, metadata=None) -> bool` |
| `delete_thread` | class | `(thread_id) -> bool` |
| `get_messages` | instance | `() -> list[ChatMessage]` |
| `store_chat_message` | instance | `(chat_message) -> str` |
| `storage_callback` | instance | `(chat_message) -> str \| None` |
| `rate_message` | instance | `(message_id, rating, feedback="", user=None) -> bool` |
| `delete_message` | instance | `(message_id) -> bool` |
| `restore_message` | instance | `(message_id) -> bool` |

## Design Notes

- **Class methods for threads**: thread lookup (`get_thread`) must work before you know which adapter holds the thread; no instance exists yet. Instance methods operate on a bound thread.
- **Enum for cost**: sorting by `StorageType` yields fastest-first lookup ordering without extra metadata.
- **Edge cases**: multiple adapters of the same type follow registration order; a thread in multiple storages resolves to the fastest adapter; `clear()` exists for tests.

## Testing a Custom Adapter

All SDK adapters are tested against the same behavior. A useful test suite:

```python
import pytest
import uuid
from django_ai_sdk.common import ChatMessage

@pytest.mark.asyncio
async def test_custom_adapter_roundtrip():
    thread_id = await RedisStorageAdapter.create_thread(title="Test")
    storage = RedisStorageAdapter(thread_id)

    msg_id = await storage.store_chat_message(ChatMessage(role="user", content="Hello"))
    history = await storage.get_messages()

    assert len(history) == 1
    assert history[0].content == "Hello"
    assert history[0].id == msg_id

    assert await storage.rate_message(msg_id, rating=1) is True
    assert await storage.delete_message(msg_id) is True
    assert await storage.restore_message(msg_id) is True
```

See [Testing](../testing/) for the full test setup.
