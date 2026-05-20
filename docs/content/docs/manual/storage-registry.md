---
title: Storage Adapter Registry
type: docs
weight: 106
---

Internal documentation for the storage adapter registry system. This is primarily for SDK contributors and advanced users implementing custom storage adapters.

## Overview

The `StorageAdapterRegistry` provides a global registry for storage adapters, enabling cross-storage thread lookup and storage-type-aware operations.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              StorageAdapterRegistry                     │
│  (Global Singleton)                                       │
│                                                           │
│  _adapters: Dict[Type, StorageType]                      │
│  ├─ MemoryStorageAdapter → MEMORY (1)                   │
│  ├─ DbStorageAdapter → DATABASE (3)                     │
│  └─ CustomStorageAdapter → FILE (2)                     │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│            Cross-Storage Operations                     │
│                                                           │
│  get_all_adapters() → Sorted by StorageType             │
│  register() → Add adapter with type                       │
└─────────────────────────────────────────────────────────┘
```

## Storage Types

The `StorageType` enum defines adapter performance characteristics:

| Type | Value | Speed | Examples |
|------|-------|-------|----------|
| `MEMORY` | 1 | Fastest | MemoryStorageAdapter |
| `FILE` | 2 | Fast | FileStorageAdapter (custom) |
| `DATABASE` | 3 | Slower | DbStorageAdapter |
| `REST_API` | 4 | Slowest | ExternalAPIStorageAdapter (custom) |

**Purpose:** Guide cross-storage lookups to check fastest adapters first.

## Registration

Adapters register themselves automatically via `__init_subclass__`:

```python
# In storage/base.py - automatic registration
class BaseStorageAdapter(ABC):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Auto-register if not the base class
        if cls.__name__ != "BaseStorageAdapter":
            StorageAdapterRegistry.register(cls, cls.storage_type)
```

Each adapter declares its type:

```python
class MemoryStorageAdapter(BaseStorageAdapter):
    storage_type = StorageType.MEMORY  # Defined on class
    # ... implementation

class DbStorageAdapter(BaseStorageAdapter):
    storage_type = StorageType.DATABASE
    # ... implementation
```

## Cross-Storage Thread Lookup

The registry enables finding threads regardless of which storage contains them:

```python
# In Assistant.get_storage_adapter()
async def get_storage_adapter(self, thread_id):
    if thread_id is None:
        return None
    
    # Try all registered adapters, fastest first
    for adapter_class in StorageAdapterRegistry.get_all_adapters():
        thread = await adapter_class.get_thread(thread_id)
        if thread:
            return adapter_class(thread_id)
    
    # Thread not found - use default
    if self.storage_adapter is not None:
        return self.storage_adapter(thread_id)
    
    return MemoryStorageAdapter(thread_id)
```

**Why this matters:**
- Threads can migrate between storage types (e.g., Memory → Database)
- Assistant doesn't need to know where thread lives
- Transparent storage switching

## ThreadService Integration

`ThreadService` uses the registry for cross-storage operations:

```python
class ThreadService:
    @staticmethod
    async def get_assistant(thread_id: str) -> Thread | None:
        """Get thread from any storage adapter."""
        for adapter_class in StorageAdapterRegistry.get_all_adapters():
            thread = await adapter_class.get_thread(thread_id)
            if thread:
                return thread
        return None
    
    @staticmethod
    async def delete_thread(thread_id: str) -> bool:
        """Delete thread from wherever it exists."""
        for adapter_class in StorageAdapterRegistry.get_all_adapters():
            thread = await adapter_class.get_thread(thread_id)
            if thread:
                await adapter_class.delete_thread(thread_id)
                return True
        return False
```

## Implementing Custom Storage

To create a custom storage adapter:

```python
from django_ai_sdk.storage.base import BaseStorageAdapter, StorageType
from django_ai_sdk.storage.schemas import ChatMessage, ThreadInfo

class RedisStorageAdapter(BaseStorageAdapter):
    """Redis-backed storage adapter."""
    
    storage_type = StorageType.DATABASE  # Choose appropriate type
    
    def __init__(self, thread_id: str):
        self.thread_id = thread_id
        self.redis = get_redis_client()
    
    @classmethod
    async def get_thread(cls, thread_id: str) -> ThreadInfo | None:
        """Check if thread exists in Redis."""
        data = await cls._redis.get(f"thread:{thread_id}")
        if data:
            return ThreadInfo.parse_raw(data)
        return None
    
    @classmethod
    async def create_thread(
        cls,
        title: str,
        assistant_id: str,
        model: str,
        metadata: dict = None,
        user_id: str = None,
    ) -> ThreadInfo:
        """Create new thread in Redis."""
        thread = ThreadInfo(
            id=str(uuid.uuid4()),
            title=title,
            assistant_id=assistant_id,
            model=model,
            metadata=metadata or {},
            user_id=user_id,
        )
        await cls._redis.set(
            f"thread:{thread.id}",
            thread.json(),
            ex=86400,  # 24 hour expiry
        )
        return thread
    
    async def store_chat_message(self, message: ChatMessage) -> str:
        """Store message in thread."""
        key = f"thread:{self.thread_id}:messages"
        await self.redis.lpush(key, message.json())
        return message.id
    
    async def get_messages(self) -> list[ChatMessage]:
        """Retrieve all messages."""
        key = f"thread:{self.thread_id}:messages"
        data = await self.redis.lrange(key, 0, -1)
        return [ChatMessage.parse_raw(d) for d in data]
    
    async def rate_message(self, message_id: str, rating: int) -> bool:
        """Rate a message."""
        # Implementation...
        pass
    
    async def delete_message(self, message_id: str) -> bool:
        """Soft delete message."""
        # Implementation...
        pass
    
    async def restore_message(self, message_id: str) -> bool:
        """Restore deleted message."""
        # Implementation...
        pass
```

**Registration is automatic** — the adapter registers itself when the class is defined.

## Registry API

### StorageAdapterRegistry

```python
from django_ai_sdk.storage.base import StorageAdapterRegistry, StorageType

# Register adapter (usually automatic)
StorageAdapterRegistry.register(MyAdapter, StorageType.DATABASE)

# Get all adapters sorted by speed
adapters = StorageAdapterRegistry.get_all_adapters()
# Returns: [MemoryStorageAdapter, DbStorageAdapter, ...]

# Check if adapter is registered
is_registered = MyAdapter in StorageAdapterRegistry

# Get adapter count
count = len(StorageAdapterRegistry)
```

### StorageType

```python
from django_ai_sdk.storage.base import StorageType

# Compare performance levels
assert StorageType.MEMORY < StorageType.DATABASE
assert StorageType.DATABASE < StorageType.REST_API

# Use in conditionals
if adapter.storage_type == StorageType.MEMORY:
    # Fast operations
    pass
elif adapter.storage_type == StorageType.DATABASE:
    # Persistent operations
    pass
```

## Design Decisions

### Why Enum Values (1, 2, 3, 4)?

Enum values encode performance ordering. Sorting adapters by type naturally yields fastest-first ordering for lookups.

### Why Global Singleton?

- Adapters must be discoverable before `Assistant` instantiates
- Cross-storage queries need centralized adapter list
- Consistent with `AssistantRegistry` pattern

### Why Class Methods for Thread Operations?

Thread lookup (`get_thread`) must work without adapter instance (don't know which adapter has the thread yet). Instance methods (`store_chat_message`) require bound adapter for thread-specific operations.

## Testing

### Mocking Storage

```python
import pytest

@pytest.fixture
def mock_registry():
    """Provide isolated registry for tests."""
    from django_ai_sdk.storage.base import StorageAdapterRegistry
    
    # Save original state
    original = StorageAdapterRegistry._adapters.copy()
    
    # Clear for test
    StorageAdapterRegistry._adapters.clear()
    
    yield StorageAdapterRegistry
    
    # Restore original
    StorageAdapterRegistry._adapters = original

@pytest.mark.asyncio
async def test_cross_storage_lookup(mock_registry):
    """Test finding thread across adapters."""
    # Register mock adapters
    mock_registry.register(MockMemoryAdapter, StorageType.MEMORY)
    mock_registry.register(MockDbAdapter, StorageType.DATABASE)
    
    # Test lookup finds thread in second adapter
    MockDbAdapter._threads["test-123"] = ThreadInfo(id="test-123", ...)
    
    for adapter_class in mock_registry.get_all_adapters():
        thread = await adapter_class.get_thread("test-123")
        if thread:
            assert adapter_class == MockDbAdapter
            break
```

## Edge Cases

### Multiple Adapters Same Type

If multiple adapters share a type, order is undefined but deterministic (Python dict ordering):

```python
StorageAdapterRegistry.register(AdapterA, StorageType.MEMORY)
StorageAdapterRegistry.register(AdapterB, StorageType.MEMORY)

# Both are MEMORY type, order is registration order
```

### Adapter Unregistration

Not officially supported. Adapters are intended to be registered for application lifetime. If needed:

```python
# Not recommended, but possible
del StorageAdapterRegistry._adapters[MyAdapter]
```

### Thread in Multiple Storages

Theoretically possible if thread was copied. Registry returns first match (fastest adapter wins):

```python
# If thread exists in both Memory and DB:
for adapter_class in StorageAdapterRegistry.get_all_adapters():
    # MemoryStorageAdapter checked first (faster type)
    # Returns MemoryStorageAdapter(thread_id)
```

## Future Enhancements

Potential improvements:

1. **Storage migration utilities** — Move threads between adapters
2. **Replication support** — Sync threads across multiple adapters
3. **Performance metrics** — Track lookup times per adapter
4. **Async registration** — Support async adapter initialization

## Related Documentation

- [Storage Guide](storage/) — User-facing storage documentation
- [Architecture Guide](architecture/) — Component interactions
- [Testing Guide](testing/) — Test patterns
