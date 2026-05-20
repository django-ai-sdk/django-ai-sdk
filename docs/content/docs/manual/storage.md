---
title: Storage Guide
type: docs
weight: 104
---

Complete guide to conversation persistence.

## Table of Contents

1. [What is Storage?](#what-is-storage)
2. [Storage Architecture](#storage-architecture)
3. [Available Adapters](#available-adapters)
4. [MemoryStorageAdapter](#memorystorageadapter)
5. [DbStorageAdapter](#dbstorageadapter)
6. [Universal Storage Format](#universal-storage-format)
7. [Storage Registry](#storage-registry)
8. [ID Consistency](#id-consistency)
9. [Examples](#examples)
10. [Best Practices](#best-practices)

---

## What is Storage?

**Storage** persists conversation history so assistants can maintain context across interactions.

### Key Features

- **Thread Management** - Create, list, update, delete conversations
- **Message Persistence** - Store complete ChatMessage objects
- **Universal Format** - Same JSON structure for all adapters
- **Message Rating** - Thumbs up/down feedback
- **Soft Delete** - Recover accidentally deleted messages
- **Cross-Storage Queries** - Find threads across all adapters

---

## Storage Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Storage Layer                            │
│                                                           │
│  ┌─────────────────┐    ┌─────────────────┐             │
│  │ MemoryStorage   │    │  DbStorage      │             │
│  │ (In-Memory)     │    │  (Django ORM)   │             │
│  └────────┬────────┘    └────────┬────────┘             │
│           │                      │                       │
│           └──────────┬───────────┘                       │
│                      │                                    │
│                      ▼                                    │
│           ┌─────────────────┐                           │
│           │  BaseStorage    │                           │
│           │  (Abstract)    │                           │
│           └────────┬────────┘                           │
│                    │                                      │
└────────────────────┼──────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│            Universal ChatMessage Format                  │
│                                                           │
│  {                                                        │
│    "id": "uuid",                                         │
│    "role": "assistant",                                  │
│    "content": "Hello!",                                  │
│    "model": "gpt-4o-mini",                               │
│    "finish_reason": "stop",                              │
│    "tool_calls": [],                                     │
│    "sources": [],                                       │
│    ...                                                   │
│  }                                                        │
└─────────────────────────────────────────────────────────┘
```

![Storage Architecture](/images/graphs/storage_architecture.png)

---

## Available Adapters

| Adapter | Persistence | Speed | Best For |
|---------|-------------|-------|----------|
| `MemoryStorageAdapter` | In-memory | Fastest | Testing, development |
| `DbStorageAdapter` | Django ORM | Slower | Production, long-term |

### Switching Storage

```python
# Development - fast, no database needed
storage_adapter = MemoryStorageAdapter

# Production - persistent, scalable
storage_adapter = DbStorageAdapter
```

**No code changes needed!** Both use the same ChatMessage format.

---

## MemoryStorageAdapter

In-memory storage using Pydantic models. Data persists for the process lifetime.

### Usage

```python
from django_ai_sdk.storage.memory import MemoryStorageAdapter, MemoryStore

# Create storage for thread
storage = MemoryStorageAdapter(thread_id="thread-uuid")

# Create thread first
MemoryStore.create_thread(
    thread_id="thread-uuid",
    title="My Conversation",
    assistant_id="my-assistant",
    model="gpt-4o-mini",
)

# Store message
from django_ai_sdk.common import ChatMessage
message_id = await storage.store_chat_message(
    ChatMessage(role="user", content="Hello!")
)

# Retrieve history
history = await storage.get_history()
# Returns: [ChatMessage, ChatMessage, ...]
```

### Class Methods (Global)

```python
# Thread management
await MemoryStorageAdapter.create_thread(
    title="New Thread",
    metadata={"assistant": "my-bot"}
)

thread = await MemoryStorageAdapter.get_thread(thread_id)
threads = await MemoryStorageAdapter.list_threads(user_id="user-123")
await MemoryStorageAdapter.update_thread(thread_id, title="Updated")
await MemoryStorageAdapter.delete_thread(thread_id)
```

### Instance Methods (Thread-Specific)

```python
storage = MemoryStorageAdapter(thread_id)

# Message operations
message_id = await storage.store_chat_message(chat_message)
history = await storage.get_history()
await storage.rate_message(message_id, rating=1)  # Positive rating
await storage.rate_message(message_id, rating=-1) # Negative rating
await storage.delete_message(message_id)  # Soft delete
await storage.restore_message(message_id) # Undo delete
```

---

## DbStorageAdapter

Django ORM storage using database models. Persistent across restarts.

### Database Schema

```sql
-- Thread model
CREATE TABLE threads (
    id UUID PRIMARY KEY,
    title VARCHAR(255),
    metadata JSONB,
    user_id VARCHAR(255),
    assistant_id VARCHAR(255),
    model VARCHAR(100),
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

-- Message model
CREATE TABLE messages (
    id UUID PRIMARY KEY,
    thread_id UUID REFERENCES threads(id),
    result JSONB,           -- ChatMessage as JSON
    rating INT,            -- 1 (good), -1 (bad), NULL (none)
    is_deleted BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP
);
```

### Usage

```python
from django_ai_sdk.storage.db import DbStorageAdapter

# Create storage for thread
storage = DbStorageAdapter(thread_id="thread-uuid")

# Create thread first
await DbStorageAdapter.create_thread(
    title="My Conversation",
    metadata={"assistant": "my-bot"}
)

# Store message (same API as MemoryStorage)
message_id = await storage.store_chat_message(chat_message)

# Retrieve history (same API)
history = await storage.get_history()
```

### Querying Directly

```python
from django_ai_sdk.conversation.models import Thread, Message

# Get thread
thread = await Thread.objects.aget(id=thread_id)

# Get messages
messages = await Message.objects.filter(
    thread=thread,
    is_deleted=False
).order_by("created_at").all()

# Convert to ChatMessage
for msg in messages:
    chat_message = msg.to_chat_message()
    print(chat_message.content)
```

---

## Universal Storage Format

All storage adapters use the same ChatMessage JSON format:

```python
{
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "role": "assistant",
    "content": "Hello! How can I help you today?",
    "reasoning": None,
    "tool_calls": [],
    "sources": [
        {
            "id": "doc-123",
            "content": "Document content...",
            "score": 0.95
        }
    ],
    "model": "gpt-4o-mini",
    "finish_reason": "stop",
    "processing_time_ms": 1250,
    "started_at": 1712345678.0,
    "completed_at": 1712345679.25,
    "adapter_type": "openai",
    "errors": [],
    "usage": {
        "prompt_tokens": 25,
        "completion_tokens": 15,
        "total_tokens": 40
    },
    "metadata": {}
}
```

### Benefits

1. **Switch Storage Easily**
   ```python
   # Dev
   storage_adapter = MemoryStorageAdapter
   
   # Prod
   storage_adapter = DbStorageAdapter
   ```

2. **Consistent API**
   ```python
   # Same code works with both
   history = await storage.get_history()
   ```

3. **Portable Data**
   ```python
   # Export from Memory, import to Database
   history = await memory_storage.get_history()
   for msg in history:
       await db_storage.store_chat_message(msg)
   ```

---

## Storage Registry

Automatic storage detection across all registered adapters.

### How It Works

```python
from django_ai_sdk.storage.base import StorageAdapterRegistry

# Register adapters (done automatically on import)
StorageAdapterRegistry.register(MemoryStorageAdapter, StorageType.MEMORY)
StorageAdapterRegistry.register(DbStorageAdapter, StorageType.DATABASE)

# Get all adapters (sorted by speed)
adapters = StorageAdapterRegistry.get_all_adapters()
# Returns: [MemoryStorageAdapter, DbStorageAdapter]
```

### Intensiveness Levels

| Level | Speed | Examples |
|-------|-------|----------|
| `MEMORY` | Fastest | MemoryStorageAdapter |
| `FILE` | Fast | FileStorageAdapter |
| `DATABASE` | Slower | DbStorageAdapter |
| `REST_API` | Slowest | ExternalAPIStorageAdapter |

### Cross-Storage Thread Lookup

```python
class Assistant:
    async def get_storage_adapter(self, thread_id):
        # 1. Try all registered adapters
        for adapter_class in StorageAdapterRegistry.get_all_adapters():
            thread = await adapter_class.get_thread(thread_id)
            if thread:
                # Found! Use this adapter
                return adapter_class(thread_id)
        
        # 2. Not found - use default
        return MemoryStorageAdapter(thread_id)
```

**Use Case:** Find a thread even if you don't know which storage it's in.

---

## ID Consistency

**Critical:** The same ID flows through the entire system.

### ID Flow Diagram

```
┌─────────────────────────────────────────────────────────┐
│ 1. User Request                                           │
│    "What is the pirate code?"                           │
└──────────────────────┬────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ 2. Adapter Layer (ID Generation)                        │
│    message_id = "550e8400-e29b-41d4-a716-446655440000"   │
│    ← GENERATED ONCE                                     │
└──────────────────────┬────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ 3. Streaming to Frontend                                  │
│    SSE: {"messageId": "550e8400-e29b-41d4-a716-..."}     │
│    Frontend tracks message                              │
└──────────────────────┬────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ 4. Storage Layer                                          │
│    Message(id="550e8400-e29b-41d4-a716-446655440000")    │
│    Same ID in database!                                   │
└──────────────────────┬────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ 5. API Endpoints                                          │
│    GET /threads/{tid}/messages/{msg_id}/rate             │
│    msg_id = "550e8400-e29b-41d4-a716-446655440000"       │
│    Same ID for rating/deletion!                           │
└─────────────────────────────────────────────────────────┘
```

![ID Consistency Flow](/images/graphs/id_consistency.png)

### Why This Matters

1. **Frontend Tracking**
   ```javascript
   // Frontend receives SSE
   const messageId = event.messageId;
   // Can track message throughout conversation
   ```

2. **API Operations**
   ```python
   # Rate a message
   POST /threads/abc/messages/550e8400-e29b-41d4-a716-446655440000/rate/
   ```

3. **History Consistency**
   ```python
   # Same ID in history
   for msg in history:
       print(msg.id)  # "550e8400-e29b-41d4-a716-446655440000"
   ```

4. **Message Operations**
   ```python
   # Delete, rate, reference - all use same ID
   await storage.rate_message(message_id, rating=1)
   await storage.delete_message(message_id)
   ```

---

## Examples

### Example 1: Basic Thread Management

```python
from django_ai_sdk.storage.memory import MemoryStorageAdapter, MemoryStore

# 1. Create thread
thread_id = await MemoryStorageAdapter.create_thread(
    title="Pirate Conversation",
    metadata={"assistant": "pirate", "model": "gpt-4o-mini"},
    user_id="user-123"
)

# 2. Get storage adapter
storage = MemoryStorageAdapter(thread_id)

# 3. Store messages
from django_ai_sdk.common import ChatMessage

user_msg = ChatMessage(role="user", content="Tell me a joke")
user_id = await storage.store_chat_message(user_msg)

assistant_msg = ChatMessage(role="assistant", content="Why did the pirate...")
assistant_id = await storage.store_chat_message(assistant_msg)

# 4. Retrieve history
history = await storage.get_history()
for msg in history:
    print(f"{msg.role}: {msg.content}")

# 5. Rate message
await storage.rate_message(assistant_id, rating=1)  # Positive rating

# 6. List all threads for user
threads = await MemoryStorageAdapter.list_threads(user_id="user-123")
for thread in threads:
    print(f"Thread: {thread.title}")
```

### Example 2: Message Rating System

```python
from django_ai_sdk.storage.db import DbStorageAdapter

class RatingService:
    async def rate_message(self, thread_id: str, message_id: str, rating: int):
        """Rate a message: 1 (good), -1 (bad), 0 (neutral)"""
        storage = DbStorageAdapter(thread_id)
        
        success = await storage.rate_message(message_id, rating)
        if not success:
            raise ValueError(f"Message {message_id} not found")
        
        return {"status": "rated", "rating": rating}
    
    async def get_message_rating(self, thread_id: str, message_id: str):
        """Get current rating for a message"""
        from django_ai_sdk.conversation.models import Message
        
        message = await Message.objects.aget(id=message_id, thread_id=thread_id)
        return {
            "rating": message.rating,
            "is_rated": message.rating is not None
        }
```

### Example 3: Conversation History

```python
from django_ai_sdk.storage.db import DbStorageAdapter

class ConversationService:
    async def get_conversation_history(
        self,
        thread_id: str,
        include_deleted: bool = False
    ) -> list[dict]:
        """Get conversation history for frontend"""
        storage = DbStorageAdapter(thread_id)
        
        history = await storage.get_history()
        
        # Convert to frontend format
        return [
            {
                "id": msg.id,
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.started_at,
                "rating": msg.rating,
                "model": msg.model,
            }
            for msg in history
        ]
    
    async def delete_conversation(self, thread_id: str):
        """Soft delete entire conversation"""
        await DbStorageAdapter.delete_thread(thread_id)
```

### Example 4: Storage-Agnostic Assistant

```python
from django_ai_sdk import Assistant
from django_ai_sdk.storage.memory import MemoryStorageAdapter
from django_ai_sdk.storage.db import DbStorageAdapter

class ConfigurableAssistant(Assistant):
    """Assistant that can use different storage backends."""
    
    # Set via class variable
    storage_adapter = MemoryStorageAdapter  # Default
    
    async def store_interaction(self, user_msg: str, assistant_msg: str):
        """Store conversation - works with any storage!"""
        storage = await self.get_storage_adapter(None)
        
        # Store user message
        await storage.store_chat_message(
            ChatMessage(role="user", content=user_msg)
        )
        
        # Store assistant response
        await storage.store_chat_message(
            ChatMessage(role="assistant", content=assistant_msg)
        )

# Usage in development
dev_assistant = type(
    "DevAssistant",
    (ConfigurableAssistant,),
    {"storage_adapter": MemoryStorageAdapter}
)()

# Usage in production
prod_assistant = type(
    "ProdAssistant",
    (ConfigurableAssistant,),
    {"storage_adapter": DbStorageAdapter}
)()
```

---

## Best Practices

### 1. Always Create Thread Before Storing

```python
# Good
await MemoryStorageAdapter.create_thread(thread_id, title="My Thread")
storage = MemoryStorageAdapter(thread_id)
await storage.store_chat_message(message)

# Bad - will raise error
storage = MemoryStorageAdapter(thread_id)
await storage.store_chat_message(message)  # Error: Thread not found
```

### 2. Use Storage for Message Rating

```python
# Good - storage handles rating
await storage.rate_message(message_id, rating=1)

# Bad - don't modify ChatMessage directly
chat_message.rating = 1  # Won't persist!
```

### 3. Handle UUID Format

```python
# Good - valid UUID format
thread_id = str(uuid.uuid4())

# Bad - invalid format (will fail in DbStorageAdapter)
thread_id = "thread-123"  # Only works in MemoryStorageAdapter
```

### 4. Clear Storage Cache When Needed

```python
# After document changes in RAG
assistant.rag_provider.clear_cache()

# After thread deletion
StorageAdapterRegistry.clear_cache()
```

### 5. Use Async Methods

```python
# Good
history = await storage.get_history()

# Bad - blocking operation
history = storage.get_history()  # TypeError!
```

### 6. Handle Storage Not Found

```python
# Good - handle None
storage = await assistant.get_storage_adapter(thread_id)
if storage is None:
    # New conversation
    pass

# Bad - assume always exists
storage = await assistant.get_storage_adapter(thread_id)
await storage.store_chat_message(msg)  # May crash if None
```

---

## Next Steps

- See [Architecture Guide](architecture/) for core concepts
- Check [RAG Guide](rag/) for knowledge retrieval
- Review [Adapters](adapters/) for backend integration
- Check [Testing](testing/) for test examples
