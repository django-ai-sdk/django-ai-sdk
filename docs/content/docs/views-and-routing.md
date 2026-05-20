---
title: Views and Routing
type: docs
prev: tools-and-agents
next: protocols-and-adapters
weight: 4
---

This page covers how to wire your assistants into Django views using the Assistant Registry, and how to handle conversation threads.

## Basic View

The simplest setup -- retrieve assistant from registry by ID:

```python
from ninja import Router
from django_ai_sdk.assistants.registry import registry

router = Router()

@router.post("/chat/{assistant_id}")
async def chat(request, assistant_id: str, payload: ChatRequest):
    assistant = registry.get(assistant_id)
    if not assistant:
        return {"error": "Assistant not found"}, 404
    
    return await assistant.as_view(payload.messages)
```

`as_view()` returns a `StreamingHttpResponse` with SSE headers already set. You return it directly from your view.

## Request Schema

You'll need a schema for the incoming messages. The Vercel AI SDK sends messages with `parts` (not plain `content`), so your schema should match:

```python
from typing import List, Optional
from ninja import Schema

class MessagePart(Schema):
    type: str
    text: Optional[str] = None
    toolCallId: Optional[str] = None
    state: Optional[str] = None
    input: Optional[dict] = None
    output: Optional[dict] = None

    class Config:
        extra = "allow"  # Allow additional fields for tool parts, etc.

class Message(Schema):
    role: str
    parts: List[MessagePart]
    id: Optional[str] = None

class ChatRequest(Schema):
    messages: List[Message]
    id: Optional[str] = None
    assistant_id: Optional[str] = None
```

The protocol handler takes care of converting these into internal `ChatMessage` objects -- you don't need to do that yourself.

## Using the Registry

The Assistant Registry provides centralized access to all registered assistants:

```python
from django_ai_sdk.assistants.registry import registry

@router.post("/chat/{assistant_id}")
async def chat_with_assistant(request, assistant_id: str, payload: ChatRequest):
    assistant = registry.get(assistant_id)
    if not assistant:
        return {"error": "Assistant not found"}, 404

    return await assistant.as_view(payload.messages, thread_id=payload.id)
```

The registry is automatically populated when Django starts based on your `AI_SDK_ASSISTANTS` setting or `@auto_register` decorators.

## ThreadService

The SDK provides `ThreadService` for convenient thread management across all storage adapters:

```python
from django_ai_sdk.storage.services import ThreadService

# Create a thread
thread = await ThreadService.create_thread(
    title="Support Chat",
    assistant_id=assistant_id,
    model="gpt-4",
    metadata={"source": "web", "priority": "high"},
    user_id=request.user.id if request.user.is_authenticated else None,
)

# Get thread with assistant info
thread_info = await ThreadService.get_assistant(thread_id)

# List all threads
threads = await ThreadService.threads(
    assistant_id=None,  # Optional filter
    user_id=request.user.id,  # Optional filter
    search=None,  # Optional title search
)

# Update thread
await ThreadService.update_thread(
    thread_id,
    title="Updated Title",
    metadata={"status": "resolved"},
)

# Delete thread (works across all storage types)
await ThreadService.delete_thread(thread_id)
```

**Why use ThreadService?**

- Works with any storage adapter (Memory, Database, or custom)
- Handles cross-storage queries automatically
- Consistent API regardless of storage backend
- Automatic metadata management

## Listing Available Assistants

The `info()` method on each assistant returns metadata you can expose to the frontend:

```python
from django_ai_sdk.assistants.registry import registry

@router.get("/assistants")
def list_assistants(request):
    assistants_info = []
    for assistant_id, assistant in registry.all().items():
        info = assistant.info()
        info["id"] = assistant_id
        assistants_info.append(info)
    return {"assistants": assistants_info}
```

This returns name, model, description, available tools, etc. -- useful for building an assistant selector in your UI.

## Conversation Threads

For persistent conversations, pass a `thread_id` to `as_view()`. The SDK will automatically store messages in the database using the `Thread` and `Message` models.

### Creating a Thread

```python
from django_ai_sdk.conversation.models import Thread
from django_ai_sdk.assistants.registry import registry

@router.post("/threads")
async def create_thread(request, payload: ChatRequest):
    assistant_id = payload.assistant_id
    assistant = registry.get(assistant_id)
    if not assistant:
        return {"error": "Assistant not found"}, 404

    thread = await Thread.objects.acreate(
        title="New Conversation",
        assistant_id=assistant_id,
        model=assistant.model,
    )
    return {"thread_id": str(thread.id)}
```

### Sending Messages to a Thread

```python
from django_ai_sdk.conversation.models import Thread
from django_ai_sdk.assistants.registry import registry

@router.post("/threads/{thread_id}/")
async def add_message_to_thread(request, thread_id: str, payload: ChatRequest):
    try:
        thread = await Thread.objects.aget(id=thread_id)
    except Thread.DoesNotExist:
        return {"error": "Thread not found"}, 404

    assistant = registry.get(thread.assistant_id)
    if not assistant:
        return {"error": "Assistant not found"}, 404
    
    return await assistant.as_view(payload.messages, thread_id=thread_id)
```

When `thread_id` is provided, `as_view()` automatically:
1. Creates a `DbStorageAdapter` for the thread
2. Stores the incoming user message
3. Passes the storage adapter to the pipeline adapter so the assistant's response gets stored too

### Retrieving Thread History

```python
from django_ai_sdk.storage import ThreadDetail
from django_ai_sdk.assistants.registry import registry

@router.get("/threads/{thread_id}/", response={200: ThreadDetail, 404: Error})
async def get_thread_history(request, thread_id: str):
    """Get conversation history for a thread."""
    from django_ai_sdk.storage import ThreadService
    
    # Find the thread and assistant
    thread = await ThreadService.get_assistant(thread_id)
    if not thread:
        return {"error": "Thread not found"}, 404
    
    assistant = registry.get(thread.assistant_id)
    if not assistant:
        return {"error": "Assistant not found"}, 404
    
    # Returns ThreadDetail with thread metadata and protocol-formatted messages
    return await assistant.history(thread_id)
```

### Listing and Deleting Threads

```python
@router.get("/threads")
async def list_threads(request):
    from django_ai_sdk.conversation.models import Thread
    threads = Thread.objects.with_messages().order_by("-updated_at")
    threads_list = []
    async for thread in threads:
        threads_list.append({
            "id": str(thread.id),
            "title": thread.title,
            "assistant_id": thread.assistant_id,
            "created_at": thread.created_at.isoformat(),
            "updated_at": thread.updated_at.isoformat(),
        })
    return {"threads": threads_list}

@router.delete("/threads/{thread_id}")
async def delete_thread(request, thread_id: str):
    try:
        thread = await Thread.objects.aget(id=thread_id)
        await thread.adelete()
        return {"success": True}
    except Thread.DoesNotExist:
        return {"error": "Thread not found"}, 404
```

## Message Management

Threads support message rating, deletion, and restoration via the storage adapter:

### Rate a Message

```python
from django_ai_sdk.storage.services import ThreadService

@router.post("/threads/{thread_id}/messages/{message_id}/rate")
async def rate_message(
    request,
    thread_id: str,
    message_id: str,
    rating: int,  # 1 = positive, -1 = negative, 0 = neutral
):
    """Rate a message (thumbs up/down)."""
    storage = await ThreadService.get_assistant(thread_id)
    if not storage:
        return {"error": "Thread not found"}, 404
    
    success = await storage.rate_message(message_id, rating)
    if not success:
        return {"error": "Message not found"}, 404
    
    return {"status": "rated", "rating": rating}
```

### Delete (Soft Delete) a Message

```python
@router.post("/threads/{thread_id}/messages/{message_id}/delete")
async def delete_message(request, thread_id: str, message_id: str):
    """Soft delete a message (can be restored later)."""
    storage = await ThreadService.get_assistant(thread_id)
    if not storage:
        return {"error": "Thread not found"}, 404
    
    success = await storage.delete_message(message_id)
    if not success:
        return {"error": "Message not found"}, 404
    
    return {"status": "deleted"}
```

### Restore a Deleted Message

```python
@router.post("/threads/{thread_id}/messages/{message_id}/restore")
async def restore_message(request, thread_id: str, message_id: str):
    """Restore a previously deleted message."""
    storage = await ThreadService.get_assistant(thread_id)
    if not storage:
        return {"error": "Thread not found"}, 404
    
    success = await storage.restore_message(message_id)
    if not success:
        return {"error": "Message not found"}, 404
    
    return {"status": "restored"}
```

## Assistant Management

### Reindex RAG Pipeline

When documents change, you can trigger a RAG reindex:

```python
from django_ai_sdk.assistants.registry import registry

@router.post("/assistants/{assistant_id}/reindex")
async def reindex_assistant(request, assistant_id: str, silo_id: str = None):
    """Reindex the assistant's RAG pipeline."""
    assistant = registry.get(assistant_id)
    if not assistant:
        return {"error": "Assistant not found"}, 404
    
    # Trigger reindex (clears cache and rebuilds index)
    result = await assistant.reindex(silo_id)
    
    return {
        "status": "reindexed",
        "assistant_id": assistant_id,
        "silo_id": silo_id,
    }
```

### Health Check

Simple endpoint to verify the SDK is working:

```python
@router.get("/health")
async def health_check(request):
    """Health check endpoint."""
    from django_ai_sdk.assistants.registry import registry
    
    return {
        "status": "ok",
        "assistants": len(registry.all()),
    }
```

## Complete Example

Here's a complete router with all common endpoints:

```python
from ninja import Router
from django_ai_sdk.assistants.registry import registry
from django_ai_sdk.storage.services import ThreadService

router = Router()

# Assistants
@router.get("/assistants")
async def list_assistants(request):
    assistants = []
    for assistant_id, assistant in registry.all().items():
        info = assistant.info()
        info["id"] = assistant_id
        assistants.append(info)
    return {"assistants": assistants}

# Threads
@router.post("/threads")
async def create_thread(request, assistant_id: str, title: str = "New Chat"):
    assistant = registry.get(assistant_id)
    if not assistant:
        return {"error": "Assistant not found"}, 404
    
    thread = await ThreadService.create_thread(
        title=title,
        assistant_id=assistant_id,
        model=assistant.model,
        user_id=getattr(request.user, "id", None),
    )
    return {"thread_id": str(thread.id)}

@router.get("/threads")
async def list_threads(request):
    threads = await ThreadService.threads(
        user_id=getattr(request.user, "id", None),
    )
    return {"threads": [
        {
            "id": str(t.id),
            "title": t.title,
            "assistant_id": t.assistant_id,
            "created_at": t.created_at.isoformat(),
        }
        for t in threads
    ]}

@router.get("/threads/{thread_id}")
async def get_thread(request, thread_id: str):
    thread = await ThreadService.get_assistant(thread_id)
    if not thread:
        return {"error": "Thread not found"}, 404
    
    assistant = registry.get(thread.assistant_id)
    if not assistant:
        return {"error": "Assistant not found"}, 404
    
    return await assistant.history(thread_id)

@router.delete("/threads/{thread_id}")
async def delete_thread(request, thread_id: str):
    await ThreadService.delete_thread(thread_id)
    return {"status": "deleted"}

# Chat
@router.post("/chat/{assistant_id}")
async def chat(request, assistant_id: str, payload: ChatRequest):
    assistant = registry.get(assistant_id)
    if not assistant:
        return {"error": "Assistant not found"}, 404
    
    return await assistant.as_view(payload.messages, thread_id=payload.id)

@router.post("/threads/{thread_id}/chat")
async def chat_in_thread(request, thread_id: str, payload: ChatRequest):
    thread = await ThreadService.get_assistant(thread_id)
    if not thread:
        return {"error": "Thread not found"}, 404
    
    assistant = registry.get(thread.assistant_id)
    if not assistant:
        return {"error": "Assistant not found"}, 404
    
    return await assistant.as_view(payload.messages, thread_id=thread_id)

# Message management
@router.post("/threads/{thread_id}/messages/{message_id}/rate")
async def rate_message(request, thread_id: str, message_id: str, rating: int):
    storage = await ThreadService.get_assistant(thread_id)
    if not storage:
        return {"error": "Thread not found"}, 404
    
    success = await storage.rate_message(message_id, rating)
    if not success:
        return {"error": "Message not found"}, 404
    
    return {"status": "rated", "rating": rating}
```
