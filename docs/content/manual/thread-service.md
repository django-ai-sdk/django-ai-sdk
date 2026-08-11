---
title: ThreadService
type: docs
weight: 110
---

The recommended entry point for thread operations in views: resolves the agent's storage, checks permissions, and works across all adapters.

All methods are async and take a `user` for permission checking.

## Thread Operations

```python
from django_ai_sdk.storage.services import ThreadService

thread_id = await ThreadService.create_thread(agent_id, title="", metadata=None, user=request.user)
thread = await ThreadService.get_thread(thread_id, user=request.user)   # -> ThreadInfo | None
threads = await ThreadService.threads(user=request.user, limit=100, offset=0)
await ThreadService.update_thread(thread_id, title="...", user=request.user)
await ThreadService.delete_thread(thread_id, user=request.user)
await ThreadService.delete_all_threads(user=request.user)
```

`create_thread()` auto-populates `model`, `agent_name`, `agent_class`, and `created_via` in the thread metadata (caller-provided values take precedence) and auto-generates thread titles when `agent.title_generation` is enabled.

## Message Operations

```python
await ThreadService.rate_message(thread_id, message_id, rating=1, user=request.user)
await ThreadService.delete_message(thread_id, message_id, user=request.user)      # soft delete
await ThreadService.restore_message(thread_id, message_id, user=request.user)    # undo delete
```

`rate_message` takes `rating=1 | -1 | None` (None unrates) and an optional `feedback`.

## Storage Access

```python
storage = await ThreadService.storage_for_thread(thread_id, user=request.user)
```

## Permissions

Each operation checks the user's permission via the agent's `permissions` classes and raises `PermissionDenied` when access is denied. See the [Views and Routing guide](/views-and-routing/).

Next: [Custom Storage Adapters](../storage-registry/), implementing your own storage.
