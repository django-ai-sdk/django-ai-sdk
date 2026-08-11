---
title: Permissions
type: docs
weight: 120
---

How the SDK gates operations: the `Operation` enum, permission domains, the built-in classes, and how to write your own. Every service (`AgentService`, `ThreadService`, `MemoryService`, `as_view()`, `history()`) checks permissions before acting and raises `PermissionDenied` on failure.

## Operations and Domains

An **operation** is the unit of access control. A **domain** is the default permission chain for a group of operations:

```python
from django_ai_sdk.permissions import Operation, PermissionDomain

Operation.CHAT            # "chat"
PermissionDomain.MEMORY   # "memory"
```

| Domain | Default class |
| --- | --- |
| `agent` | `AgentDefaultPermission` |
| `thread` | `ThreadDefaultPermission` |
| `memory` | `MemoryDefaultPermission` |
| `integrations` | `IntegrationDefaultPermission` |

Override a domain globally via `AI_SDK_PERMISSIONS`:

```python
AI_SDK_PERMISSIONS = {
    "memory": ["apps.memories.permissions.AllowAnonymousMemoryPermission"],
    "thread": ["apps.agents.permissions.DemoThreadPermission"],
}
```

## The Operation Enum

```python
# Agent
VIEW_AGENT, CREATE_AGENT, UPDATE_AGENT, DELETE_AGENT
# Threads
LIST_THREADS, CREATE_THREAD, VIEW_THREAD, UPDATE_THREAD, DELETE_THREAD,
DELETE_ALL_THREADS, VIEW_HISTORY
# Messages
VIEW_MESSAGES, SEND_MESSAGE, RATE_MESSAGE, DELETE_MESSAGE, RESTORE_MESSAGE
# Files
UPLOAD_FILE, DELETE_FILE, VIEW_FILE
# Memories
VIEW_MEMORY, CREATE_MEMORY, UPDATE_MEMORY, DELETE_MEMORY,
VIEW_DOCUMENT, LIST_DOCUMENTS, UPLOAD_DOCUMENT, DELETE_DOCUMENT,
LIST_THREAD_MEMORIES, LINK_MEMORY, UNLINK_MEMORY
# Integrations
USE_INTEGRATION, MANAGE_INTEGRATION
# Misc
CHAT, REINDEX
```

## Built-in Permission Classes

| Class | Behavior |
| --- | --- |
| `AllowAll` | Grants everything (the agent default). |
| `DenyAll` | Denies everything. |
| `IsAuthenticated` | Allows any authenticated user. |
| `IsAdminUser` | Allows staff or superusers. |
| `IsInAllowedGroups(groups=[...])` | Allows authenticated users in one of the named auth groups. |
| `IsOwner(field="user_id")` | Allows the object's owner; `field` selects the ownership attribute. |

### Tiered default permissions

Three built-ins expose `READ` / `WRITE` / `MANAGE` operation tiers:

- **`ThreadDefaultPermission`**: authenticated users get full access to their own threads; anonymous users get none.
- **`MemoryDefaultPermission`**: `can_manage` members get everything; `can_manage=False` members get read + write entries; public read-only is gated by `is_public`; anonymous is always blocked.
- **`AgentDefaultPermission`**: `can_manage` members manage the agent; other members view and use it; anonymous is blocked for agent operations and passes through everything else.

`MemoryService.get_object_permissions_map()` and `agent_permissions()` auto-discover these `READ`/`WRITE`/`MANAGE` tiers and return an `ObjectPermissions` (`can_read` / `can_write` / `can_manage`) for the frontend.

## The BasePermission API

```python
class BasePermission(ABC):
    async def has_permission(self, user, operation, **kwargs) -> bool: ...
    async def has_object_permission(self, user, operation, obj, **kwargs) -> bool: ...
    def get_queryset_perms(self, user, operation, queryset) -> QuerySet: ...
```

`has_permission` gates the operation itself; `has_object_permission` gates it on a specific object. Implementing `get_queryset_perms` lets list endpoints filter at the database level instead of post-checking every row: `MemoryDefaultPermission` uses it for `VIEW_MEMORY`.

## Writing a Custom Permission

```python
from django_ai_sdk.permissions import BasePermission, Operation

class IsSalesTeam(BasePermission):
    async def has_permission(self, user, operation, **kwargs) -> bool:
        if user is None or not user.is_authenticated:
            return False
        return await user.groups.filter(name="Sales").aexists()
```

Parameterized permissions are instances, not classes:

```python
class IsInAllowedGroups(BasePermission):
    def __init__(self, groups=None):
        self.groups = list(groups or [])

    async def has_permission(self, user, operation, **kwargs) -> bool:
        return user.is_authenticated and await user.groups.filter(name__in=self.groups).aexists()
```

Declare them on agents (`permissions = [IsInAllowedGroups(groups=["eng"])]`), on integrations (`service.permissions`), or as a domain override in `AI_SDK_PERMISSIONS`. Both class and instance forms are accepted anywhere.

## PermissionsMixin and Helpers

Services subclass `PermissionsMixin` and set a `domain`:

```python
class MyService(PermissionsMixin):
    domain = PermissionDomain.THREAD

    async def do_thing(self, *, user):
        await self.has_perms(user, Operation.VIEW_THREAD, obj, raise_on_deny=True)
```

- `has_perms(user, operation, obj=None)`: the check used by the built-in services; `raise_on_deny=False` returns a boolean instead of raising.
- `get_object_permissions_map(user, obj)`: returns `{"read": ..., "write": ..., "manage": ...}` from the tiered classes.
- `has_queryset_perms(user, operation, queryset)`: runs `get_queryset_perms` across the domain chain.

The low-level `check_permissions()` / `check_object_permissions()` / `has_perms()` free functions power all of the above and are what you'd use outside a mixin.
