---
title: Permissions
sidebar:
  order: 120
---

How the SDK gates operations: the `Operation` enum, permission domains, the built-in classes, and how to write your own. Every service entry point (`AgentService`, `ThreadService`, `MemoryService`, `WorkflowService`, `as_view()`, `history()`) checks permissions before it acts and raises `PermissionDenied` when the check fails.

## Operations and domains

An operation is the unit of access control. A domain is the default permission chain for a group of operations:

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
| `workflow` | `WorkflowDefaultPermission` |
| `oversight` | `django_ai_sdk.oversight.permissions.OversightDefaultPermission` |

Override a domain globally with `AI_SDK_PERMISSIONS`:

```python
AI_SDK_PERMISSIONS = {
    "memory": ["apps.memories.permissions.AllowAnonymousMemoryPermission"],
    "thread": ["apps.agents.permissions.DemoThreadPermission"],
}
```

## The Operation enum

:::details[Show every operation]
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
# Workflows
VIEW_WORKFLOW, RUN_WORKFLOW, MANAGE_WORKFLOW, APPROVE_WORKFLOW
# Oversight
VIEW_OVERSIGHT_METRICS, VIEW_OVERSIGHT_CONTENT, REVIEW_OVERSIGHT
# Misc
CHAT, REINDEX
```
:::

## Built-in permission classes

| Class | Behavior |
| --- | --- |
| `AllowAll` | Grants everything. |
| `DenyAll` | Denies everything. |
| `IsAuthenticated` | Allows any authenticated user. |
| `IsAdminUser` | Allows staff or superusers. |
| `IsInAllowedGroups(groups=[...])` | Allows authenticated users in one of the named auth groups. |
| `IsOwner(field="user_id")` | Allows the object's owner. `field` selects the ownership attribute. |

### Tiered default permissions

These are the domains' default classes. The first four expose `READ`, `WRITE` and `MANAGE` operation tiers:

- `ThreadDefaultPermission` gives authenticated users full access to their own threads. Anonymous users get none.
- `MemoryDefaultPermission` gives `can_manage` members everything. `can_manage=False` members can read and write entries. `is_public` grants public read-only access. Anonymous users are always blocked.
- `AgentDefaultPermission` lets `can_manage` members manage the agent. Other members view and use it. `is_public` lets anyone read and use it. Otherwise, anonymous users are blocked for agent operations and pass through everything else. Chat, thread and file operations have no object to check, so the class checks them against the agent passed as the `agent` keyword.
- `WorkflowDefaultPermission` lets any authenticated user run an active workflow. Creating, editing and deleting stored definitions needs `change_workflow`. Whoever last changed a definition's steps becomes its `created_by`, the account its triggered runs act as. Only the principal a run ran as can read it, because its outputs are that person's content. Staff manage and read every definition and run. The class implements `get_queryset_perms`, so list endpoints filter in the database, not afterwards. See [Workflows](workflows.md#who-may-do-what).
- `OversightDefaultPermission` (`django_ai_sdk.oversight.permissions`) requires one Django model permission per operation and never checks `is_staff`. The `scope_turns` and `scope_reviews` hooks narrow it. [Oversight](oversight.md#review-api) lists the operations and their permissions.

`MemoryService.get_object_permissions_map()` and `agent_permissions()` discover these `READ`, `WRITE` and `MANAGE` tiers and return an `ObjectPermissions` (`can_read`, `can_write`, `can_manage`) for the frontend.

## The BasePermission API

```python
class BasePermission(ABC):
    async def has_permission(self, user, operation, **kwargs) -> bool: ...
    async def has_object_permission(self, user, operation, obj, **kwargs) -> bool: ...
    def get_queryset_perms(self, user, operation, queryset) -> QuerySet: ...
```

`has_permission` checks the operation itself. `has_object_permission` checks it on a specific object. With `get_queryset_perms`, list endpoints filter in the database instead of checking every row afterwards. `MemoryDefaultPermission` uses it for `VIEW_MEMORY`.

## Writing a custom permission

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

Declare them on agents (`permissions = [IsInAllowedGroups(groups=["eng"])]`), on integrations (`service.permissions`), or as a domain override in `AI_SDK_PERMISSIONS`. Every place accepts both the class and the instance form.

## PermissionsMixin and helpers

Services subclass `PermissionsMixin` and set a `domain`:

```python
class MyService(PermissionsMixin):
    domain = PermissionDomain.THREAD

    async def do_thing(self, *, user):
        await self.has_perms(user, Operation.VIEW_THREAD, obj, raise_on_deny=True)
```

- `has_perms(user, operation, obj=None)` is the check the built-in services use. With `raise_on_deny=False` it returns a boolean instead of raising.
- `get_object_permissions_map(user, obj)` returns `{"read": ..., "write": ..., "manage": ...}` from the tiered classes.
- `has_queryset_perms(user, operation, queryset)` runs `get_queryset_perms` across the domain chain.

The low-level free functions `check_permissions()`, `check_object_permissions()` and `has_perms()` power all of the above. Use them outside a mixin.
