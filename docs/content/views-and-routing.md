---
title: Views and Routing
type: docs
weight: 5
---

This page covers wiring agents into Django views: the chat endpoint, thread management, runtime-configured agents, permissions, and workflows. The demo (`demo/apps/agents/views/`) contains complete Ninja and experimental DRF routers. This guide walks through the same API.

## The Chat Endpoint

Agents expose `as_view()`, which returns a ready-to-return `StreamingHttpResponse`. The minimal endpoint:

```python
from ninja import Router
from django_ai_sdk.agents.services import AgentService
from django_ai_sdk.views.schemas import ChatRequest

router = Router()

@router.post("/chat")
async def chat(request, payload: ChatRequest):
    agent = await AgentService.get(payload.agent_id)
    return await agent.as_view(payload.messages, user=request.user)
```

- **`payload.messages`**: Vercel protocol messages. `as_view()` converts them via the agent's protocol handler, applies `max_history`, stores the last user message, builds the pipeline adapter, and streams the response.
- **`user`**: passed to the adapter and used for permission checks and conversation attribution.
- **`thread_id`**: pass it to persist the conversation (see below).

`ChatRequest` comes from `django_ai_sdk.views.schemas`:

```python
class ChatRequest(BaseModel):
    messages: list[Message]
    agent_id: str | None = None
    id: str | None = None
    trigger: str | None = None
```

---

## AgentService

`AgentService` is the single entry point for resolving agents: it checks the registry first, then falls back to DB-configured runtime agents.

```python
from django_ai_sdk.agents.services import AgentService

# By stable ID (registry or AgentSettings)
agent = await AgentService.get(agent_id)

# The agent attached to a thread
agent = await AgentService.get_agent(thread_id, user=request.user)

# Agents the user may see (registry + DB-backed)
items = await AgentService.list_agents(user=request.user)

# Metadata for an agent listing
info = await AgentService.get_agent_info(agent_id, user=request.user)
```

Common agent endpoints:

| Endpoint | Purpose |
| --- | --- |
| `GET /agents/` | List agents with name, model, file upload, RAG flags |
| `GET /agents/{id}/` | Agent info (description, instructions, permissions) |
| `GET /agents/{id}/tools/` | Tool list + per-integration status |
| `POST /agents/{id}/run/` | Non-streaming `agent.run()` |
| `POST /agents/{id}/reindex/` | `Agent.reindex(agent, memory_id, force_rebuild)` |

The run endpoint uses `protocol_handler.to_chat_messages(payload.messages)` then `agent.run(chat_messages, user=...)`:

{{< callout type="info" >}}
`/agents/{id}/run/` returns a JSON reply instead of a stream, useful for quick answers, extraction, or structured output.
{{< /callout >}}

## Threads and Messages

Conversations live in threads. `ThreadService` manages them; `MemoryService` links the memories (documents) an agent can retrieve in that thread. See the [Memories reference](/docs/manual/memories/) for the full `MemoryService` API and the demo's memory endpoints.

{{< callout type="info" >}}
Contributor? The [ThreadService](/docs/manual/thread-service/) manual page documents every method and its permission checks.
{{< /callout >}}

```python
from django_ai_sdk.memories.services import MemoryService
from django_ai_sdk.storage.services import ThreadService

# Create a thread for an agent
thread_id = await ThreadService.create_thread(agent_id=agent_id, user=request.user)
await MemoryService.link_memories(agent_id, thread_id, user=request.user)
```

### Send a message to a thread

```python
@router.post("/threads/{thread_id}/")
async def add_message_to_thread(request, thread_id: str, payload: ChatRequest):
    agent = await AgentService.get_agent(thread_id, user=request.user)
    return await agent.as_view(payload.messages, thread_id=thread_id, user=request.user)
```

When `thread_id` is provided, `as_view()`:

1. Resolves the storage adapter that holds the thread
2. Stores the incoming user message
3. Streams the assistant reply, persisted with the same `message_id` the frontend saw

### History and file metadata

```python
from django_ai_sdk.storage.services import aget_thread_file_meta, aget_thread_history

data = await aget_thread_history(thread_id, user=request.user)
# -> {"thread": ThreadInfo, "messages": [...]} in protocol format

meta = await aget_thread_file_meta(thread_id, user=request.user)
# -> {"file_count": int, "file_memory_id": str | None}
```

### Managing threads and messages

```python
# Threads
await ThreadService.threads(user=request.user, limit=100, offset=0)
await ThreadService.get_thread(thread_id, user=request.user)
await ThreadService.update_thread(thread_id, metadata={"agent_id": new_agent_id}, user=request.user)
await ThreadService.delete_thread(thread_id, user=request.user)
await ThreadService.delete_all_threads(user=request.user)

# Switch a thread to another agent (and relink memories)
await MemoryService.unlink_memories(old_agent_id, thread_id, user=request.user)
await ThreadService.update_thread(thread_id, metadata={"agent_id": new_agent_id}, user=request.user)
await MemoryService.link_memories(new_agent_id, thread_id, user=request.user)

# Messages — rate, soft delete, restore
await ThreadService.rate_message(
    thread_id, message_id, rating, feedback="...", user=request.user
)
await ThreadService.delete_message(thread_id, message_id, user=request.user)
await ThreadService.restore_message(thread_id, message_id, user=request.user)
```

---

## Non-Streaming Runs on Threads

Useful for structured extraction or a "quick answer" that doesn't need SSE:

```python
@router.post("/threads/{thread_id}/run/")
async def run_thread(request, thread_id: str, payload: ChatRequest):
    agent = await AgentService.get_agent(thread_id, user=request.user)
    chat_messages = agent.protocol_handler.to_chat_messages(payload.messages)
    result = await agent.run(chat_messages, thread_id=thread_id, user=request.user)
    return RunResponse(result=result, thread_id=thread_id)
```

---

## Runtime-Configured Agents

Besides code-defined agents, the SDK supports **runtime agents**: `AgentSettings` rows in the database that configure a base class, model, system prompt, tools, integrations, and access control without code changes. A UI (or admin) creates them via `AgentService`:

```python
config = await AgentService.create_runtime_agent(
    {
        "name": "Support Bot",
        "slug": "support-bot",
        "agent": "apps.agents.runtime.DefaultRuntimeAgent",
        "model": "openai/gpt-oss-120b",
        "system_prompt": "You are the support bot.",
        "tools": ["get_today", "get_memory_files"],
        "integrations": ["linear"],
        "title_generation": True,
    },
    user=request.user,
)

# Per-user / per-group access
await AgentService.add_agent_user(str(config.id), user_id, can_manage=True, user=request.user)
await AgentService.add_agent_group(str(config.id), group_id, can_manage=False, user=request.user)

# Manage
await AgentService.list_runtime_agents(user=request.user)
await AgentService.get_runtime_agent(runtime_id, user=request.user)
await AgentService.update_runtime_agent(runtime_id, data, user=request.user)
await AgentService.delete_runtime_agent(runtime_id, user=request.user)
```

What's available for runtime agents is declared in settings:

```python
# settings.py
# Base classes a runtime agent can be built on
AI_SDK_RUNTIME_AGENT_BASES = [
    "apps.agents.runtime.DefaultRuntimeAgent",
]

# Tools selectable in runtime agent configuration (key -> import path)
AI_SDK_RUNTIME_AGENT_TOOLS = {
    "get_today": "apps.agents.tools.get_today",
    "get_memory_files": "apps.agents.tools.get_memory_files",
}
```

`AgentService.get()` resolves runtime agents by their settings row ID, so the chat and thread endpoints work for them unchanged.

---

## Permissions

Agents declare `permissions` classes; `as_view()`, `history()`, and `AgentService` check them before acting, raising `PermissionDenied` when access is denied.

```python
from django_ai_sdk.permissions import ObjectPermissions

perms: ObjectPermissions = await agent_permissions(request.user, agent_id)
```

Return `ObjectPermissions` in your agent-info response so the frontend can show/hide controls. Domain-wide overrides live in settings:

```python
AI_SDK_PERMISSIONS = {
    "memory": ["apps.memories.permissions.AllowAnonymousMemoryPermission"],
    "thread": ["apps.agents.permissions.DemoThreadPermission"],
}
```

See the [Agents guide](/docs/agents/#permissions) for declaring permissions on agents, and the [Permissions reference](/docs/manual/permissions/) for the operation enum, domains, and built-in classes.

---

## Workflows

The SDK ships a workflow engine for orchestrating multi-step agent tasks. A `WorkflowDefinition` describes steps; `WorkflowService` runs and tracks them.

```python
from django_ai_sdk.workflows import WorkflowDefinition, WorkflowService

# Run an ad-hoc workflow
run = await WorkflowService.run(workflow, messages, user=request.user)

# Persisted workflows (WorkflowSettings)
record = await WorkflowService.create(name, workflow, user=request.user)
await WorkflowService.update(workflow_id, name=..., workflow=..., active=...)
await WorkflowService.run_by_id(workflow_id, messages, user=request.user)
await WorkflowService.get_run(run_id)

# Available actions (declared in AI_SDK_WORKFLOW_ACTIONS)
await WorkflowService.list_actions()
```

Action implementations are wired in settings:

```python
AI_SDK_WORKFLOW_ACTIONS = {
    "console_log": "apps.agents.actions.ConsoleLogAction",
}
```

## Ninja or DRF?

{{< callout type="warning" >}}
**DRF support is experimental.** We're actively building it out: the DRF router and serializers are not yet production-ready. **Ninja is the supported path.**
{{< /callout >}}

The demo includes a complete **Ninja router** (`views/ninja.py`) with typed schemas, plus an **experimental DRF router** (`views/drf.py`) with serializers, including a plain Django `View` chat handler that returns the SSE stream directly. Start with Ninja, and reach for DRF only if your project is already committed to it. Either way, `AgentService`, `ThreadService`, and the schemas in `django_ai_sdk.views.schemas` are framework-neutral.

## CORS and Streaming

{{< callout type="info" >}}
`stream_response()` (used by `as_view()`) sets SSE headers and a `x-vercel-ai-ui-message-stream: v1` header for Vercel-compatible frontends. Configure CORS for the streaming endpoint via `django-cors-headers` (already a dependency) or `AI_SDK_STREAM_CORS_ORIGIN`.
{{< /callout >}}
