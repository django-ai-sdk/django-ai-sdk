---
title: Views and Routing
type: docs
weight: 5
---

This page covers exposing agents over HTTP: ready-made endpoints for django-ninja and Django REST framework, a framework-free chat view, and the services underneath if you'd rather write your own.

## Pick a starting point

| You use | Install | Mount |
| --- | --- | --- |
| django-ninja | `pip install django-ai-sdk[ninja]` | `get_*_router()` from `django_ai_sdk.contrib.ninja` |
| Django REST framework | `pip install django-ai-sdk[drf]` | `include("django_ai_sdk.contrib.drf.urls")` |
| Neither / your own style | nothing extra | `ChatView` plus the services below |

The SDK never imports Ninja or DRF itself; the `contrib` layers are optional. Both are thin: every endpoint calls a service (`ThreadService`, `AgentService`, `MemoryService`, `WorkflowService`, `IntegrationService`, `TraceService`), and the services do the permission checks. So whichever you choose, behaviour is the same.

{{< callout type="warning" >}}
The contrib layers are **beta**: URLs and response shapes may still change between minor releases. Pin your version if a frontend client is generated from them.
{{< /callout >}}

### django-ninja

```python
# urls.py
from django.urls import include, path
from ninja import NinjaAPI
from ninja.security import SessionAuth

from django_ai_sdk.contrib import ninja as ai_sdk_routers

api = NinjaAPI(auth=SessionAuth())
ai_sdk_routers.register_error_handlers(api)  # every error answers with a code, see Error Handling

api.add_router("/", ai_sdk_routers.get_threads_router())
api.add_router("/", ai_sdk_routers.get_agents_router())
api.add_router("/", ai_sdk_routers.get_workflows_router())
api.add_router("/memories", ai_sdk_routers.get_memories_router())
api.add_router("/integrations", ai_sdk_routers.get_integrations_router())

urlpatterns = [
    path("api/", api.urls),
    path("api/integrations/", include("django_ai_sdk.integrations.mcp.urls")),  # OAuth callback
]
```

Each `get_*_router()` builds a **new** Router, so you can trim it and add your own endpoints next to the SDK's without changing the router for anyone else. That is why they are functions rather than module-level routers: Ninja can't remove an endpoint from a router, so `exclude` has to happen while it is built.

```python
router = ai_sdk_routers.get_threads_router(exclude={"delete_all_threads"})

@router.get("/threads/{thread_id}/export/")
async def export_thread(request, thread_id: str):
    ...

api.add_router("/", router)
```

List endpoints take `?limit=&offset=`: `limit` defaults to 100 and is capped at 100 (a larger value is a 422), the same bound as the DRF layer. The services themselves accept `limit=None` for everything; over HTTP a client pages through with `offset`, and an endpoint that really must return everything is one you write yourself.

`exclude` takes endpoint function names, which are also the OpenAPI `operationId`s; a misspelt name raises at startup. Your own routers can live on the same `NinjaAPI`, which is how the studio demo adds its health, digest and account endpoints (`demos/studio/apps/agents/views/ninja.py`).

`register_error_handlers()` covers the whole API, so your own endpoints on it answer errors the same way: raise `NotFound` or `UserError` from `django_ai_sdk.errors` and the client gets the matching code. See [Error Handling](/errors/).

### Django REST framework

```python
# urls.py
urlpatterns = [
    path("api/", include("django_ai_sdk.contrib.drf.urls")),
    path("api/integrations/", include("django_ai_sdk.integrations.mcp.urls")),  # OAuth callback
]
```

That registers `ThreadViewSet`, `MessageViewSet`, `AgentViewSet`, `RuntimeAgentViewSet`, `MemoryViewSet`, `WorkflowViewSet` and `IntegrationViewSet`, plus the streaming `ChatView` at `threads/<id>/chat/`. Authentication, permission and throttle classes come from your `REST_FRAMEWORK` settings. The viewsets answer errors with [error codes](/errors/) whatever your `EXCEPTION_HANDLER` is; set it to `django_ai_sdk.contrib.drf.exception_handler` to give your own views the same.

To change one resource, subclass its viewset and register it on your own router:

```python
from rest_framework.decorators import action
from rest_framework.routers import DefaultRouter
from rest_framework.throttling import UserRateThrottle

from django_ai_sdk.contrib.drf import ThreadViewSet

class MyThreadViewSet(ThreadViewSet):
    throttle_classes = [UserRateThrottle]

    @action(detail=True, methods=["get"])
    def export(self, request, thread_id=None):
        ...

router = DefaultRouter()
router.register("threads", MyThreadViewSet, basename="thread")
```

The viewsets are ordinary `GenericViewSet`s:

- **Requests** are validated by DRF serializers (`django_ai_sdk.contrib.drf.serializers`), so you get DRF's 400 format, browsable-API forms and schema generation. Each viewset lists its request serializers per action in `serializer_classes`.
- **Responses** go through serializers too; they read the services' pydantic models by attribute.
- **Lists** take `?limit=&offset=` through `ApiPagination` (a `LimitOffsetPagination`, default 100, max 100) and return plain lists. Set `pagination_class` on a subclass to change the bounds.
- **URL names** use the `ai-sdk-` prefix: `ai-sdk-thread-list`, `ai-sdk-memory-detail`, `ai-sdk-chat`, and so on.
- DRF views are sync; they call the services' sync wrappers (`django_ai_sdk.storage.services.list_threads` and friends).

Need to change what these endpoints do? See [Overriding Views](/overriding-views/) for replacing, removing and extending endpoints in both frameworks.

### Your own views

Chat is the one endpoint that is awkward to write in any framework, so the SDK ships it as a plain async Django view:

```python
from django_ai_sdk.views.chat import ChatView

urlpatterns = [
    path("api/threads/<str:thread_id>/chat/", ChatView.as_view()),  # the thread's agent
    path("api/chat/", ChatView.as_view(agent="support-bot")),  # no thread: stores nothing
]
```

Without a thread the conversation isn't persisted and the client sends the whole history every turn. That suits a help widget or a one-off question. The agent comes from the view (`agent=`), never from the request body, so a client can't switch agents through that URL. The `CHAT` permission is checked either way.

Subclass it and override `get_agent()` or `error_response(exc)` to change how the agent is picked or how failures look. For everything else, call the services as shown in the rest of this page. They raise typed errors (`NotFound`, `PermissionDenied`, ...), and `django_ai_sdk.errors.error_response(exc)` turns any exception into the status and body both contrib layers send. See [Error Handling](/errors/) for the codes.

## The Chat Endpoint

Agents expose `as_view()`, which returns a ready-to-return `StreamingHttpResponse`. The minimal hand-written endpoint:

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

Conversations live in threads. `ThreadService` manages them; `MemoryService` links the memories (documents) an agent can retrieve in that thread. See the [Memories reference](/manual/memories/) for the full `MemoryService` API and the demo's memory endpoints.

{{< callout type="info" >}}
Contributor? The [ThreadService](/manual/thread-service/) manual page documents every method and its permission checks.
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
await ThreadService.threads(user=request.user, limit=100, offset=0)  # limit=None returns all
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
from django_ai_sdk.views.permissions import agent_permissions, memory_permissions, thread_permissions

perms = await agent_permissions(request.user, agent_id)  # ObjectPermissions(can_read=..., ...)
```

Return `ObjectPermissions` in your responses so the frontend can show or hide controls; both contrib layers already do. Domain-wide overrides live in settings:

```python
AI_SDK_PERMISSIONS = {
    "memory": ["apps.memories.permissions.AllowAnonymousMemoryPermission"],
    "thread": ["apps.agents.permissions.DemoThreadPermission"],
}
```

See the [Agents guide](/agents/#permissions) for declaring permissions on agents, and the [Permissions reference](/manual/permissions/) for the operation enum, domains, and built-in classes.

---

## Workflows

The SDK ships a workflow engine for orchestrating multi-step agent tasks. A `WorkflowDefinition` describes steps; `WorkflowService` runs and tracks them.

```python
from django_ai_sdk.workflows import WorkflowDefinition, WorkflowService

# Run an ad-hoc workflow
run = await WorkflowService.run(workflow, inputs={"document": doc}, user=request.user)

# Persisted workflows (WorkflowSettings)
record = await WorkflowService.create(name, workflow, user=request.user)
await WorkflowService.update(workflow_id, name=..., workflow=..., active=...)
await WorkflowService.run_by_id(workflow_id, inputs={"document": doc}, user=request.user)
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

## CORS and Streaming

{{< callout type="info" >}}
`stream_response()` (used by `as_view()`) sets SSE headers and a `x-vercel-ai-ui-message-stream: v1` header for Vercel-compatible frontends. Configure CORS for the streaming endpoint via `django-cors-headers` (already a dependency) or `AI_SDK_STREAM_CORS_ORIGIN`.
{{< /callout >}}
