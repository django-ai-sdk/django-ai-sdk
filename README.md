# Django AI SDK

A Django SDK for building AI-powered applications with support for multiple LLM providers, RAG (Retrieval-Augmented Generation), and streaming responses.

## Project Status: Read This First

This is an **early preview**. We're actively iterating on the API and learning from real usage. Here's what that means for you:

- **Expect breaking changes**: APIs will shift as we find better patterns.
- **Migrations might be reset**: Don't rely on database schema stability between versions.
- **Not for production**: Use this for experimentation, prototypes, and side projects. Keep critical workloads elsewhere.
- **Watch the repo**: Things change quickly. Star & watch to stay in the loop.
- **Your feedback shapes the SDK**: Break things, open issues, tell us what hurts.

We'd love to have you along for the ride, just keep your seatbelt on.


## Install

```bash
pip install django-ai-sdk
```

Or with uv:

```bash
uv add django-ai-sdk
```

## Quick Start

### 1. Add to INSTALLED_APPS

```python
# settings.py
INSTALLED_APPS = [
    ...
    "django_ai_sdk",
]
```

Then run `python manage.py migrate`.

### 2. Define your agent

```python
# agents.py
from django_ai_sdk import Agent
from django_ai_sdk.adapters.base import Stream
from django_ai_sdk.generators import openai_responses_chat
from haystack import Pipeline


class HelpDeskAgent(Agent):
    name = "Help Desk"
    model = "gpt-5-mini"
    instructions = "You are a helpful support agent."
    llm = openai_responses_chat

    async def get_pipeline_adapter(self, thread_id=None, user=None):
        storage_adapter = await self.get_storage_adapter(thread_id)
        generator = self.get_llm()
        return Stream(
            pipeline=Pipeline(),
            generator=generator,
            storage_adapter=storage_adapter,
        )
```

For non-streaming tasks (title generation, structured output), use `Run` instead:

```python
from django_ai_sdk.adapters.base import Run

    async def get_run_adapter(self, thread_id=None, user=None):
        return Run(generator=self.get_llm())
```

### 3. Return a streaming response

```python
# views.py
from .agents import HelpDeskAgent

agent = HelpDeskAgent()


@router.post("/chat")
async def chat(request, payload: ChatRequest):
    return await agent.as_view(
        payload.messages,
        thread_id=payload.thread_id,
    )
```

## Integrations

An integration gives an agent tools from a third party: an MCP server, or an API
you wrap yourself. Each one is a small Django app that registers itself on `ready()`.

Enable a shipped integration by installing its app and naming it on an agent:

```python
# settings.py
INSTALLED_APPS = [
    "django_ai_sdk",
    "django_ai_sdk.integrations.mcp",  # required by any MCP-backed integration
    "django_ai_sdk.integrations.github",  # also: .linear, .notion, .weather
]

# INSTALLED_APPS decides which integrations exist; this configures them. The same
# shape as DATABASES or CACHES, keyed by integration name.
AI_SDK_INTEGRATIONS = {
    "github": {"TOKEN": env("GITHUB_MCP_TOKEN")},
}
```

```python
class HelpDeskAgent(Agent):
    integrations = ["github"]
```

A missing credential never breaks boot: the integration reports that it needs setup
and contributes no tools. Alongside secrets, each entry accepts `URL`, `TOOLS`,
`LABEL`, `SCOPE` and `AUTH`, so a self-hosted server, a narrower tool allow-list, or
per-user OAuth instead of a shared token is a settings change rather than a subclass.

It's an ordinary dict, so pulling values from a vault or an ini file needs no hook.
Just call whatever you like inside it.

### Adding your own

Three files, no models and no migrations:

```python
# myapp/integrations/zendesk/apps.py
from django_ai_sdk.integrations import IntegrationAppConfig


class ZendeskConfig(IntegrationAppConfig):
    default = True  # Django needs this to pick your AppConfig over the base
    name = "myapp.integrations.zendesk"
    integration = "myapp.integrations.zendesk.integration.ZendeskIntegration"
```

```python
# myapp/integrations/zendesk/integration.py
from django_ai_sdk.integrations import MCPIntegration


class ZendeskIntegration(MCPIntegration):
    name = "zendesk"  # registry key, and the AI_SDK_INTEGRATIONS key
    label = "Zendesk"
    url = "https://mcp.zendesk.example/mcp"
    auth = "token"  # "static" | "token" | "oauth"
    default_tools = []  # [] discovers every tool the server offers
```

Add `"myapp.integrations.zendesk"` to `INSTALLED_APPS`, add
`"zendesk": {"TOKEN": env("ZENDESK_API_TOKEN")}` to `AI_SDK_INTEGRATIONS`,
and list `"zendesk"` on an agent. For an API you wrap by hand, subclass
`APIIntegration` and set `tools` to `@haystack.tools.tool`-decorated functions.
`django_ai_sdk/integrations/weather/` is a complete, credential-free example, and
`github/`, `linear/` and `notion/` cover token and OAuth MCP servers.

Tools are namespaced per integration (`zendesk_search_tickets`), so two servers
exposing the same tool name don't collide. Tool lists are cached
stale-while-revalidate behind a per-integration circuit breaker, so a slow or dead
server costs one bounded wait and then reports itself as degraded.

### HTTP endpoints

The SDK ships no integrations router; it doesn't pick your web framework. Build
list/connect/disconnect/reconnect over `IntegrationService`
(`demo/apps/integrations/views/ninja.py` is a working reference), and include
the OAuth callback, which must sit at a fixed URL:

```python
(path("api/integrations/", include("django_ai_sdk.integrations.mcp.urls")),)
```

## Tracing

Every Haystack span of a run — the agent, each step, each LLM call, each tool
call — persisted as a `Trace` row and linked to the `Thread` and `Message` it was
produced for. Opt in the same way Haystack itself treats tracing: installing the
SDK changes nothing until you add the app.

```python
INSTALLED_APPS = [
    "django_ai_sdk",
    "django_ai_sdk.tracing",
]
```

That's the whole switch. The app ships the model plus migrations, and its
`AppConfig.ready()` enables the tracer at startup. If you'd rather call it
yourself, `tracing.enable_tracing(DefaultTracer())` swaps Haystack's single
global tracer — last call wins.

Spans record timing, hierarchy, model name and token counts. They do **not**
record prompts or replies unless you set `HAYSTACK_CONTENT_TRACING_ENABLED=true`,
which stores the content of queries, documents and answers in the `tags` JSON
column — treat that data as sensitive, and note that deleting a `Thread` or
`Message` cascades to its traces. Token counts are captured either way: Haystack
reports usage through the same content tags it gates, so the numbers are
harvested before the flag decides whether the payload is kept. That flag is read
once, when `haystack.tracing` is first imported, so it must be a real environment
variable set before the process starts — assigning it in `settings.py` does
nothing.

Some tags are static configuration rather than observability data, and they are
large: `haystack.agent.tools` serializes every tool definition in full. Drop them
by key:

```python
AI_SDK_TRACING_EXCLUDED_TAGS = [
    "haystack.agent.tools",
    "haystack.agent.state_schema",
]
```

### Reading traces

`TraceService` is the permission-checked entry point. A trace belongs to its
thread, so every method resolves the thread through `ThreadService` — enforcing
`VIEW_THREAD` and raising `PermissionDenied` — and raises `ValueError` when the
thread or message doesn't exist. There's no separate trace permission domain.

```python
from django_ai_sdk.tracing.services import TraceService

await TraceService.thread_traces(
    thread_id, user=user, message_id=None, operation_name=None, limit=100, offset=0
)
await TraceService.message_traces(message_id, user=user)
await TraceService.thread_token_usage(thread_id, user=user)
await TraceService.message_token_usage(message_id, user=user)
```

Message-scoped calls resolve the owning thread themselves, so a caller holding
only a message id doesn't need to know its thread. All four return pydantic
schemas from `django_ai_sdk.tracing.schemas`. For sync contexts (DRF
class-based views), import the module-level `thread_traces` /
`thread_token_usage` aliases instead.

For direct ORM work, where you're doing your own access control, the manager
carries the same query helpers:

```python
Trace.objects.for_thread(thread.id).token_usage()
# {'prompt_tokens': 1841, 'completion_tokens': 317, 'total_tokens': 2158}

Trace.objects.for_message(message.id).roots()  # the top span of the run
Trace.objects.llm_calls()  # one row per LLM call
```

**Never `Sum` the token columns over raw rows.** Usage is recorded on every span
that wraps an LLM call *and* aggregated again onto the agent's own
`haystack.agent.run` span, so a plain sum counts each call twice — three times
for an Agent inside a Pipeline. `.llm_calls()` selects leaf spans, which are
exactly the LLM calls in every path and never a rollup. Both the service and
`token_usage()` already count that way.

### Behaviour worth knowing

Correlation is automatic for anything streamed through the SDK: each run stamps
its spans with the assistant `Message` it mints and the `Thread` it belongs to.
For pipelines you run yourself, wrap them in `bind()`:

```python
from django_ai_sdk.tracing import bind

with bind(thread_id=thread.id, message_id=message.id):
    await pipeline.run_async({"messages": messages})
```

Writes are buffered: a whole trace tree lands in one `bulk_create` when its root
span exits, scheduled on the event loop so the ORM never blocks it. **Spans
appear only once the run completes** — a hung pipeline has no rows yet, so this
is not a live view. `await aflush()` is the barrier when you need one (tests,
shutdown).

One provider caveat: the OpenAI Responses API reports usage on a streamed
response without being asked, but Chat Completions omits it unless
`stream_options.include_usage` is requested. The adapters never reconfigure a
generator, so declare that on the agent (`llm_kwargs`) when streaming through
`openai_chat`; a generator that reports no usage leaves the token columns null.

## Features

- **RAG Pipelines**: BM25, ChromaDB, and Qdrant hybrid search with query expansion.
- **Streaming Responses**: Built-in SSE streaming. Works with Vercel AI SDK protocol.
- **Conversation Storage**: Automatic message persistence. Thread-based history out of the box.
- **Tool Calling**: MCP, memory, and custom tools, all managed by your Agent.
- **Artifacts**: 16 structured UI types (tables, plans, approval cards, code blocks, and more)
  submitted by the LLM via tool calls.
- **File Processing**: Document upload with pipeline-based processing (text, CSV, JSON,
  DOCX, PPTX, XLSX. Extraction transforms for metadata embedding.
- **Integrations**: Third-party tools as self-registering Django apps, with caching,
  circuit breaking and OAuth built in. See [Integrations](#integrations).
- **Tracing**: Opt-in Haystack tracing persisted to the ORM, with per-thread and
  per-message token accounting. See [Tracing](#tracing).
- **Automations**: Run agent workflows on a cron schedule, either as the deployment
  itself or once per user who opted in. You supply the clock.
- **Reindexing**: Hot-reload documents. Cached embeddings with simple refresh API.

## Documentation

Full documentation and examples: [github.com/django-ai-sdk/django-ai-sdk](https://github.com/django-ai-sdk/django-ai-sdk)
