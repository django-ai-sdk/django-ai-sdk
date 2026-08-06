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
from django.conf import settings
from django_ai_sdk import Agent
from django_ai_sdk.adapters.base import Stream
from haystack import Pipeline
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.utils import Secret


class HelpDeskAgent(Agent):
    name = "Help Desk"
    model = "gpt-4o"
    instructions = "You are a helpful support agent."

    async def get_pipeline_adapter(self, thread_id=None, user=None):
        storage_adapter = await self.get_storage_adapter(thread_id)
        generator = OpenAIChatGenerator(
            model=self.get_model(),
            api_key=Secret.from_token(settings.OPENAI_API_KEY),
        )
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
        generator = OpenAIChatGenerator(
            model=self.get_model(),
            api_key=Secret.from_token(settings.OPENAI_API_KEY),
        )
        return Run(generator=generator)
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
- **Reindexing**: Hot-reload documents. Cached embeddings with simple refresh API.

## Documentation

Full documentation and examples: [github.com/django-ai-sdk/django-ai-sdk](https://github.com/django-ai-sdk/django-ai-sdk)
