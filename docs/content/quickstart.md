---
title: Quick Start
type: docs
weight: 1
---

Let's build your first AI agent in under 5 minutes.

## What You'll Build

A simple agent that responds to messages with streaming AI responses. By the end, you'll have a working `/api/chat` endpoint.

---

## Install

```bash
pip install django-ai-sdk
```

The Haystack runtime is a plain dependency, not an extra: the adapters, pipelines
and generators are Haystack components. Everything else is opt-in, so you install
only what you use:

| Extra | Brings | Cost |
| --- | --- | --- |
| `qdrant` / `chroma` | a vector store for RAG, plus fastembed | `chroma` also pulls kubernetes, grpc and uvicorn[standard] |
| `rag` | both vector stores | |
| `mcp`, `files` | MCP integrations, document extraction | small |
| `anthropic`, `mistral`, `ollama`, `openrouter`, `huggingface` | one hosted provider each (see [Generators](/manual/generators/)) | small |
| `providers` | all five of those | small |
| `transformers` | runs a Hugging Face model locally | **pulls torch (~500 MB)** |
| `all` | everything above | |

Each optional feature lives in its own module that imports its package at the top,
and those modules are loaded on first use - so a Qdrant project never installs
chromadb, and an OpenAI project never installs mistral. Reach for a feature whose
extra is missing and you get the import error naming the package, e.g.
`No module named 'haystack_integrations.document_stores.qdrant'`; the table above
says which extra provides it.

Streaming needs an ASGI server, but the SDK does not choose one for you: install
daphne, uvicorn or hypercorn yourself. Nothing the demo project happens to use -
allauth, DRF, django-cors-headers, a `django-tasks` backend - is a dependency of the
SDK either.

```bash
```

The `haystack` extra pulls in Haystack and the components agents run on. That's all you need to start.

Want the full set of extras (MCP, DRF views, document parsing)?

```bash
pip install "django-ai-sdk[all]"
```

{{< callout type="warning" >}}
**DRF views are experimental.** The `all` extra includes DRF routers and serializers, but the DRF path is still in active development: use the [Ninja views](/views-and-routing/#ninja-or-drf) for production.
{{< /callout >}}

## Configure Django

Add the app to your `INSTALLED_APPS`:

```python
# settings.py
INSTALLED_APPS = [
    # ... your apps
    "django_ai_sdk",
]
```

Run migrations:

```bash
python manage.py migrate
```

This creates tables for conversation storage. You won't need to think about them: the SDK handles it automatically.

## Create an Agent

An agent is just a Python class with personality:

```python
# agents.py
from django.conf import settings
from django_ai_sdk import Agent
from django_ai_sdk.adapters.base import Stream
from django_ai_sdk.agents import auto_register
from django_ai_sdk.common import prompt
from django_ai_sdk.generators import openai_responses_chat
from django_ai_sdk.pipelines.haystack import ToolAgent, ToolAgentConfig
from django_ai_sdk.protocols.vercel import VercelProtocolHandler
from django_ai_sdk.storage.db import DbStorageAdapter

@auto_register
class ShakespeareAgent(Agent):
    """An agent that speaks like Shakespeare."""

    name = "Shakespeare Bot"
    model = settings.AI_SDK_DEFAULT_MODEL
    instructions = prompt(
        "You are a helpful assistant who speaks in Shakespearean English. "
        "Use thee, thou, and other Elizabethan expressions. "
        "Be poetic but always answer the user's question."
    )
    protocol = VercelProtocolHandler
    storage_adapter = DbStorageAdapter
    llm = openai_responses_chat

    async def get_pipeline_adapter(self, thread_id=None, user=None):
        generator = self.get_llm()
        tool_agent = ToolAgent(
            config=ToolAgentConfig(
                model=self.get_model(),
                system_prompt=self.get_system_prompt(),
                tools=await self.get_tools(thread_id=thread_id or "", user=user),
            ),
            generator=generator,
        )
        return Stream(
            pipeline=tool_agent.pipeline(),
            generator=generator,
            storage_adapter=await self.get_storage_adapter(thread_id),
        )
```

Three things to note:
- **name**: What to call your agent
- **instructions**: How it should behave (the system prompt)
- **get_pipeline_adapter**: Returns the `Stream` that powers streaming chat
- **llm**: The generator factory `get_llm()` builds - it defaults to `openai_responses_chat`, so you can leave it out for OpenAI

`openai_responses_chat` works with any endpoint that speaks the OpenAI Responses API. Point `OPENAI_API_URL` at a local server (vLLM, llama.cpp) or leave it unset for OpenAI, and use `openai_chat` for a Chat Completions-only endpoint. Other providers get their own factory - see [Generators](/manual/generators/). Every `AI_SDK_*` setting is listed in the [Settings Reference](/manual/settings/).

That's it. No complex configuration. No framework setup.

## Register Your Agent

Your agent must be **registered** before it can be used. Add it to `AI_SDK_AGENTS` in your `settings.py`:

```python
# settings.py
AI_SDK_AGENTS = [
    "your_app.agents.ShakespeareAgent",
]
```

The SDK loads these classes at startup. Each agent gets a **stable UUID** derived from its module and class name, so you can retrieve it anywhere:

```python
from django_ai_sdk.agents.services import AgentService

# Get an agent by its stable ID
agent = await AgentService.get(agent_id)
```

Agents are also registered automatically by the `@auto_register` decorator: `AI_SDK_AGENTS` just makes sure the module is imported.

## Wire It to a View

Connect your agent to Django Ninja:

```python
# views.py
from ninja import Router
from django_ai_sdk.agents.services import AgentService
from django_ai_sdk.views.schemas import ChatRequest

router = Router()

@router.post("/chat")
async def chat(request, payload: ChatRequest):
    """Chat with Shakespeare Bot."""
    agent = await AgentService.get(payload.agent_id)
    return await agent.as_view(payload.messages, user=request.user)
```

The `as_view()` method handles everything:
- Protocol conversion (Vercel AI SDK Data Stream Protocol in, `ChatMessage`s internally)
- Building the pipeline adapter and streaming to the model
- Returning an SSE `StreamingHttpResponse`

## Add to URLs

```python
# urls.py
from ninja import NinjaAPI
from your_app.views import router

api = NinjaAPI()
api.add_router("/", router)

urlpatterns = [
    path("api/", api.urls),
]
```

## Test It

Start your server:

```bash
python manage.py runserver
```

Send a message:

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "messages": [{"role": "user", "parts": [{"type": "text", "text": "Hello!"}]}],
    "agent_id": "your-agent-uuid"
  }'
```

You'll see a streaming response:

```
data: {"type":"start","messageId":"msg_abc123"}
data: {"type":"text-start","id":"text_001"}
data: {"type":"text-delta","id":"text_001","delta":"Hark!"}
data: {"type":"text-delta","id":"text_001","delta":" Good"}
data: {"type":"text-delta","id":"text_001","delta":" morrow"}
data: {"type":"text-delta","id":"text_001","delta":" to"}
data: {"type":"text-delta","id":"text_001","delta":" thee!"}
data: {"type":"text-end","id":"text_001"}
data: {"type":"finish"}
data: [DONE]
```

**You did it!** Your agent is live and streaming responses.

{{< callout type="info" >}}
**What just happened?** `as_view()` converted the protocol messages to internal `ChatMessage`s, `get_pipeline_adapter()` built a `ToolAgent` pipeline and returned a `Stream`, Haystack streamed chunks normalized into `StreamEvent`s, and the events were serialized to the Vercel protocol over SSE. All in about 50 lines of code.
{{< /callout >}}

## Next Steps

{{< cards >}}
  {{< card link="/agents" title="Give It Tools" icon="cog" subtitle="Let it call Python functions" >}}
  {{< card link="/agents#retrieval-augmented-generation-rag" title="Add Knowledge" icon="search" subtitle="Connect a knowledge base with RAG" >}}
  {{< card link="/integrations" title="Add Integrations" icon="collection" subtitle="GitHub, Linear, weather, MCP" >}}
  {{< card link="/views-and-routing" title="Add Persistence" icon="database" subtitle="Threads and message history" >}}
  {{< card link="/how-it-works" title="Go Deeper" icon="cube" subtitle="How requests flow through the SDK" >}}
  {{< card link="/manual" title="Developer Manual" icon="book-open" subtitle="Architecture details for contributors" >}}
{{< /cards >}}
