---
title: Quick Start
sidebar:
  order: 1
---

This page builds your first AI agent in under 5 minutes.

## What you'll build

An agent that answers messages with streamed responses. By the end, you'll have a working `/api/chat` endpoint.

---

## Install

::install[django-ai-sdk]{tools="uv pip"}

The Haystack runtime is a plain dependency, not an extra. The adapters, pipelines
and generators are Haystack components. Everything else is opt-in, so you install
only what you use:

| Extra | Brings | Cost |
| --- | --- | --- |
| `qdrant` / `chroma` | a vector store for RAG, plus fastembed | `chroma` also pulls kubernetes, grpc and uvicorn[standard] |
| `rag` | both vector stores | |
| `mcp`, `files` | MCP integrations, document extraction | small |
| `anthropic`, `mistral`, `ollama`, `openrouter`, `huggingface` | one hosted provider each (see [Generators](../manual/generators.md)) | small |
| `providers` | all five of those | small |
| `transformers` | runs a Hugging Face model locally | pulls torch (about 500 MB) |
| `all` | everything above | |

::::details[Why a missing extra fails on first use]
Each optional feature lives in its own module that imports its package at the top,
and the SDK loads those modules on first use. So a Qdrant project never installs
chromadb, and an OpenAI project never installs mistral. Use a feature whose extra
is missing and you get the import error naming the package, for example
`No module named 'haystack_integrations.document_stores.qdrant'`. The table above
says which extra provides it.
::::

:::caution[Streaming needs an ASGI server]
Streaming needs an ASGI server, but the SDK does not choose one for you. Install
daphne, uvicorn or hypercorn yourself. The SDK doesn't depend on the packages the
demo project happens to use either: allauth, DRF, django-cors-headers or a
`django-tasks` backend.
:::

::install[django-ai-sdk[haystack]]{tools="uv pip"}

The `haystack` extra pulls in Haystack and the components agents run on. That's all you need to start.

Want the full set of extras (MCP, vector stores, document parsing, providers)?

::install[django-ai-sdk[all]]{tools="uv pip"}

Ready-made HTTP endpoints are separate, so you only pull in the framework you use:

::::tabs{sync="framework"}
:::tab[django-ninja]
::install[django-ai-sdk[ninja]]{tools="uv pip"}
:::

:::tab[Django REST framework]
::install[django-ai-sdk[drf]]{tools="uv pip"}
:::
::::

See [Views and Routing](../serving/views-and-routing.md#pick-a-starting-point).

Optional extras you can install narrowly:

| Extra | Adds |
| --- | --- |
| `typesafe` | The client for oversight's optional `TypeSafe` judge (a hosted commercial service) |
| `rag` / `chroma` / `qdrant` | Vector stores |
| `mcp` | MCP integrations |
| `files` | Document parsers |

## Configure Django

::::steps
1. Add the app to your `INSTALLED_APPS`:

   ```python
   # settings.py
   INSTALLED_APPS = [
       # ... your apps
       "django_ai_sdk",
   ]
   ```

2. Run migrations:

   ```bash
   python manage.py migrate
   ```
::::

This creates tables for conversation storage. The SDK manages them, so you won't need to think about them.

## Create an agent

An agent is a Python class:

```python
# agents.py
from django.conf import settings
from django_ai_sdk import Agent
from django_ai_sdk.adapters.base import Stream
from django_ai_sdk.agents import ToolAgent, ToolAgentConfig, auto_register
from django_ai_sdk.common import prompt
from django_ai_sdk.generators import openai_responses_chat
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

Four things to note:
- `name`: what to call your agent
- `instructions`: how it should behave (the system prompt)
- `get_pipeline_adapter`: returns the `Stream` that runs streaming chat
- `llm`: the generator factory `get_llm()` builds. It defaults to `openai_responses_chat`, so you can leave it out for OpenAI

`openai_responses_chat` works with any endpoint that speaks the OpenAI Responses API. Point `OPENAI_API_URL` at a local server (vLLM, llama.cpp) or leave it unset for OpenAI, and use `openai_chat` for a Chat Completions-only endpoint. Other providers get their own factory. See [Generators](../manual/generators.md). Every `AI_SDK_*` setting is listed in the [Settings Reference](../manual/settings.md).
## Register your agent

Your agent must be registered before it can be used. Add it to `AI_SDK_AGENTS` in your `settings.py`:

```python
# settings.py
AI_SDK_AGENTS = [
    "your_app.agents.ShakespeareAgent",
]
```

The SDK loads these classes at startup. Each agent gets a stable UUID derived from its module and class name, so you can retrieve it anywhere:

```python
from django_ai_sdk.agents.services import AgentService

# Get an agent by its stable ID
agent = await AgentService.get(agent_id)
```

The `@auto_register` decorator also registers agents. `AI_SDK_AGENTS` makes sure the module is imported.

## Wire it to a view

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

The `as_view()` method does the rest:
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

## Test it

::::steps
1. Start your server:

   ```bash
   python manage.py runserver
   ```

2. Send a message:

   ```bash
   curl -X POST http://localhost:8000/api/chat \
     -H "Content-Type: application/json" \
     -H "Accept: text/event-stream" \
     -d '{
       "messages": [{"role": "user", "parts": [{"type": "text", "text": "Hello!"}]}],
       "agent_id": "your-agent-uuid"
     }'
   ```
::::

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

Your agent now streams responses. Put together, the exchange reads:

::::chat[Shakespeare Bot]
:::system
You are a helpful assistant who speaks in Shakespearean English. Use thee, thou, and other Elizabethan expressions. Be poetic but always answer the user's question.
:::

:::user
Hello!
:::

:::assistant[Shakespeare Bot]
Hark! Good morrow to thee!
:::
::::

::::trace[What just happened?]
:::step[as_view() converted the protocol messages to internal ChatMessages.]{icon="code"}
:::

:::step[get_pipeline_adapter() built a ToolAgent pipeline and returned a Stream.]{icon="wrench"}
:::

:::step[Haystack streamed chunks, normalized into StreamEvents.]{icon="brain"}
:::

:::step[The protocol handler serialized the events to the Vercel protocol over SSE.]{icon="globe"}
:::
::::

All of this takes about 50 lines of code.

## Next steps
::::cards
:::card[Give It Tools]{href="../agents/agents.md"}
Let it call Python functions
:::
:::card[Add Knowledge]{href="../agents/agents.md#retrieval-augmented-generation-rag"}
Connect a knowledge base with RAG
:::
:::card[Add Integrations]{href="../agents/integrations.md"}
GitHub, Linear, weather, MCP
:::
:::card[Add Persistence]{href="../serving/views-and-routing.md"}
Threads and message history
:::
:::card[Go Deeper]{href="../architecture/how-it-works.md"}
How requests flow through the SDK
:::
:::card[Developer Manual]{href="../manual/index.md"}
Architecture details for contributors
:::
::::
