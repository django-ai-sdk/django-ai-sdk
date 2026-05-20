---
title: Getting Started
next: building-assistants
weight: 1
---

## Installation

```bash
pip install django-ai-sdk
```

Or add it to your project alongside its core dependencies:

```bash
pip install django>=6.0.2 django-ninja>=1.5.3 haystack-ai>=2.24.1 openai>=1.0.0
```

## Setup

Add `django_ai_sdk` to your installed apps and run migrations:

```python
# settings.py
INSTALLED_APPS = [
    ...,
    "django_ai_sdk",
]

# Define your assistants
AI_SDK_ASSISTANTS = [
    "your_app.assistants.MyAssistant",
]
```

```bash
python manage.py migrate
```

This creates the `Thread` and `Message` models used for conversation persistence.

## Your First Assistant

Create an assistant class. You need three things: a name, instructions, and an adapter:

```python
# your_app/assistants.py
from django_ai_sdk import Assistant
from django_ai_sdk.adapters.openai import OpenAIAdapter
from django_ai_sdk.protocols.vercel import VercelProtocolHandler
from openai import AsyncOpenAI

class MyAssistant(Assistant):
    name = "My Bot"
    model = "gpt-4"
    instructions = [
        "You are a helpful assistant.",
        "Be concise and friendly.",
    ]
    protocol = VercelProtocolHandler

    async def get_pipeline_adapter(self, thread_id=None):
        storage = await self.get_storage_adapter(thread_id)
        return OpenAIAdapter(
            client=AsyncOpenAI(api_key="your-api-key"),
            model=self.model,
            store=True,
            storage_adapter=storage,
        )
```

## Wire It to a View

Create a Django Ninja endpoint that retrieves the assistant from the registry and calls `as_view()`:

```python
# your_app/views.py
from typing import List, Optional
from ninja import Router, Schema
from django_ai_sdk.assistants.registry import registry

router = Router()

class MessagePart(Schema):
    type: str
    text: Optional[str] = None

class Message(Schema):
    role: str
    parts: List[MessagePart]

class ChatRequest(Schema):
    messages: List[Message]

@router.post("/chat")
async def chat(request, payload: ChatRequest):
    # Get assistant by its stable UUID from registry
    assistant = registry.get("db9540d3-37ef-5c7a-83be-70f1798994f1")
    if not assistant:
        return {"error": "Assistant not found"}, 404
    
    return await assistant.as_view(payload.messages)
```

Hook the router into your URL config:

```python
# your_project/urls.py
from ninja import NinjaAPI
from your_app.views import router

api = NinjaAPI()
api.add_router("/your-app", router)

urlpatterns = [
    path("api/", api.urls),
]
```

## Test It

```bash
curl -X POST http://localhost:8000/api/your-app/chat \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"messages":[{"role":"user","parts":[{"type":"text","text":"Hello!"}]}]}'
```

You'll get back a stream of Server-Sent Events in the Vercel AI SDK Data Stream Protocol format:

```
data: {"type":"start","messageId":"msg_abc123"}
data: {"type":"text-start","id":"text_def456"}
data: {"type":"text-delta","id":"text_def456","delta":"Hello"}
data: {"type":"text-delta","id":"text_def456","delta":"! How can I help?"}
data: {"type":"text-end","id":"text_def456"}
data: {"type":"finish"}
data: [DONE]
```

This works out of the box with the Vercel AI SDK frontend (`@ai-sdk/react`'s `useChat` hook).

## Next Steps

**New to Django AI SDK?** → Start with the [5-Minute Quick Start](/docs/quickstart/)

**Ready to go deeper?** → Explore the guides:
- [Building Assistants](/docs/building-assistants/) — Configuration, memory, adapters
- [CLI](/docs/cli/) — Command line interface for testing and debugging
- [Tools and Agents](/docs/tools-and-agents/) — Give your AI capabilities
- [Protocols and Adapters](/docs/protocols-and-adapters/) — Backend flexibility
- [Views and Routing](/docs/views-and-routing/) — API endpoints and threads
- [How It Works](/docs/how-it-works/) — Internal architecture

**Contributing or need internals?** → See the [Developer Manual](/docs/manual/)

---

## Running the Demo

The repo includes a full demo project with pirate-themed assistants:

```bash
cd demo
cp .env.example .env   # add your API keys
python manage.py migrate
python manage.py runserver
```

The demo has four different assistant implementations showing different patterns.
