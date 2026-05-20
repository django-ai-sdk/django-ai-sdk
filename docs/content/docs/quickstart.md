---
title: Quick Start
type: docs
weight: 1
next: /docs
---

Let's build your first AI assistant in under 5 minutes.

## What You'll Build

A simple assistant that responds to messages with streaming AI responses. By the end, you'll have a working `/api/chat` endpoint.

---

## Step 1: Install

```bash
pip install django-ai-sdk openai
```

That's all you need to start. The SDK handles the rest.

---

## Step 2: Configure Django

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

This creates tables for conversation storage. You won't need to think about them—the SDK handles it automatically.

---

## Step 3: Create an Assistant

An assistant is just a Python class with personality:

```python
# assistants.py
from django.conf import settings
from openai import AsyncOpenAI
from django_ai_sdk import Assistant
from django_ai_sdk.adapters.openai import OpenAIAdapter

class ShakespeareAssistant(Assistant):
    """An assistant that speaks like Shakespeare."""
    
    name = "Shakespeare Bot"
    model = "gpt-4o-mini"
    instructions = [
        "You are a helpful assistant who speaks in Shakespearean English.",
        "Use thee, thou, and other Elizabethan expressions.",
        "Be poetic but always answer the user's question.",
    ]
    
    async def get_pipeline_adapter(self, thread_id=None):
        return OpenAIAdapter(
            client=AsyncOpenAI(api_key=settings.OPENAI_API_KEY),
            model=self.model,
            store=True,  # Automatically save conversations
        )
```

Three things to note:
- **name** — What to call your assistant
- **instructions** — How it should behave (the system prompt)
- **get_pipeline_adapter** — Which AI backend to use

That's it. No complex configuration. No framework setup.

---

## Step 4: Register Your Assistant

Your assistant must be **registered** before it can be used. Add it to `AI_SDK_ASSISTANTS` in your `settings.py`:

```python
# settings.py
AI_SDK_ASSISTANTS = [
    "your_app.assistants.ShakespeareAssistant",
]
```

The SDK automatically loads and instantiates these assistants when Django starts. Each assistant gets a **stable UUID** based on its module and class name, so you can retrieve it anywhere:

```python
from django_ai_sdk.assistants.registry import registry

# Get assistant by its stable ID
assistant = registry.get(assistant_id)
```

---

## Step 5: Wire It to a View

Connect your assistant to Django Ninja:

```python
# views.py
from typing import List, Optional
from ninja import Router, Schema
from django_ai_sdk.assistants.registry import registry

router = Router()

# Pydantic schemas for request/response
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
    """Chat with Shakespeare Bot."""
    # Get assistant from registry by its UUID
    assistant = registry.get("uuid-from-your-settings")
    if not assistant:
        return {"error": "Assistant not found"}, 404
    
    return await assistant.as_view(payload.messages)
```

The `as_view()` method handles everything:
- Protocol conversion
- Streaming to the AI backend
- Returning SSE responses

---

## Step 6: Add to URLs

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

---

## Step 7: Test It

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
    "messages": [{
      "role": "user",
      "parts": [{"type": "text", "text": "Hello!"}]
    }]
  }'
```

You'll see a streaming response:

```
data: {"type":"start","messageId":"msg_abc123"}
data: {"type":"text-delta","id":"text_001","delta":"Hark!"}
data: {"type":"text-delta","id":"text_001","delta":" Good"}
data: {"type":"text-delta","id":"text_001","delta":" morrow"}
data: {"type":"text-delta","id":"text_001","delta":" to"}
data: {"type":"text-delta","id":"text_001","delta":" thee!"}
data: {"type":"finish"}
data: [DONE]
```

**You did it!** Your assistant is live and streaming responses.

---

## What's Happening Under the Hood?

Don't worry about this for now, but here's what just happened:

1. Your view received the POST request
2. `assistant.as_view()` converted the request to internal format
3. The adapter connected to OpenAI with streaming enabled
4. Server-Sent Events streamed the response back
5. The conversation was automatically saved

All in 30 lines of code.

---

## Next Steps

Now that you have a working assistant, let's make it more powerful:

**Add Memory** — Make it remember conversations → [Building Assistants](/docs/building-assistants)

**Give It Tools** — Let it call functions → [Tools and Agents](/docs/tools-and-agents)

**Add Knowledge** — Connect a knowledge base → [RAG Guide](/docs/rag)

**Go to Production** — Learn about deployment and scaling → [Guides](/docs)

---

*Questions? The [Guides](/docs) cover everything in depth, or check the [Developer Manual](/docs/manual) for architecture details.*
