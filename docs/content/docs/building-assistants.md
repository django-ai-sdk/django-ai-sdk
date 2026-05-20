---
title: Building Assistants
type: docs
prev: /docs
next: tools-and-agents
weight: 2
---

Assistants are the main thing you build with Django AI SDK. Each assistant is a class that encapsulates an AI personality, its configuration, and how it connects to an AI backend.

## The Assistant Class

Every assistant subclasses `Assistant` and implements `get_pipeline_adapter()`:

```python
from django_ai_sdk import Assistant

class MyAssistant(Assistant):
    name = "My Bot"
    model = "gpt-4"
    instructions = "You are a helpful assistant."
    protocol = VercelProtocolHandler
    storage_adapter = DbStorageAdapter  # or MemoryStorageAdapter

    async def get_pipeline_adapter(self, thread_id=None):
        # Get storage adapter for this thread
        storage = await self.get_storage_adapter(thread_id)
        # Return an adapter that connects to your AI backend
        return SomeAdapter(..., storage_adapter=storage)
```

### Registration

Before an assistant can be used, it must be **registered**. There are two ways to register assistants:

**Method 1: Settings-based (Recommended)**

Define `AI_SDK_ASSISTANTS` in your `settings.py`. The registry automatically loads these when Django starts:

```python
# settings.py
AI_SDK_ASSISTANTS = [
    "myapp.assistants.MyAssistant",
    "myapp.assistants.PirateAssistant",
]
```

**Method 2: Decorator-based**

Apply the `@auto_register` decorator to your assistant class:

```python
from django_ai_sdk.assistants import auto_register

@auto_register
class MyAssistant(Assistant):
    name = "My Bot"
    # ... rest of configuration
```

**Both methods can be combined** — a class will only be registered once. The recommended approach is settings-based for production, with the decorator as a convenient alternative.

### Stable Assistant IDs

Each registered assistant gets a **stable UUID v5 ID** based on its module and class name (`module.ClassName`). This ID is:

- **Deterministic**: Same class always gets the same ID across restarts
- **Unique**: Different classes get different IDs automatically
- **Retrievable**: Look up assistants by ID in your views

```python
from django_ai_sdk.assistants.registry import registry

# Get assistant by its stable UUID
assistant = registry.get("db9540d3-37ef-5c7a-83be-70f1798994f1")
```

### Configuration (Class Variables)

| Attribute | Type | What it does |
|-----------|------|-------------|
| `name` | `str` | Display name for the assistant |
| `description` | `str` | Optional description |
| `model` | `str` | Model identifier (e.g. `"gpt-4"`) |
| `instructions` | `str` or `list[str]` | System prompt -- lists get joined with newlines |
| `protocol` | class | Protocol handler class (defaults to `VercelProtocolHandler`) |
| `storage_adapter` | class | Storage adapter class (defaults to `DbStorageAdapter`) |
| `rag_provider` | BaseRAGProvider | RAG provider for knowledge retrieval (optional) |
| `max_history` | `int` or `None` | Maximum messages to send to LLM (None = unlimited) |
| `warmup_on_init` | `bool` | Auto-warmup RAG on initialization (default: False) |

### max_history — Limit Conversation Context

Control how much conversation history is sent to the LLM:

```python
class ConciseAssistant(Assistant):
    name = "Concise Bot"
    model = "gpt-4"
    max_history = 10  # Only send last 10 messages to LLM
    instructions = ["You are a helpful assistant."]
```

Use this to:
- Reduce API costs (fewer tokens per request)
- Prevent context window overflow
- Keep conversations focused on recent context

When `max_history` is set, only the most recent N messages are sent to the LLM. Older messages remain in storage but aren't included in the context.

### warmup_on_init — Auto-warmup RAG

For assistants that need RAG ready immediately:

```python
class DocumentAssistant(Assistant):
    name = "Document Helper"
    model = "gpt-4"
    rag_provider = BaseRAGProvider()
    warmup_on_init = True  # Build RAG index on startup
```

**Note:** This can slow down Django startup if you have many documents. Consider calling `await assistant.warmup()` in a background task instead.

## Simple OpenAI Assistant

The simplest pattern -- direct OpenAI API, no tools, no pipelines. From the demo (`demo/piratespeak/assistants/pirate_openai.py`):

```python
from django.conf import settings
from openai import AsyncOpenAI
from django_ai_sdk import Assistant
from django_ai_sdk.adapters.openai import OpenAIAdapter
from django_ai_sdk.protocols.vercel import VercelProtocolHandler

class PirateOpenAIAssistant(Assistant):
    name = "Simple OpenAI Pirate"
    model = "gpt-4"
    instructions = [
        "You are a helpful AI assistant who always responds like a pirate.",
        "Use pirate language, expressions, and mannerisms in all your responses.",
        "Be creative with pirate slang but keep responses helpful and informative.",
        "Address users as 'matey', 'landlubber', or 'crew member'.",
    ]
    protocol = VercelProtocolHandler

    async def get_pipeline_adapter(self, thread_id=None):
        # Get the storage adapter for this thread (creates thread if needed)
        storage_adapter = await self.get_storage_adapter(thread_id)
        
        return OpenAIAdapter(
            client=AsyncOpenAI(
                api_key=settings.OPENAI_API_KEY,
                base_url=getattr(settings, "OPENAI_API_URL", None),
            ),
            model=self.model,
            store=True,
            storage_adapter=storage_adapter,
        )
```

That's it. Instantiate it and call `as_view()` from any async Django view.

## Haystack Pipeline Assistant

If you need Haystack pipelines (for multi-provider support, RAG, or complex workflows), use `HaystackAdapter`. From the demo (`demo/piratespeak/assistants/pirate_basic.py`):

```python
from django.conf import settings
from haystack import Pipeline
from haystack.components.agents import Agent as HaystackAgent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.utils import Secret
from django_ai_sdk import Assistant
from django_ai_sdk.adapters.haystack import HaystackAdapter
from django_ai_sdk.protocols.vercel import VercelProtocolHandler

class PirateBasicAssistant(Assistant):
    name = "Basic Pirate Assistant"
    model = "gpt-4"
    instructions = [
        "You are a helpful AI assistant who always responds like a pirate.",
        "Use pirate language, expressions, and mannerisms.",
    ]
    protocol = VercelProtocolHandler

    async def get_pipeline_adapter(self, thread_id=None):
        # Get the storage adapter for this thread
        storage_adapter = await self.get_storage_adapter(thread_id)
        
        pipeline = Pipeline()

        pirate_agent = HaystackAgent(
            chat_generator=OpenAIChatGenerator(
                model=self.get_model(),
                api_key=Secret.from_env_var("OPENAI_API_KEY"),
                api_base_url=getattr(settings, "OPENAI_API_URL", None),
            ),
            tools=[],
            system_prompt=self.get_system_prompt(),
            exit_conditions=["text"],
        )
        pipeline.add_component("pirate_agent", pirate_agent)

        return HaystackAdapter(
            pipeline=pipeline,
            generator_component=pirate_agent.chat_generator,
            storage_adapter=storage_adapter,
        )
```

## Instructions Format

Instructions can be a string or a list. Lists are joined with newlines, which makes multi-line prompts easier to read:

```python
# List format -- recommended for readability
instructions = [
    "You are a helpful assistant.",
    "Always be polite and professional.",
    "",
    "When asked about weather, use the weather tool.",
]

# String format -- fine for short prompts
instructions = "You are a helpful assistant."
```

You can access the resolved prompt string via `self.get_system_prompt()` (or `self.get_instructions()`).

## Helper Methods

The `Assistant` base class gives you these for free:

- `get_name()` -- returns `self.name`
- `get_model()` -- returns `self.model`
- `get_system_prompt()` -- returns instructions as a single string
- `get_tools()` -- returns `[]` by default, override to provide tools
- `info()` -- returns a metadata dict about the assistant (useful for listing available assistants to the frontend)

## Choosing an Adapter

The adapter you return from `get_pipeline_adapter()` determines how your assistant talks to the AI backend:

| Adapter | When to use it |
|---------|---------------|
| `OpenAIAdapter` | Simple chat, no tools needed |
| `OpenAIAgentAdapter` | Need function calling / tools via the `agents` library |
| `HaystackAdapter` | Need Haystack pipelines, multiple providers, or complex workflows |

All adapters produce the same normalized streaming events, so you can swap them without changing your views or frontend.

See [Tools and Agents](tools-and-agents) for adding tools to your assistants.

## RAG (Retrieval-Augmented Generation)

To give your assistant access to a knowledge base, use the `rag_provider` class variable:

```python
from django_ai_sdk import Assistant
from django_ai_sdk.rags import BaseRAGProvider, BM25RAG

class MyRAGAssistant(Assistant):
    name = "Knowledgeable Bot"
    model = "gpt-4"
    instructions = ["You are a helpful assistant with access to a knowledge base."]
    rag_provider = BaseRAGProvider()
    
    async def get_rag_pipeline(self):
        # Return your RAG pipeline
        documents = await self.get_rag_documents()
        return BM25RAG(documents=documents)
```

The assistant will automatically warm up the RAG index on first use. See the [Developer Manual](/docs/manual/rag/) for complete RAG documentation.

## Retrieving Conversation History

The `history()` method retrieves a thread's messages in protocol format:

```python
from django_ai_sdk.assistants.registry import registry

@router.get("/threads/{thread_id}")
async def get_thread_history(request, thread_id: str):
    assistant = registry.get(assistant_id)
    if not assistant:
        return {"error": "Assistant not found"}, 404
    
    # Returns ThreadDetail with thread info and protocol-formatted messages
    thread_detail = await assistant.history(thread_id)
    
    return {
        "thread": {
            "id": str(thread_detail.thread.id),
            "title": thread_detail.thread.title,
            "created_at": thread_detail.thread.created_at.isoformat(),
        },
        "messages": thread_detail.messages,
    }
```

The returned messages are already formatted for the Vercel AI SDK protocol, ready to be loaded into your frontend's chat interface.

## RAG Management

### Manual Warmup

Trigger RAG warmup explicitly:

```python
# Warmup RAG for default silo
await assistant.warmup()

# Warmup for specific silo
await assistant.warmup(silo_id="my-silo-id")
```

### Clear RAG Cache

After documents change, clear the cache to force rebuilding:

```python
assistant.clear_rag_cache()
```

### Reindex RAG

Completely rebuild the RAG index:

```python
# Reindex default silo
await assistant.reindex()

# Reindex specific silo
await assistant.reindex(silo_id="my-silo-id")
```

This is useful when:
- Documents have been updated
- You want to refresh the search index
- RAG results seem stale

**Note:** Reindexing can be slow for large document collections. Consider running it in a background task.
