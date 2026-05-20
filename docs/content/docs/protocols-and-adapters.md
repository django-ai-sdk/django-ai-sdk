---
title: Protocols and Adapters
type: docs
prev: views-and-routing
next: how-it-works
weight: 5
---

Two extension points that keep your code clean: **Protocols** handle message formats, **Adapters** handle AI backends.

---

## Protocols

Protocol handlers convert between client formats and internal `ChatMessage` objects.

### Vercel AI SDK Protocol (Default)

Works out of the box with `@ai-sdk/react`:

```python
from django_ai_sdk.protocols.vercel import VercelProtocolHandler

class MyAssistant(Assistant):
    protocol = VercelProtocolHandler
```

This is the default—you don't need to set it unless you're using something else.

### Custom Protocols

For non-standard clients:

```python
from django_ai_sdk.protocols.base import BaseProtocolHandler

class MyProtocolHandler(BaseProtocolHandler):
    def to_chat_messages(self, protocol_messages):
        return [
            ChatMessage(role=msg["role"], content=msg["content"])
            for msg in protocol_messages
        ]

    async def sse(self, adapter, messages):
        async for event in adapter.stream(messages):
            yield f"data: {event.content}\n\n".encode()
        yield b"data: [DONE]\n\n"
```

---

## Adapters

Adapters connect to AI backends. All produce the same events, so backends are interchangeable.

### OpenAI Adapter

```python
from django_ai_sdk.adapters.openai import OpenAIAdapter

async def get_pipeline_adapter(self, thread_id=None):
    storage = await self.get_storage_adapter(thread_id)
    return OpenAIAdapter(
        client=AsyncOpenAI(api_key=settings.OPENAI_API_KEY),
        model=self.model,
        store=True,
        storage_adapter=storage,
    )
```

Options:
- `client` — `AsyncOpenAI` instance
- `model` — model identifier  
- `store` — auto-save messages (default `True`)
- `storage_adapter` — where to persist messages
- `instructions` — optional system prompt

### OpenAI Agent Adapter

For function calling with the `agents` library:

```python
from django_ai_sdk.adapters.openai import OpenAIAgentAdapter

async def get_pipeline_adapter(self, thread_id=None):
    storage = await self.get_storage_adapter(thread_id)
    agent = Agent(
        name=self.name,
        model=self.get_model(),
        instructions=self.get_instructions(),
        tools=self.get_tools(),
    )
    return OpenAIAgentAdapter(agent=agent, storage_adapter=storage)
```

See [Tools and Agents](tools-and-agents) for complete examples.

### Haystack Adapter

For pipelines, RAG, or non-OpenAI providers:

```python
from django_ai_sdk.adapters.haystack import HaystackAdapter

async def get_pipeline_adapter(self, thread_id=None):
    storage = await self.get_storage_adapter(thread_id)
    pipeline = Pipeline()
    # ... build pipeline ...
    return HaystackAdapter(
        pipeline=pipeline,
        generator_component=agent.chat_generator,
        storage_adapter=storage,
    )
```

### Custom Adapters

Integrate any AI backend by subclassing `BasePipelineAdapter`:

```python
from django_ai_sdk.adapters.base import BasePipelineAdapter
from django_ai_sdk.events import (
    MessageStartEvent, TextChunkEvent, MessageEndEvent, StreamEndEvent
)

class MyAdapter(BasePipelineAdapter):
    async def stream(self, messages):
        yield MessageStartEvent(message_id=str(uuid.uuid4()))
        
        async for chunk in self.backend.stream(messages):
            yield TextChunkEvent(content=chunk.text)
        
        yield MessageEndEvent(finish_reason="stop")
        yield StreamEndEvent()
```

The contract: `stream()` yields `StreamEvent` objects. The protocol handler does the rest.

---

## Backend Independence

Because all adapters produce the same events, swapping backends changes one method:

```python
async def get_pipeline_adapter(self, thread_id=None):
    storage = await self.get_storage_adapter(thread_id)
    
    if settings.USE_HAYSTACK:
        return HaystackAdapter(..., storage_adapter=storage)
    else:
        return OpenAIAdapter(..., storage_adapter=storage)
```

Views, frontend, and protocol handling stay identical.
