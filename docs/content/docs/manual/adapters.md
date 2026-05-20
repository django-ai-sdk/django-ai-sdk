---
title: Adapter Guide
type: docs
weight: 102
---

Complete guide to AI backend adapters.

## Table of Contents

1. [What are Adapters?](#what-are-adapters)
2. [Available Adapters](#available-adapters)
3. [OpenAIAdapter](#openaiadapter)
4. [HaystackAdapter](#haystackadapter)
5. [OpenAIAgentAdapter](#openaiagentadapter)
6. [ID Generation](#id-generation)
7. [Event System](#event-system)
8. [RAG Integration](#rag-integration)
9. [Examples](#examples)

---

## What are Adapters?

**Adapters** connect to specific AI backends (OpenAI, Haystack, etc.) and normalize their outputs into a common event format.

### Responsibilities

1. **Backend Connection** - Connect to AI provider APIs
2. **ID Generation** - Create UUID for each message (single source of truth)
3. **Streaming** - Handle async streaming responses
4. **Event Normalization** - Convert provider-specific to `StreamEvent`
5. **RAG Integration** - Inject retrieved context
6. **Storage Callback** - Pass completed messages to storage

### Architecture

```
┌─────────────────┐
│   Assistant     │
│ get_pipeline_adapter()
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Adapter      │
│ ├─ stream()     │ → Generate ID
│ ├─ get_messages()│ → Format for provider
│ └─ emit events  │ → StreamEvent format
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  AI Provider    │
│ OpenAI/Haystack │
└─────────────────┘
```

![Adapter Flow](/images/graphs/adapter_flow.png)

---

## Available Adapters

| Adapter | Backend | Best For |
|---------|---------|----------|
| `OpenAIAdapter` | OpenAI SDK | OpenAI GPT models, streaming |
| `HaystackAdapter` | Haystack AI | Custom pipelines, agents |
| `OpenAIAgentAdapter` | OpenAI Agents | Agent-based assistants |

---

## OpenAIAdapter

Connects to OpenAI's chat completions API.

### Constructor Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `client` | `AsyncOpenAI` | Yes | OpenAI client instance |
| `model` | `str` | No | Model name (default: "gpt-4o-mini") |
| `instructions` | `str` | No | System instructions |
| `store` | `bool` | No | Enable storage (default: True) |
| `storage_adapter` | `BaseStorageAdapter` | No | For auto-storage |
| `rag_pipeline` | `BaseRAGAdapter` | No | For context injection |

### Example

```python
from openai import AsyncOpenAI
from django_ai_sdk.adapters.openai import OpenAIAdapter

adapter = OpenAIAdapter(
    client=AsyncOpenAI(api_key="your-key"),
    model="gpt-4o-mini",
    instructions="You are a helpful assistant.",
    store=True,
    storage_adapter=MemoryStorageAdapter(thread_id),
)

# Stream response
async for event in adapter.stream(messages):
    if isinstance(event, TextChunkEvent):
        print(event.content, end="")
```

### Features

#### Message Role Alternation

OpenAI requires alternating user/assistant roles. The adapter handles this:

```python
# Input: Multiple consecutive user messages
messages = [
    ChatMessage(role="user", content="Hello"),
    ChatMessage(role="user", content="How are you?"),
]

# Adapter automatically inserts assistant message:
# user: "Hello"
# assistant: "..."  # ← Auto-inserted
# user: "How are you?"
```

#### Reasoning Content (o1, o3-mini, DeepSeek)

Supports reasoning models that show their thinking:

```python
async for event in adapter.stream(messages):
    if isinstance(event, ReasoningChunkEvent):
        print(f"Thinking: {event.content}")  # Model's reasoning
    elif isinstance(event, TextChunkEvent):
        print(event.content)  # Final answer
```

#### Tool Calling

Automatically extracts and emits tool call events:

```python
async for event in adapter.stream(messages):
    match event:
        case ToolCallStartEvent():
            print(f"Tool: {event.tool_name}")
        case ToolInputCompleteEvent():
            print(f"Input: {event.tool_input}")
        case ToolOutputEvent():
            print(f"Output: {event.tool_output}")
```

### RAG Integration

**Context Injection Mode:**

```python
from django_ai_sdk.rags import BM25RAG

adapter = OpenAIAdapter(
    client=AsyncOpenAI(),
    rag_pipeline=BM25RAG(documents=docs),  # ← Automatic retrieval
    ...
)

# On each request:
# 1. Extract last user message as query
# 2. rag.retrieve(query) → Get documents
# 3. Inject into system message
# 4. Send to OpenAI
```

---

## HaystackAdapter

Connects to Haystack AI pipelines.

### Constructor Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `pipeline` | `Pipeline` | Yes | Haystack pipeline |
| `generator_component` | `Generator` | Yes | Chat generator with streaming |
| `store` | `bool` | No | Enable storage |
| `storage_adapter` | `BaseStorageAdapter` | No | For auto-storage |
| `rag_pipeline` | `BaseRAGAdapter` | No | For consistency |

### Example

```python
from haystack import Pipeline
from haystack.components.generators import OpenAIChatGenerator
from haystack.dataclasses import StreamingChunk
from haystack.utils import Secret

from django_ai_sdk.adapters.haystack import HaystackAdapter

# Create generator with streaming
generator = OpenAIChatGenerator(
    model="gpt-4o-mini",
    api_key=Secret.from_env_var("OPENAI_API_KEY"),
    streaming_callback=callback,
)

# Build pipeline
pipeline = Pipeline()
pipeline.add_component("generator", generator)

adapter = HaystackAdapter(
    pipeline=pipeline,
    generator_component=generator,
    storage_adapter=MemoryStorageAdapter(thread_id),
)

# Stream
async for event in adapter.stream(messages):
    print(event)
```

### Features

#### Tool Agent Support

Detects and uses Haystack ToolAgent components:

```python
from haystack.components.agents import ToolAgent

# If pipeline has ToolAgent, adapter uses it directly
tool_agent = ToolAgent(
    config=ToolAgentConfig(
        model="gpt-4o-mini",
        system_prompt="You have access to tools.",
        tools=[search_tool],
    ),
    generator=generator,
)

adapter = HaystackAdapter(
    pipeline=tool_agent.pipeline(),
    generator_component=generator,
)
```

#### Async Streaming

Handles streaming via thread executor:

```python
# Haystack callbacks run in separate thread
# Adapter queues tokens for async iteration
queue = asyncio.Queue()

def callback(chunk: StreamingChunk):
    queue.put_nowait(chunk.content)

# Main thread async iteration
async for event in adapter.stream(messages):
    # Events from queue
```

#### RAG Source Extraction

Extracts RAG sources from pipeline results:

```python
# Pipeline returns documents in response
result = pipeline.run(...)

# Adapter extracts:
# - Document content
# - Scores
# - Metadata
# Emits as message sources
```

---

## OpenAIAgentAdapter

Connects to OpenAI's Agents API (beta).

### Constructor Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `agent` | `Agent` | Yes | OpenAI Agent instance |
| `runner_config` | `RunConfig` | No | Agent runner configuration |
| `store` | `bool` | No | Enable storage |
| `storage_adapter` | `BaseStorageAdapter` | No | For auto-storage |

### Example

```python
from agents import Agent, RunConfig
from agents.run import Runner

from django_ai_sdk.adapters.openai import OpenAIAgentAdapter

# Create agent
agent = Agent(
    name="Pirate Agent",
    instructions="You are a pirate. Use your tools wisely.",
    tools=[get_pirate_joke, search_treasure],
)

adapter = OpenAIAgentAdapter(
    agent=agent,
    runner_config=RunConfig(model_provider=...),
    store=True,
    storage_adapter=MemoryStorageAdapter(thread_id),
)

# Stream agent responses
async for event in adapter.stream(messages):
    print(event)
```

### Features

#### Tool Execution

Agent automatically decides when to use tools:

```python
# User: "Tell me a joke and find treasure"
# Agent decides:
# 1. Call get_pirate_joke() → "Why did the pirate..."
# 2. Call search_treasure(location="island") → "Found at..."
# 3. Combine results into response
```

#### Built-in Tools

Tools are methods on the Assistant:

```python
class PirateAgentAssistant(Assistant):
    @function_tool(name_override="pirate_joke")
    def get_pirate_joke(self) -> str:
        """Get a pirate joke."""
        return "Why did the pirate go to the Apple store?"
    
    def get_tools(self):
        return [self.get_pirate_joke]
```

---

## ID Generation

**Critical for consistency** - The adapter generates the UUID once.

### Flow

```
┌────────────────────────────────────────────────────┐
│ Adapter.stream()                                     │
│                                                      │
│ message_id = str(uuid.uuid4())  ← GENERATE          │
│                                                      │
│ yield MessageStartEvent(message_id=message_id)      │
│                           ↓                         │
│                     SSE Stream                      │
│                     {"messageId": "..."}            │
│                           ↓                         │
│ StreamWriter(message_id=message_id)                │
│                           ↓                         │
│ ChatMessage(id=message_id)                         │
│                           ↓                         │
│ Storage.save()                                       │
│ Same ID everywhere!                                 │
└────────────────────────────────────────────────────┘
```

![ID Generation Flow](/images/graphs/id_generation.png)

### Why This Matters

- **Frontend** tracks message via ID in SSE
- **Storage** retrieves message via same ID
- **API endpoints** use ID for rating, deletion
- **History** maintains consistent references

---

## Event System

Adapters emit normalized events regardless of backend.

### Event Types

| Event | When | Data |
|-------|------|------|
| `MessageStartEvent` | Stream begins | `message_id` |
| `TextChunkEvent` | Text token received | `content` |
| `ReasoningChunkEvent` | Reasoning token (o1/o3-mini) | `content` |
| `ToolCallStartEvent` | Tool call begins | `tool_call_id`, `tool_name` |
| `ToolInputChunkEvent` | Tool args incremental | `tool_input_delta` |
| `ToolInputCompleteEvent` | Tool arguments ready | `tool_input` |
| `ToolOutputEvent` | Tool execution complete | `tool_output` |
| `DataEvent` | Custom data streaming | `data` (arbitrary dict) |
| `MessageEndEvent` | Message complete | `finish_reason` |
| `ErrorEvent` | Error occurred | `error_message` |
| `StreamEndEvent` | Stream terminated | - |

**DataEvent** is used for custom structured data streaming. Most commonly used for RAG source references:

```python
# In adapter - emit RAG retrieval info
yield DataEvent(data={
    "rag_retrieval": {
        "query": user_query,
        "sources": [
            {"id": doc.id, "score": doc.score, "title": doc.title}
            for doc in retrieved_docs
        ]
    }
})
```

This allows the frontend to display source information alongside responses.

### Example: Handling Events

```python
async for event in adapter.stream(messages):
    match event:
        case MessageStartEvent():
            print(f"Message: {event.message_id}")
            
        case TextChunkEvent():
            print(event.content, end="")  # Stream to user
            
        case ReasoningChunkEvent():
            print(f" {event.content}")  # Show reasoning
            
        case ToolCallStartEvent():
            print(f" Using: {event.tool_name}")
            
        case ToolInputCompleteEvent():
            print(f" Input: {event.tool_input}")
            
        case ToolOutputEvent():
            print(f" Output: {event.tool_output}")
            
        case MessageEndEvent():
            print(f"Yes Done: {event.finish_reason}")
            
        case ErrorEvent():
            print(f"No Error: {event.error_message}")
            
        case StreamEndEvent():
            print(" Stream ended")
```

---

## RAG Integration

### Context Injection (OpenAI)

```python
adapter = OpenAIAdapter(
    client=AsyncOpenAI(),
    rag_pipeline=BM25RAG(documents=docs),
    ...
)

# Behind the scenes:
# 1. User: "What is the pirate code?"
# 2. Query = "What is the pirate code?"
# 3. docs = rag.retrieve(query) → [doc1, doc2, doc3]
# 4. context = format_context(docs)
# 5. messages[0]["content"] = f"{context}\n\n{system_msg}"
# 6. Send to OpenAI
```

### Query Expansion

For better RAG results, the SDK supports **query expansion** — generating multiple query variations from the user's input:

```
User: "What is the pirate code?"
       ↓
Query Expansion (via OpenAI)
├─ "What is the pirate code?"
├─ "Explain the pirate code rules"
├─ "Pirate code of conduct"
└─ "Pirate laws and regulations"
       ↓
Search All Variations → Merge Results
```

**Why it helps:** Users often don't use exact keywords from documents. Query expansion increases recall without requiring query reformulation.

**Implementation:**

```python
# All Haystack RAGs use query expansion internally
from django_ai_sdk.rags.haystack import QdrantBM25HybridRAG

rag = QdrantBM25HybridRAG(
    documents=documents,
    config=QdrantBM25HybridRAGConfig(
        top_k=5,
        n_expansions=4,  # Generate 4 query variations
    )
)

# On retrieve(), the RAG:
# 1. Takes user query
# 2. Generates n_expansions variations via OpenAI
# 3. Searches with all variations
# 4. Merges and deduplicates results
# 5. Returns top_k documents
```

**Configuration options:**
- `n_expansions` — Number of query variations (default: 4)
- Expansion maintains language consistency (forces same language as input)

### Tool Calling (Haystack)

```python
adapter = HaystackAdapter(
    pipeline=tool_agent.pipeline(),
    generator_component=generator,
    ...
)

# Tool agent decides when to search:
# 1. User: "What is the pirate code?"
# 2. Agent: "I should search for this"
# 3. Calls: search_documents(query="pirate code")
# 4. RAG retrieves documents
# 5. Agent generates response
```

---

## Examples

### Complete OpenAI Example

```python
from django.conf import settings
from openai import AsyncOpenAI

from django_ai_sdk import Assistant
from django_ai_sdk.adapters.openai import OpenAIAdapter
from django_ai_sdk.protocols.vercel import VercelProtocolHandler
from django_ai_sdk.storage.memory import MemoryStorageAdapter
from django_ai_sdk.rags import BM25RAG, BM25Config

class OpenAIAssistant(Assistant):
    """OpenAI assistant with RAG."""
    
    name = "OpenAI Helper"
    model = settings.AI_SDK_DEFAULT_MODEL
    instructions = [
        "You are a helpful assistant.",
        "Use the provided context to answer accurately.",
    ]
    protocol = VercelProtocolHandler
    storage_adapter = MemoryStorageAdapter
    
    async def get_pipeline_adapter(self, thread_id=None):
        storage = await self.get_storage_adapter(thread_id)
        
        # Get RAG if available
        rag = None
        if self.rag_provider:
            rag = await self.rag_provider.get_rag_instance(self, None)
        
        return OpenAIAdapter(
            client=AsyncOpenAI(api_key=settings.OPENAI_API_KEY),
            model=self.model,
            instructions=self.get_instructions(),
            store=True,
            storage_adapter=storage,
            rag_pipeline=rag,
        )
```

### Complete Haystack Example

```python
from haystack import Pipeline
from haystack.components.generators import OpenAIChatGenerator
from haystack.dataclasses import StreamingChunk
from haystack.utils import Secret

from django_ai_sdk import Assistant
from django_ai_sdk.adapters.haystack import HaystackAdapter
from django_ai_sdk.protocols.vercel import VercelProtocolHandler
from django_ai_sdk.storage.db import DbStorageAdapter
from django_ai_sdk.rags.haystack import HaystackRAGProvider

class HaystackAssistant(Assistant):
    """Haystack assistant with tools."""
    
    name = "Haystack Helper"
    model = "gpt-4o-mini"
    instructions = ["You are a helpful assistant with access to tools."]
    protocol = VercelProtocolHandler
    storage_adapter = DbStorageAdapter
    rag_provider = HaystackRAGProvider()
    
    async def get_pipeline_adapter(self, thread_id=None):
        storage = await self.get_storage_adapter(thread_id)
        
        # Get RAG and build tool
        rag = await self.rag_provider.get_rag_instance(self, None)
        generator = OpenAIChatGenerator(
            model=self.model,
            api_key=Secret.from_env_var("OPENAI_API_KEY"),
        )
        search_tool = await self.rag_provider.build_tool(rag, generator)
        
        # Create tool agent
        from haystack.components.agents import ToolAgent, ToolAgentConfig
        tool_agent = ToolAgent(
            config=ToolAgentConfig(
                model=self.model,
                system_prompt=self.get_system_prompt(),
                tools=[search_tool],
            ),
            generator=generator,
        )
        
        return HaystackAdapter(
            pipeline=tool_agent.pipeline(),
            generator_component=generator,
            storage_adapter=storage,
        )
```

---

## Next Steps

- See [Architecture Guide](architecture/) for core concepts
- Check [RAG Guide](rag/) for knowledge retrieval
- Review [Storage](storage/) for persistence patterns
- Check [Testing](testing/) for test examples
