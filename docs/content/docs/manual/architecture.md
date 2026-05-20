---
title: Architecture Guide
type: docs
weight: 101
---

Deep dive into the core abstractions and design patterns.

## Table of Contents

1. [Assistant (The Coordinator)](#assistant-the-coordinator)
2. [Assistant Registry](#assistant-registry)
3. [Adapter (The AI Connector)](#adapter-the-ai-connector)
4. [Protocol Handler (The Format Converter)](#protocol-handler-the-format-converter)
5. [Storage (The Persistence Layer)](#storage-the-persistence-layer)
6. [RAG (The Knowledge Layer)](#rag-the-knowledge-layer)
7. [Data Flow](#data-flow)
8. [Design Patterns](#design-patterns)

---

## Assistant (The Coordinator)

The Assistant is the central class that orchestrates all components.

### Responsibilities

- Defines AI personality (name, model, instructions)
- Configures protocol handler and storage adapter
- Implements abstract methods for pipeline creation
- Manages conversation flow

### Configuration (Class Variables)

| Variable | Type | Description | Example |
|----------|------|-------------|---------|
| `name` | `str` | Assistant display name | `"Pirate Assistant"` |
| `model` | `str` | AI model identifier | `"gpt-4o-mini"` |
| `instructions` | `list[str]` | System instructions | `["You are a pirate..."]` |
| `protocol` | Protocol class | Protocol handler | `VercelProtocolHandler` |
| `storage_adapter` | Storage class | Storage adapter class | `MemoryStorageAdapter` |
| `rag_provider` | BaseRAGProvider | RAG provider instance | `BaseRAGProvider()` |

### Key Methods

#### `get_pipeline_adapter(thread_id: str | None)` [ABSTRACT]

**Must be implemented.** Returns an adapter for the specific AI backend.

```python
async def get_pipeline_adapter(self, thread_id=None):
    storage = await self.get_storage_adapter(thread_id)
    
    return OpenAIAdapter(
        client=AsyncOpenAI(api_key=settings.OPENAI_API_KEY),
        model=self.model,
        instructions=self.get_instructions(),
        store=True,
        storage_adapter=storage,
    )
```

#### `get_storage_adapter(thread_id: str | None)`

Returns a storage adapter bound to the thread. Searches all registered adapters.

```python
storage = await assistant.get_storage_adapter(thread_id)
# Returns: MemoryStorageAdapter or DbStorageAdapter
```

#### `as_view(protocol_messages, thread_id=None)`

Main entry point. Converts protocol messages, stores user message, streams response.

```python
response = await assistant.as_view(
    protocol_messages=[{"role": "user", "content": "Hello!"}],
    thread_id="thread-uuid"
)
# Returns: AsyncGenerator[bytes, None] (SSE stream)
```

### Complete Example

```python
from django_ai_sdk import Assistant
from django_ai_sdk.adapters.openai import OpenAIAdapter
from django_ai_sdk.protocols.vercel import VercelProtocolHandler
from django_ai_sdk.storage.memory import MemoryStorageAdapter
from openai import AsyncOpenAI

class PirateAssistant(Assistant):
    """A pirate-themed AI assistant."""
    
    name = "Captain Blackbeard Bot"
    model = "gpt-4o-mini"
    instructions = [
        "You are Captain Blackbeard, a swashbuckling pirate!",
        "Use pirate language like 'Arr!', 'Ahoy!', 'Shiver me timbers!'",
        "Always sign off with pirate emojis 🏴‍☠️",
    ]
    protocol = VercelProtocolHandler
    storage_adapter = MemoryStorageAdapter
    
    async def get_pipeline_adapter(self, thread_id=None):
        """Create OpenAI adapter for this assistant."""
        storage_adapter = await self.get_storage_adapter(thread_id)
        
        return OpenAIAdapter(
            client=AsyncOpenAI(),
            model=self.model,
            instructions=self.get_instructions(),
            store=True,
            storage_adapter=storage_adapter,
        )
```

---

## Assistant Registry

The Assistant Registry manages assistant registration and provides centralized access to all assistants in your application.

### Registration Methods

**Method 1: Settings-based (Recommended)**

Define `AI_SDK_ASSISTANTS` in your `settings.py`:

```python
# settings.py
AI_SDK_ASSISTANTS = [
    "myapp.assistants.PirateAssistant",
    "myapp.assistants.ShakespeareAssistant",
]
```

**Method 2: Decorator-based**

Use the `@auto_register` decorator:

```python
from django_ai_sdk.assistants import auto_register

@auto_register
class PirateAssistant(Assistant):
    name = "Pirate Bot"
    # ... configuration
```

**Both methods work together** — a class is registered only once, regardless of which method(s) you use.

### Stable Assistant IDs

Each assistant receives a **deterministic UUID v5 ID** generated from `module.ClassName`:

```python
# ID is always the same for the same class
assistant_id = str(uuid.uuid5(NAMESPACE, "myapp.assistants.PirateAssistant"))
```

This ensures:
- **Consistency**: Same ID across restarts and deployments
- **Predictability**: You can reference assistants by ID in your code
- **No conflicts**: Different classes get different IDs automatically

### Using the Registry

```python
from django_ai_sdk.assistants.registry import registry

# Get assistant by ID
assistant = registry.get("db9540d3-37ef-5c7a-83be-70f1798994f1")

# List all registered assistants
for assistant_id in registry.ids():
    print(f"Registered: {assistant_id}")

# Check if assistant exists
if "db9540d3-37ef-5c7a-83be-70f1798994f1" in registry:
    assistant = registry.get(assistant_id)
```

### Initialization

The registry is automatically initialized when Django starts. In your `AppConfig`:

```python
# apps.py
from django.apps import AppConfig

class MyAppConfig(AppConfig):
    def ready(self):
        from django_ai_sdk.assistants.registry import registry
        registry.setup()  # Loads from AI_SDK_ASSISTANTS and instantiates
```

---

## Adapter (The AI Connector)

Adapters connect to specific AI backends (OpenAI, Haystack, etc.) and normalize their outputs.

### Available Adapters

| Adapter | Backend | Use Case |
|---------|---------|----------|
| `OpenAIAdapter` | OpenAI SDK | OpenAI GPT models |
| `HaystackAdapter` | Haystack AI | Custom pipelines |
| `OpenAIAgentAdapter` | OpenAI Agents | Agent-based assistants |

### Key Responsibilities

1. **ID Generation**: Creates UUID for each message (single source of truth)
2. **Streaming**: Handles async streaming from provider
3. **Event Normalization**: Converts provider-specific to `StreamEvent` format
4. **RAG Integration**: Injects retrieved context into prompts
5. **Storage Callback**: Passes completed messages to StreamWriter

### ID Generation Flow

```
Adapter Level (stream() method)
    ↓
message_id = str(uuid.uuid4())  # Generate once
    ↓
yield MessageStartEvent(message_id)  # Used in SSE
    ↓
StreamWriter(message_id=message_id)  # Used for storage
    ↓
ChatMessage(id=message_id)  # Message object
    ↓
Storage.save()  # Same ID in database
```

This ensures the same ID flows from generation → SSE → storage → API endpoints.

### Event Flow

```python
async for event in adapter.stream(messages):
    # Events emitted:
    # - MessageStartEvent(message_id)
    # - TextChunkEvent(content)
    # - ReasoningChunkEvent(content)  # For o1/o3-mini
    # - ToolCallStartEvent(tool_call_id, tool_name)
    # - ToolInputCompleteEvent(tool_input)
    # - ToolOutputEvent(tool_output)
    # - MessageEndEvent(finish_reason)
    # - ErrorEvent(error_message)
    # - StreamEndEvent()
```

### Example: OpenAIAdapter with RAG

```python
from django_ai_sdk.adapters.openai import OpenAIAdapter
from django_ai_sdk.rags import BM25RAG

adapter = OpenAIAdapter(
    client=AsyncOpenAI(),
    model="gpt-4o-mini",
    instructions="You are a helpful assistant.",
    store=True,
    storage_adapter=storage,
    rag_pipeline=BM25RAG(documents=docs),  # RAG context injection
)
```

---

## Protocol Handler (The Format Converter)

Protocol handlers convert internal events to external streaming formats.

### Default: VercelProtocolHandler

Converts events to **Vercel AI SDK Data Stream Protocol** format, used by the frontend.

### Protocol Parts

| Part | Description | Example |
|------|-------------|---------|
| `MessageStartPart` | Start of message | `{"type":"start","messageId":"msg-123"}` |
| `TextStartPart` | Text block begins | `{"type":"text-start","id":"text-456"}` |
| `TextDeltaPart` | Text content | `{"type":"text-delta","id":"text-456","delta":"Hello"}` |
| `TextEndPart` | Text block ends | `{"type":"text-end","id":"text-456"}` |
| `ReasoningStartPart` | Reasoning begins | For o1/o3-mini models |
| `ReasoningDeltaPart` | Reasoning content | Model's thinking process |
| `ToolInputStartPart` | Tool call starts | `{"type":"tool-input-start",...}` |
| `DataPart` | Custom data | `{"type":"data-rag-retrieval",...}` |
| `ErrorPart` | Error occurred | `{"type":"error","errorText":"..."}` |
| `FinishPart` | Message complete | `{"type":"finish","finishReason":"stop"}` |

### SSE Format

```
data: {"type":"start","messageId":"msg-123"}

data: {"type":"text-start","id":"text-456"}

data: {"type":"text-delta","id":"text-456","delta":"Hello"}

data: {"type":"text-end","id":"text-456"}

data: {"type":"finish","finishReason":"stop"}

data: [DONE]
```

### Usage in Assistant

```python
class MyAssistant(Assistant):
    protocol = VercelProtocolHandler  # Class, not instance
```

The Assistant's `__init__` will call `self.protocol()` to instantiate it.

---

## Storage (The Persistence Layer)

Storage adapters persist conversation history.

### Available Adapters

| Adapter | Persistence | Speed | Use Case |
|---------|-------------|-------|----------|
| `MemoryStorageAdapter` | In-memory | Fastest | Testing, development |
| `DbStorageAdapter` | Django ORM | Slower | Production, long-term |

### Universal Storage via ChatMessage

All storage adapters use the same `ChatMessage` JSON format:

```python
{
    "id": "msg-uuid",
    "role": "assistant",
    "content": "Hello!",
    "model": "gpt-4o-mini",
    "finish_reason": "stop",
    "tool_calls": [],
    "sources": [],  # RAG sources
    ...
}
```

This means you can switch from Memory → Database without changing your code.

### Storage Methods

```python
# Instance methods (thread-specific)
storage = MemoryStorageAdapter(thread_id)
await storage.store_chat_message(chat_message)  # Save message
history = await storage.get_history()  # Get all messages
await storage.rate_message(msg_id, rating=1)  # Rate message
await storage.delete_message(msg_id)  # Soft delete

# Class methods (global thread management)
thread = await MemoryStorageAdapter.get_thread(thread_id)
threads = await MemoryStorageAdapter.list_threads(user_id)
```

### ID Consistency

```
User Request
    ↓
Adapter generates: msg_id = "550e8400-e29b-41d4-a716-446655440000"
    ↓
SSE Event: {"messageId": "550e8400-e29b-41d4-a716-446655440000"}
    ↓
Storage: Message(id="550e8400-e29b-41d4-a716-446655440000")
    ↓
API: GET /threads/{thread_id}/messages/550e8400-e29b-41d4-a716-446655440000/
```

Same ID everywhere! This is crucial for:
- Frontend message tracking
- API endpoints
- Message rating
- Message deletion

---

## RAG (The Knowledge Layer)

Retrieval-Augmented Generation adds knowledge to your assistant.

### Provider Pattern

```python
class MyAssistant(Assistant):
    rag_provider = BaseRAGProvider()  # Caches RAG instances
    
    async def get_rag_pipeline(self, silo_id=None):
        documents = await self.get_rag_documents(silo_id)
        return BM25RAG(documents=documents)
```

**Provider responsibilities:**
- Cache RAG instances per assistant + silo
- Call `warmup()` to build indexes (expensive, done once)
- Provide `get_rag_instance()` for adapters

### RAG Types

**BM25RAG** - Keyword search (lightweight, no GPU)
```python
from django_ai_sdk.rags import BM25RAG, BM25Config

rag = BM25RAG(
    documents=documents,
    config=BM25Config(top_k=5, k1=1.5, b=0.75)
)
```

**Haystack RAGs** - Vector search (Qdrant, ChromaDB)
```python
from django_ai_sdk.rags.haystack import QdrantBM25HybridRAG

rag = QdrantBM25HybridRAG(documents=documents)
```

### Integration Modes

**Mode 1: Context Injection** (OpenAIAdapter)
```python
# Adapter retrieves documents and injects into system message
adapter = OpenAIAdapter(
    rag_pipeline=rag,  # Automatically retrieves on each request
    ...
)
```

**Mode 2: Tool Calling** (Haystack, Agents)
```python
# RAG exposed as callable tool
tool = await rag_provider.build_tool(rag_instance)
# AI decides when to call: search_documents(query="pirate code")
```

---

## Data Flow

Complete request lifecycle:

```
┌─────────────────────────────────────────────────────────────┐
│ 1. USER REQUEST                                               │
│    POST /api/chat with messages                               │
└──────────────────────┬────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. ASSISTANT LAYER                                          │
│    assistant.as_view(protocol_messages)                     │
│    ├─ Convert protocol → ChatMessage                        │
│    ├─ Store last user message                               │
│    └─ Get pipeline adapter                                  │
└──────────────────────┬────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. ADAPTER LAYER                                            │
│    adapter.stream(chat_messages)                             │
│    ├─ Generate UUID for message                              │
│    ├─ RAG: retrieve(query) → inject context                │
│    └─ Call AI provider (OpenAI/Haystack)                     │
└──────────────────────┬────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. AI PROVIDER RESPONSE                                     │
│    Streaming chunks:                                        │
│    "Hello", "world", "!"                                    │
└──────────────────────┬────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. EVENT NORMALIZATION                                      │
│    Chunks → StreamEvents:                                   │
│    TextChunkEvent, ToolCallStartEvent, etc.                 │
└──────────────────────┬────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. PROTOCOL CONVERSION                                      │
│    Events → Vercel Protocol Parts                           │
│    SSE format for frontend                                  │
└──────────────────────┬────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 7. STORAGE                                                  │
│    StreamWriter → ChatMessage → Storage                     │
│    Same UUID used everywhere!                               │
└─────────────────────────────────────────────────────────────┘
```

![Data Flow](/images/graphs/data_flow.png)

---

## Design Patterns

### 1. Protocol-Agnostic Messages

Internal `ChatMessage` format works with any frontend protocol:

```python
# Same ChatMessage works with:
# - VercelProtocolHandler (default)
# - Custom protocol handlers
# - Direct API access

chat_message = ChatMessage(role="assistant", content="Hello!")
```

### 2. Event-Driven Streaming

Normalized events decouple adapters from protocols:

```python
# Adapter emits events (provider-agnostic)
yield TextChunkEvent(content="Hello")

# Protocol handler converts to specific format
# Vercel → SSE
# Custom → WebSocket
# Internal → Direct method calls
```

### 3. Single ID Source

Adapter generates UUID once, ensuring consistency:

```python
# In adapter.stream():
message_id = str(uuid.uuid4())  # Generate
yield MessageStartEvent(message_id=message_id)  # SSE
stream_writer = StreamWriter(message_id=message_id)  # Storage
```

### 4. Storage Adapter Registry

Automatic storage detection across registered adapters:

```python
# Assistant finds where thread exists:
for adapter_class in StorageAdapterRegistry.get_all_adapters():
    thread = await adapter_class.get_thread(thread_id)
    if thread:
        return adapter_class(thread_id)  # Use existing storage
```

### 5. RAG Provider Caching

Expensive index building happens once, cached per assistant:

```python
# Provider cache key: "AssistantName_silo123"
cache = {"PirateAssistant_default": rag_instance}

# warmup() builds index once
# retrieve() uses cached index
```

---

## Next Steps

- See [RAG Guide](rag/) for detailed RAG documentation
- Check [Adapters](adapters/) for backend-specific setup
- Review [Storage](storage/) for persistence patterns
- Check [Testing](testing/) for test examples
