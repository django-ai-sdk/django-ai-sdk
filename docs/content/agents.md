---
title: Agents
type: docs
weight: 2
---

An **agent** is the core of the Django AI SDK. It's a plain Python class that describes how your AI behaves, what it can do, and how it streams responses.

This guide covers everything about defining agents. See the [Protocols](/protocols/) guide for how `Stream` and `Run` work, and [Views and Routing](/views-and-routing/) for wiring agents to HTTP endpoints.

{{< callout type="info" >}}
Contributor? The [Developer Manual](/manual/) covers internals: [Agent](/manual/agent/), [Agent Registry](/manual/agent-registry/), [RAG](/manual/rag/).
{{< /callout >}}

## Anatomy of an Agent

```python
from django_ai_sdk import Agent
from django_ai_sdk.adapters.base import Run, Stream
from django_ai_sdk.agents import auto_register
from django_ai_sdk.common import prompt
from django_ai_sdk.protocols.vercel import VercelProtocolHandler
from django_ai_sdk.storage.db import DbStorageAdapter

@auto_register
class MyAgent(Agent):
    name = "My Bot"
    description = "A helpful assistant"
    model = "openai/gpt-oss-120b"
    instructions = prompt("You are a helpful agent.")
    protocol = VercelProtocolHandler
    storage_adapter = DbStorageAdapter
    max_history = 20
    tools = [get_today, get_memory_files]

    async def get_pipeline_adapter(self, thread_id=None, user=None):
        # Build the Stream that powers streaming chat.
        ...

    async def get_run_adapter(self, thread_id=None, user=None):
        # Build the Run that powers non-streaming calls.
        ...
```

### Registration

Every `Agent` subclass registers itself in the registry when the module is imported. Use `AI_SDK_AGENTS` in settings to ensure your modules are loaded:

```python
# settings.py
AI_SDK_AGENTS = [
    "your_app.agents.MyAgent",
    "your_app.agents.OtherAgent",
]
```

Each agent gets a **stable UUID** derived from its module and class name, so it never changes between restarts. Resolve agents by ID with `AgentService`:

```python
from django_ai_sdk.agents.services import AgentService

agent = await AgentService.get(agent_id)
```

Mark a class `abstract = True` to use it as a shared base without registering it, or `hidden = True` to keep it out of agent listings (useful for worker-only agents).

---

## Class Configuration

These class attributes configure an agent's behavior:

| Attribute | Default | Purpose |
| --- | --- | --- |
| `name` | None | Display name |
| `description` | `None` | Short description, surfaced in agent listings |
| `model` | `None` | Model identifier passed to the generator |
| `llm` | `openai_responses_chat` | Generator factory from [`django_ai_sdk.generators`](/manual/generators/), assigned uncalled |
| `llm_kwargs` | `None` | Vendor generation parameters, e.g. `{"reasoning": {"effort": "low"}}` |
| `instructions` | `"You are a helpful agent."` | System prompt, built with `prompt()` |
| `protocol` | `VercelProtocolHandler` | Protocol handler for message conversion |
| `storage_adapter` | `MemoryStorageAdapter` | Storage used when a thread isn't found elsewhere |
| `tools` | `[]` | Class-level tool providers |
| `integrations` | `[]` | Integration names whose tools reach this agent |
| `artifacts` | `[]` | `ArtifactSchema` subclasses exposed as submission tools |
| `rag_provider` | `None` | `RAGProvider` instance to enable RAG |
| `rag_document_limit` | `10000` | Intended cap on documents fetched for RAG indexing (not yet enforced) |
| `memories` | `[]` | Default connected memories |
| `permissions` | `None` | Permission classes gating agent operations; `None` or `[]` uses the `agent` domain default |
| `max_history` | `None` | Cap on messages sent to the model |
| `title_generation` | `True` | Auto-generate thread titles |
| `response_format` | `None` | Pydantic model for structured `run()` output |
| `file_upload` | `False` | Enable file upload for threads |
| `file_pipelines` | `[]` | Per-file-type processing pipelines |
| `citation_formatter_class` | `DefaultCitationFormatter` | How RAG sources are rendered to the model |
| `suggestion_generator` | `None` | Generates follow-up suggestions after a reply |
| `hidden` | `False` | Hide from registry listings |
| `abstract` | `False` | Shared base, not registered |
| `warmup_on_init` | `False` | Warm up RAG on agent initialization |

### The `prompt()` helper

Use `prompt()` to build instructions. It wraps your text in a `Prompt` and strips leading indentation, so multi-line instruction blocks stay clean:

```python
instructions = prompt("""\
    You are a helpful AI agent who always responds like a pirate.
    Use pirate language, expressions, and mannerisms in all your responses.
    - Be creative with pirate slang but keep responses helpful and informative.
""")
```

---

## Adapters: Stream and Run

An agent exposes two adapter hooks. You implement them; the SDK uses them.

- **`get_pipeline_adapter(thread_id, user)`**: returns a `Stream` for streaming chat. `Agent.as_view()` calls this.
- **`get_run_adapter(thread_id, user)`**: returns a `Run` for non-streaming calls. `Agent.run()` calls this (title generation, structured extraction, background jobs).

{{< callout type="info" >}}
A worker-only agent (`hidden = True`, never used in chat) can leave `get_pipeline_adapter()` unimplemented.
{{< /callout >}}

See the [Protocols guide](/protocols/) for the full `Stream` and `Run` API.

---

## Tools

Tools let the model call your Python functions. The SDK expects **Haystack `Tool` objects**: the model's tool schema is derived from them.

### Class-level tools

`Agent.get_tools()` runs every callable in the class-level `tools` list, passing context kwargs. Each provider may return one tool or a list:

```python
# your_app/tools.py
from haystack.tools import Tool

def get_datetime() -> dict:
    """Return the current date and time."""
    ...

def get_today(**kwargs) -> Tool:
    """Current date and time tool."""
    return Tool(
        name="get_today",
        parameters={},
        description="Get current date and time",
        function=get_datetime,
    )
```

```python
# your_app/agents.py
class MyAgent(Agent):
    ...
    tools = [get_today, get_memory_files]
```

Providers receive context about the current request:

```python
def get_memory_files(thread_id="", user_id="", **kwargs) -> Tool:
    """List files attached to the current thread."""
```

### Overriding `get_tools()`

For full control, override `get_tools()` on your agent. The default implementation combines:

1. Class-level `tools` providers
2. Integration tools (`integrations` list, resolved via the integrations registry)
3. Artifact submission tools (`artifacts` list)

```python
class MyAgent(Agent):
    async def get_tools(self, thread_id="", user=None):
        tools = await super().get_tools(thread_id=thread_id, user=user)
        tools.append(await self.get_custom_tool(thread_id))
        return tools
```

### Integration tools

Declare `integrations = ["linear", "weather"]` and every tool that integration exposes reaches the model. Tools are namespaced (`linear_list_issues`) so unrelated integrations never collide. Unauthorized users' tools never reach the model. See the [Integrations guide](/integrations/).

### Artifact tools

Agents can submit structured results as **artifacts**. Each `ArtifactSchema` subclass in the `artifacts` list becomes a submission tool the model can call during a turn:

```python
class MyArtifact(ArtifactSchema):
    kind = "report"
    title: str
    content: str
```

```python
class MyAgent(Agent):
    artifacts = [MyArtifact]
```

---

## Retrieval-Augmented Generation (RAG)

RAG connects documents stored in **memories** to your agent. The `RAGProvider` caches pipelines per agent and memory and exposes a per-memory retrieval tool for the model to call.

### Enabling RAG

1. Set `rag_provider = RAGProvider()` on the agent.
2. Override `get_rag_queryset()` to return the documents to index.
3. Implement `get_rag_pipeline()` (or one of the named variants) to return a RAG pipeline.
4. Link memories to threads: `get_rag_tools()` builds one tool per active memory.

```python
from django_ai_sdk.memories.models import Entry
from django_ai_sdk.rags import (
    BM25QueryExpanderRAG,
    BM25QueryExpanderRAGConfig,
    QdrantBM25HybridRAG,
    QdrantBM25HybridRAGConfig,
)
from django_ai_sdk.rags.provider import RAGProvider

class MyAgent(Agent):
    rag_provider = RAGProvider()

    async def get_rag_queryset(self, memory_id=None):
        if memory_id:
            return Entry.objects.filter(memory_id=memory_id)
        return Entry.objects.all()

    async def get_rag_pipeline(self, memory_id=None):
        documents = await self.get_rag_documents(memory_id)
        if not documents:
            return None
        return BM25QueryExpanderRAG(
            documents=documents,
            config=BM25QueryExpanderRAGConfig(top_k=5, n_expansions=4),
        )
```

### Pipeline variants

| Variant | Retrieval |
| --- | --- |
| `BM25QueryExpanderRAG` | BM25 keyword retrieval with query expansion |
| `ChromaDBQueryExpanderRAG` | Dense embedding retrieval with ChromaDB |
| `QdrantBM25HybridRAG` | Hybrid SPLADE + BGE embeddings with RRF, stored in Qdrant |

Override `get_rag_pipeline()` to return one of the variants. You can factor each into its own method (e.g. `get_rag_pipeline_qdrant()`) and call it from `get_rag_pipeline()`; the base method returns `None` (RAG disabled).

### Lifecycle

- **Warm up**: `Agent.warmup(agent, memory_id)` pre-builds and caches pipelines.
- **Reindex**: `Agent.reindex(agent, memory_id, force_rebuild)` rebuilds indexes (persistent backends like Qdrant delete and recreate).
- **Clear cache**: `Agent.clear_rag_cache(agent)` drops cached pipelines.
- **CLI**: `python manage.py warmup_rag --agent <name>` warms up from the command line (see the [CLI guide](/cli/)).

### Citations

Retrieved sources are streamed back as numbered citations. `get_citation_registry()` returns a fresh per-turn registry, and `citation_formatter_class` controls how sources are rendered for the model. The `DefaultCitationFormatter` numbers sources so the model can reference them inline.

---

## Storage

Conversations live in **threads** of **messages**. Two storage adapters ship with the SDK:

- `MemoryStorageAdapter`: in-memory, useful for tests and prototypes
- `DbStorageAdapter`: persists via the Django ORM

{{< callout type="info" >}}
`MemoryStorageAdapter` is in-memory only: use `DbStorageAdapter` whenever conversations must survive a restart.
{{< /callout >}}

`get_storage_adapter(thread_id)` locates which adapter actually holds a thread (querying all registered adapters) and returns the right instance, falling back to the agent's configured `storage_adapter`.

```python
class MyAgent(Agent):
    storage_adapter = DbStorageAdapter
```

`storage_adapter` is the attribute you configure (the same name is also used as the effective instance attribute in `__init__`). `Agent` additionally declares a `storage` class annotation for the same purpose: either way, the effective adapter defaults to `MemoryStorageAdapter` and comes from `storage_adapter` at runtime.

---

## Structured Output

`Agent.run()` returns a Pydantic model when a `response_format` is set:

```python
from pydantic import BaseModel

class Movie(BaseModel):
    title: str
    year: int

class MyAgent(Agent):
    response_format = Movie

    async def get_run_adapter(self, thread_id=None, user=None):
        return Run(generator=self.get_llm())

# agent.run(messages) -> Movie | str
```

Pass `response_format=None` explicitly to `run()` to disable structured output for a single call.

---

## Permissions

Each agent declares `permissions`: classes that gate operations like `CHAT`, `VIEW_THREAD`, and `USE_INTEGRATION`. The demo shows per-domain overrides:

```python
# settings.py
AI_SDK_PERMISSIONS = {
    "memory": ["apps.memories.permissions.AllowAnonymousMemoryPermission"],
    "thread": ["apps.agents.permissions.DemoThreadPermission"],
}
```

`Agent.as_view()` and `Agent.history()` check permissions before doing anything, raising `PermissionDenied` when the user lacks access. See the [Views and Routing guide](/views-and-routing/) for wiring.

---

## Files

Set `file_upload = True` to accept file uploads on threads; leaving it `False` rejects uploads to the agent's threads with a `PermissionDenied`. Each `FilePipeline` pairs a processor (e.g. `TextFileProcessor`) with optional transforms, including LLM extraction agents, that run over the uploaded file:

```python
class MyAgent(Agent):
    file_upload = True
    file_pipelines = [
        FilePipeline(
            TextFileProcessor(),
            transforms=[DocumentExtractionTransform(ExtractionAgent())],
        ),
    ]
```

See the [Files reference](/manual/files/) for the shipped processors (Text, CSV, JSON, Docx, Pptx, Xlsx), transforms, the upload/processing lifecycle, and upload settings (`AI_SDK_MAX_UPLOAD_SIZE`, `AI_SDK_ALLOWED_FILES`, `AI_SDK_MEMORY_FILE_PIPELINE`).

---

## Suggestions

Set `suggestion_generator` to stream follow-up questions after the assistant's reply:

```python
from django_ai_sdk.suggestions import DefaultSuggestionGenerator

class MyAgent(Agent):
    suggestion_generator = DefaultSuggestionGenerator
```

---

## Reference Agents

The demo project ships five agents in `demo/apps/agents/`, each showcasing a different pattern:

| Agent | Pattern |
| --- | --- |
| `PirateBasicAgent` | Full-featured agent: RAG, tools, integrations, citations, suggestions, files |
| `AgentSwarmAgent` | Multiple tool agents composing one response |
| `DefaultRuntimeAgent` | Base for DB-configured runtime agents |
| `WorkspaceAgent` | Artifact-based workspace workflows |
| `PirateExtractionAgent` | Structured extraction used inside file pipelines |

Browse them in the repository for working examples of everything in this guide.
