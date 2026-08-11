---
title: RAG
type: docs
weight: 112
---

How RAG works in the SDK: the provider lifecycle and how retrieval is exposed to agents as tools.

{{< callout type="info" >}}
Pipeline variants, config, and the document model live on the [RAG Variants](../rag-variants/) page.
{{< /callout >}}

![RAG Architecture](/images/graphs/rag_architecture.png)

## What is RAG?

**Retrieval-Augmented Generation** grounds the model in facts:

1. User asks a question
2. Agent retrieves relevant documents from a memory
3. Documents pass back to the model as retrieved context
4. Model responds, citing numbered sources

## Enabling RAG on an Agent

### Set the provider

Set `rag_provider = RAGProvider()` on the agent.

### Return the documents

Override `get_rag_queryset()` (and/or `get_rag_documents()`) to select what gets indexed.

### Return a pipeline

Implement `get_rag_pipeline()` (or one of the named variants) to return a RAG pipeline.

### Link memories to threads

`get_rag_tools()` builds one retrieval tool per active memory.

```python
from django_ai_sdk.memories.models import Entry
from django_ai_sdk.rags import BM25QueryExpanderRAG, BM25QueryExpanderRAGConfig
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

Wire the retrieval tools into the `ToolAgent` inside `get_pipeline_adapter()`:

```python
async def get_pipeline_adapter(self, thread_id=None, user=None):
    generator = OpenAIChatGenerator(...)
    citation_registry = self.get_citation_registry()

    rag_tools = await self.get_rag_tools(
        thread_id=thread_id,
        citation_registry=citation_registry,
        citation_formatter=self.get_citation_formatter(),
        user=user,
    )
    tool_agent = ToolAgent(
        config=ToolAgentConfig(
            model=self.get_model(),
            system_prompt=self.get_system_prompt(),
            tools=[*await self.get_tools(thread_id=thread_id, user=user), *rag_tools],
        ),
        generator=generator,
    )
    return Stream(
        pipeline=tool_agent.pipeline(),
        generator=generator,
        storage_adapter=await self.get_storage_adapter(thread_id),
        citation_registry=citation_registry,
    )
```

## RAGProvider

`RAGProvider` manages the RAG lifecycle: warmup (building indexes), an LRU cache of instances (`_MAX_CACHE_SIZE = 100`), and conversion of pipelines to tools.

| Method | Purpose |
| --- | --- |
| `warmup(agent, memory_id, force_rebuild=False)` | Build indexes via `get_or_create`, then cache the RAG instance |
| `get_rag_instance(agent, memory_id)` | Return the cached RAG, creating + warming it if absent |
| `get_cached_rag_instance(agent, memory_id)` | Return the cached RAG without touching the store (safe when a second connection would conflict, e.g. Qdrant's exclusive local lock) |
| `get_tool(agent, memory_id, *, spec, citation_registry, citation_formatter)` | Build a `ComponentTool` with optional citation wiring |
| `build_tool(rag_instance, *, spec=None)` | Wrap a RAG instance as a `ComponentTool` (`get_tool(spec)` with a spec, else `as_tool()`) |
| `add_documents(agent, memory_id, documents)` | Incrementally add documents to an existing index |
| `remove_documents(agent, memory_id, document_ids)` | Incrementally remove documents from an existing index |
| `reindex(agent, memory_id, force_rebuild=False)` | Drop the cache entry and rebuild (returns the new instance) |
| `clear_cache()` | Drop all cached RAG instances |

{{< callout type="important" >}}
Warmups are serialized per cache key with a per-key `asyncio.Lock`: concurrent requests for the same memory don't double-build (important for backends like Qdrant that hold an exclusive file lock during warmup). A warm cache returns without acquiring any lock.
{{< /callout >}}

## Integration: Tool Calling

RAG is exposed as a **per-memory retrieval tool** the model decides to call: there is no automatic context injection. Each memory linked to the thread produces one tool named after its `ToolSpec`:

```python
spec = await memory.get_tool_spec()  # -> ToolSpec(name="search_documents", ...)
```

Tool names are namespaced per memory so unrelated memories don't collide; if two memories produce the same name, the later one is suffixed with a short memory-ID hash and a warning is logged. Memories with zero documents are skipped with a warning. Retrieval results stream back as numbered `SourceEvent`s.

## Citations

Retrieved sources are rendered to the model as numbered citations and streamed to the frontend:

- `get_citation_registry()`: a **fresh per-turn** `CitationRegistry` so indices reset between turns.
- `get_citation_formatter()` / `citation_formatter_class`: how sources render for the model. `DefaultCitationFormatter` numbers sources so the model can reference them inline.
- `Stream` emits a `SourceEvent(index, title, content, source_id, ...)` per source after a retrieval tool runs, in cumulative-index order.
- Protocol handlers translate these into the frontend's citation format (Vercel `source-document` parts carry the citation index in `providerMetadata`).

Wire the same registry and formatter into both `get_rag_tools()` and the `Stream` so citations round-trip consistently.

## Lifecycle

| Action | Where | Effect |
| --- | --- | --- |
| Warm up | `Agent.warmup(agent, memory_id)` (classmethod) | Pre-builds and caches pipelines; no-op without a provider |
| Auto warm up | `warmup_on_init = True` | Schedules `rag_provider.warmup()` on instantiation |
| Reindex | `Agent.reindex(agent, memory_id, force_rebuild)` | Drops the cache entry and rebuilds; persistent backends (Qdrant) delete and recreate |
| Clear cache | `Agent.clear_rag_cache(agent)` | Drops all cached instances |
| CLI | `python manage.py warmup_rag --agent <name>` | Warms up from the command line ([CLI](../cli/)) |

```python
await MyAgent.warmup(agent, memory_id="mem-1")                 # build once
await MyAgent.reindex(agent, memory_id="mem-1")                # rebuild
MyAgent.clear_rag_cache(agent)                                 # drop cache
```

## Best Practices

| Practice | Why |
| --- | --- |
| Route through the provider (`get_rag_instance`) | Avoids rebuilding the index on every call |
| Use `get_cached_rag_instance()` with Qdrant | Its local file lock is exclusive, so reuse the warmed instance |
| Return `None` from `get_rag_pipeline()` with no documents | Provider caches `None`; `get_rag_tools()` skips the memory cleanly |
| Set `top_k` to match your context budget | `1` risks missing context; `50` floods the window. Start at `5` |
| Reindex after document changes | Or use incremental `add_documents()` / `remove_documents()` |

Next: [Storage](../storage/), where conversations persist.
