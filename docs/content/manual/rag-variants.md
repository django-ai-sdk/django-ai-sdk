---
title: RAG Variants
sidebar:
  order: 113
---

The Haystack-backed RAG pipeline variants, their config, and the document model.

All variants live in `django_ai_sdk.rags` and subclass `RAGBase`.

## Comparison

| Variant | Retrieval | Document store |
| --- | --- | --- |
| `BM25QueryExpanderRAG` | BM25 keyword search with query expansion | In-memory |
| `ChromaDBQueryExpanderRAG` | Dense embeddings with query expansion | ChromaDB |
| `QdrantBM25HybridRAG` | Hybrid SPLADE + BGE with RRF fusion | Qdrant |

## Setup
::::tabs
:::tab[BM25]
BM25 needs no GPU and no embedding model. It fits small and medium document sets. `BM25QueryExpanderRAGConfig` is a plain `RAGConfig` with no extra options.

```python
from django_ai_sdk.rags import BM25QueryExpanderRAG, BM25QueryExpanderRAGConfig

rag = BM25QueryExpanderRAG(
    documents=documents,
    config=BM25QueryExpanderRAGConfig(top_k=5, n_expansions=4),
)
```
:::
:::tab[ChromaDB]
Dense embedding retrieval persisted in ChromaDB.

```python
from django_ai_sdk.rags import ChromaDBQueryExpanderRAG, ChromaDBQueryExpanderRAGConfig

rag = ChromaDBQueryExpanderRAG(
    documents=documents,
    config=ChromaDBQueryExpanderRAGConfig(top_k=5),
)
```
:::
:::tab[Qdrant]
Hybrid retrieval with SPLADE sparse and BGE dense embeddings, fused with Reciprocal Rank Fusion (RRF) and stored in Qdrant.

```python
from django_ai_sdk.rags import QdrantBM25HybridRAG, QdrantBM25HybridRAGConfig

rag = QdrantBM25HybridRAG(
    documents=documents,
    config=QdrantBM25HybridRAGConfig(top_k=5),
)
```
:::
::::

## Choosing a variant

Override `get_rag_pipeline()` on your agent to return the variant you want. A common pattern is to factor each variant into its own method and dispatch from `get_rag_pipeline()`:

```python
async def get_rag_pipeline_qdrant(self, memory_id=None): ...  # QdrantBM25HybridRAG

async def get_rag_pipeline(self, memory_id=None):
    return await self.get_rag_pipeline_qdrant(memory_id)
```

The base `get_rag_pipeline()` returns `None`, which disables RAG. Override it to return a RAG instance built from one of the variants above.

| Use case | Variant |
| --- | --- |
| Small/medium docs, no GPU | `BM25QueryExpanderRAG` |
| Large docs, dense embeddings | `ChromaDBQueryExpanderRAG` |
| Production hybrid search | `QdrantBM25HybridRAG` |

## RAGConfig

Base config for all variants:

| Field | Default | Description |
| --- | --- | --- |
| `top_k` | `5` | Maximum documents to retrieve per query |
| `min_score` | `None` | Drop documents below this relevance score (`None` disables) |
| `n_expansions` | `4` | Number of query variations to generate (`1` = no expansion) |
| `expander_model` | `"gpt-4o-mini"` | LLM used for query expansion |
| `expander_prompt` | built-in | Prompt template for query expansion; `{{context}}` gets what the searched documents are about |
| `chunk_size` | `100` | Chunk size for document splitting |
| `chunk_overlap` | `50` | Chunk overlap for document splitting |

### Query expansion

Expansion generates several phrasings of the user's query to improve recall. The first query is always the original, verbatim. `n_expansions = 1` disables expansion. Expansion keeps every query in the user's language, so results match it.

The expander gets `{{context}}`, which says what the documents are about: their most common keywords and some file names (see `expander_context`). Without it, the expander guesses. In a set of CVs, it read "SABRE" as the airline booking system.

```
User Query: "What is the pirate code?"
           ↓
Query Expansion (via expander_model)
├─ "What is the pirate code?"        ← original, verbatim
├─ "Explain the pirate code rules"
├─ "Pirate code of conduct"
└─ "Pirate laws and regulations"
           ↓
Search All Variations (4 × top_k candidates each) → Fuse ranks → Return top-k
```

:::details[Fusing and the search log]
**Fusing.** Reciprocal rank fusion (`fuse_and_rank`) merges the results. A document that several queries find ranks high, whatever score each query gave it, because scores of different queries don't compare. One more ranking counts the words of the user's own query that appear verbatim in a candidate. It weighs as much as all queries together, so an exact term the embeddings don't know, such as an acronym, a code or a name, lifts the document that contains it. Words that appear in most candidates, such as "the" and "is", don't count.

Every search logs one line (`RAG search: ...`) with the queries, the hits per query, the exact terms and what was kept. A line with one query means the expander fell back to the original.
:::

### Persistent storage

Chroma and Qdrant storage configs read their backend from settings with `from_settings`:

- With neither setting, the store is in memory.
- `AI_SDK_VECTOR_STORE_PATH` persists to a local path.
- `AI_SDK_VECTOR_STORE_URL` points at a Qdrant server.

```python
from django_ai_sdk.rags.config import QdrantStorageConfig

config = QdrantStorageConfig.from_settings(memory_id="mem-1")
```

See the [Settings Reference](settings.md) for both settings.

### Remote embeddings

Qdrant hybrid RAG embeds locally with FastEmbed by default. To compute dense embeddings with a hosted model on an OpenAI-compatible endpoint instead, with no local inference, set:

```python
AI_SDK_EMBEDDINGS_MODEL = "Qwen/Qwen3-Embedding-8B"
AI_SDK_EMBEDDINGS_DIM = 4096
```

:::caution
Vectors of different models don't compare, so after changing the model run `python manage.py reindex_memories` (see [CLI](../operations/cli.md)).
:::

## Documents

### RagDocument

The framework-agnostic document used by all variants:

```python
from django_ai_sdk.rags import RagDocument

RagDocument(
    id="doc-1",
    content="The Pirate Code includes: Every man has a vote...",
    metadata={"topic": "rules"},
    title="Pirate Code",
    source="manual.pdf",
    score=0.95,   # optional
)
```

### From a Django QuerySet

- `queryset_to_rag_documents()` turns a Django QuerySet into `RagDocument`s.
- `to_document()` turns a `RagDocument` into a Haystack `Document`.
- `RagDocument.from_document()` turns a Haystack `Document` into a `RagDocument`.

The default `Agent.get_rag_documents(memory_id)` calls your `get_rag_queryset(memory_id)` and maps the results. Override either one to choose what gets indexed.

Next: [Storage](storage.md), where conversations persist.
