---
title: RAG Variants
type: docs
weight: 113
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

{{< tabs >}}
{{< tab name="BM25" >}}
Lightweight, fast, no GPU. Best for small to medium document sets. `BM25QueryExpanderRAGConfig` is a plain `RAGConfig`: no extra options.

```python
from django_ai_sdk.rags import BM25QueryExpanderRAG, BM25QueryExpanderRAGConfig

rag = BM25QueryExpanderRAG(
    documents=documents,
    config=BM25QueryExpanderRAGConfig(top_k=5, n_expansions=4),
)
```
{{< /tab >}}
{{< tab name="ChromaDB" >}}
Dense embedding retrieval persisted in ChromaDB.

```python
from django_ai_sdk.rags import ChromaDBQueryExpanderRAG, ChromaDBQueryExpanderRAGConfig

rag = ChromaDBQueryExpanderRAG(
    documents=documents,
    config=ChromaDBQueryExpanderRAGConfig(top_k=5),
)
```
{{< /tab >}}
{{< tab name="Qdrant" >}}
Hybrid retrieval: SPLADE sparse + BGE dense embeddings fused with Reciprocal Rank Fusion (RRF), stored in Qdrant.

```python
from django_ai_sdk.rags import QdrantBM25HybridRAG, QdrantBM25HybridRAGConfig

rag = QdrantBM25HybridRAG(
    documents=documents,
    config=QdrantBM25HybridRAGConfig(top_k=5),
)
```
{{< /tab >}}
{{< /tabs >}}

## Choosing a Variant

Override `get_rag_pipeline()` on your agent to return the variant you want. A common pattern is to factor each variant into its own method and dispatch from `get_rag_pipeline()`:

```python
async def get_rag_pipeline_qdrant(self, memory_id=None): ...  # QdrantBM25HybridRAG

async def get_rag_pipeline(self, memory_id=None):
    return await self.get_rag_pipeline_qdrant(memory_id)
```

The base `get_rag_pipeline()` returns `None` (RAG disabled). Override it to return a RAG instance built from one of the variants above.

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
| `expander_prompt` | built-in | Prompt template for query expansion |
| `chunk_size` | `100` | Chunk size for document splitting |
| `chunk_overlap` | `50` | Chunk overlap for document splitting |

### Query Expansion

Expansion generates several phrasings of the user's query to improve recall. The first query is always the original, verbatim; `n_expansions = 1` disables expansion. Expansion forces **same-language** queries so results match the user's language.

```
User Query: "What is the pirate code?"
           ↓
Query Expansion (via expander_model)
├─ "What is the pirate code?"        ← original, verbatim
├─ "Explain the pirate code rules"
├─ "Pirate code of conduct"
└─ "Pirate laws and regulations"
           ↓
Search All Variations → Merge & Deduplicate → Return top-k
```

### Persistent Storage

Chroma and Qdrant storage configs derive their backend from settings via `from_settings`: in-memory when unset, `AI_SDK_VECTOR_STORE_PATH` for local persistence, and `AI_SDK_VECTOR_STORE_URL` (Qdrant server) when set:

```python
from django_ai_sdk.rags.config import QdrantStorageConfig

config = QdrantStorageConfig.from_settings(memory_id="mem-1")
```

See the [Settings Reference](/docs/manual/settings/) for both settings.

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

- `queryset_to_rag_documents()`: Django QuerySet → `RagDocument`s
- `to_document()`: `RagDocument` → Haystack `Document`
- `RagDocument.from_document()`: Haystack `Document` → `RagDocument`

The default `Agent.get_rag_documents(memory_id)` calls your `get_rag_queryset(memory_id)` and maps the results. Override either to control exactly what gets indexed.

Next: [Storage](../storage/), where conversations persist.
