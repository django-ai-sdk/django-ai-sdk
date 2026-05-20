---
title: RAG Guide
type: docs
weight: 103
---

Comprehensive guide to Retrieval-Augmented Generation (RAG) in the Django AI SDK.

## Table of Contents

1. [What is RAG?](#what-is-rag)
2. [RAG Architecture](#rag-architecture)
3. [RAG Providers](#rag-providers)
4. [RAG Implementations](#rag-implementations)
5. [Document Flow](#document-flow)
6. [Configuration](#configuration)
7. [Integration Modes](#integration-modes)
8. [Examples](#examples)
9. [Best Practices](#best-practices)

---

## What is RAG?

**Retrieval-Augmented Generation (RAG)** enhances AI assistants with external knowledge:

1. **User asks question**
2. **System retrieves relevant documents** from knowledge base
3. **Documents injected as context** into AI prompt
4. **AI generates informed response**

### Why RAG?

- **Grounds AI in facts** - Reduces hallucinations
- **Up-to-date knowledge** - Documents can be updated anytime
- **Domain-specific** - Use your own documents
- **Citations** - AI can reference sources

---

## RAG Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Assistant                            │
│  ├─ get_rag_queryset()    → Django QuerySet            │
│  ├─ get_rag_documents()   → List[RagDocument]           │
│  ├─ get_rag_pipeline()    → BM25RAG/QdrantRAG          │
│  └─ rag_provider            → BaseRAGProvider            │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                 RAG Provider                            │
│  ├─ warmup()              → Build index (expensive)     │
│  ├─ get_rag_instance()    → Return cached RAG           │
│  ├─ build_tool()          → Create callable tool        │
│  └─ clear_cache()         → Force rebuild              │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              RAG Instance                               │
│  ├─ warmup()              → Build search index          │
│  ├─ retrieve(query)       → Search documents            │
│  ├─ format_context()      → Format for LLM              │
│  └─ as_tool()             → Wrap as ComponentTool       │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                 Adapter                               │
│  ├─ Context injection     → Add to system message       │
│  └─ Tool calling          → Execute via function        │
└─────────────────────────────────────────────────────────┘
```

![RAG Architecture](/images/graphs/rag_architecture.png)

---

## RAG Providers

Providers manage RAG lifecycle and caching.

### BaseRAGProvider

For lightweight RAG (BM25, custom implementations):

```python
from django_ai_sdk.rags import BaseRAGProvider

class MyAssistant(Assistant):
    rag_provider = BaseRAGProvider()
```

**Features:**
- Caches `BaseRAGAdapter` instances
- Creates OpenAI-compatible function tools
- Works with `OpenAIAdapter` for context injection

### HaystackRAGProvider

For Haystack pipeline RAG:

```python
from django_ai_sdk.rags.haystack import HaystackRAGProvider

class MyAssistant(Assistant):
    rag_provider = HaystackRAGProvider()
```

**Features:**
- Caches `HaystackRAGBase` instances
- Creates `ComponentTool` for Haystack agents
- Integrates with Haystack pipelines

### Provider Methods

```python
# Warm up (build index once)
await assistant.rag_provider.warmup(assistant, silo_id)

# Get cached instance
rag = await assistant.rag_provider.get_rag_instance(assistant, silo_id)

# Build tool
tool = await assistant.rag_provider.build_tool(rag, generator)

# Clear cache (after document changes)
assistant.rag_provider.clear_cache()

# Reindex
await assistant.rag_provider.reindex(assistant, silo_id)
```

---

## RAG Implementations

### BM25RAG (Keyword Search)

Lightweight, fast, no GPU required.

```python
from django_ai_sdk.rags import BM25RAG, BM25Config

rag = BM25RAG(
    documents=[
        RagDocument(id="doc1", content="Pirate code rules..."),
        RagDocument(id="doc2", content="Ship commands..."),
    ],
    config=BM25Config(top_k=5, k1=1.5, b=0.75)
)

# Warm up (build index)
rag.warmup()

# Retrieve
result = await rag.retrieve("What is the pirate code?")
# Returns: RAGResult with documents and context
```

**When to use:**
- Small to medium document sets
- Keyword-heavy queries
- No GPU available
- Fast setup

### Haystack RAGs

#### QdrantBM25HybridRAG

Combines BM25 + vector search with reciprocal rank fusion.

```python
from django_ai_sdk.rags.haystack import QdrantBM25HybridRAG

rag = QdrantBM25HybridRAG(
    documents=documents,
    config=QdrantBM25HybridRAGConfig(
        top_k=5,
        use_bm25=True,
        use_embeddings=True,
    )
)
```

**Components:**
- SPLADE for sparse embeddings
- BGE for dense embeddings
- Qdrant for vector storage
- RRF for result fusion

#### ChromaDBQueryExpanderRAG

Vector search with query expansion.

```python
from django_ai_sdk.rags.haystack import ChromaDBQueryExpanderRAG

rag = ChromaDBQueryExpanderRAG(documents=documents)
```

**Components:**
- ChromaDB for vector storage
- Query expansion for better retrieval

### Query Expansion

All Haystack RAG implementations use **query expansion** to improve search results. Here's how it works:

```
User Query: "What is the pirate code?"
           ↓
Query Expansion (via OpenAI)
├─ "What is the pirate code?"
├─ "Explain the pirate code rules"
├─ "Pirate code of conduct"
└─ "Pirate laws and regulations"
           ↓
Search All Variations
├─ Query 1 → Results A
├─ Query 2 → Results B
├─ Query 3 → Results C
└─ Query 4 → Results D
           ↓
Merge & Deduplicate
└─ Return top-k unique documents
```

**Why it helps:**
- Users might not use the exact keywords in your documents
- Expands search coverage without requiring users to reformulate queries
- Finds documents that match conceptually but not literally

**Configuration:**

Control query expansion via `RAGConfig`:

```python
from django_ai_sdk.rags import RAGConfig

config = RAGConfig(
    top_k=5,
    n_expansions=4,  # Number of query variations (default: 4)
)
```

**Note:** Query expansion uses OpenAI to generate variations. It forces same-language expansion so queries maintain the user's language.

### Custom RAG

Create your own RAG implementation:

```python
from django_ai_sdk.rags import BaseRAGAdapter, RAGConfig, RAGResult, RAGSource

class MyCustomRAG(BaseRAGAdapter):
    """Custom RAG implementation."""
    
    def __init__(self, documents, config=None):
        super().__init__(config=config or RAGConfig())
        self.documents = documents
        self._index = None
        self._is_warmed_up = False
    
    def warmup(self) -> None:
        """Build search index."""
        if self._is_warmed_up:
            return
        
        # Build your index here
        self._index = self._build_index(self.documents)
        self._is_warmed_up = True
    
    @property
    def needs_warmup(self) -> bool:
        return not self._is_warmed_up
    
    async def retrieve(self, query: str) -> RAGResult:
        """Search documents."""
        if self.needs_warmup:
            self.warmup()
        
        # Search your index
        results = self._index.search(query, top_k=self.config.top_k)
        
        return RAGResult(
            documents=[{"content": r.content} for r in results],
            context=self._format_context(results),
            sources=[
                RAGSource(id=r.id, content=r.content, metadata=r.metadata)
                for r in results
            ],
            query=query,
        )
    
    def _build_index(self, documents):
        # Your indexing logic
        pass
    
    def _format_context(self, results):
        # Format for LLM
        return "\n\n".join([r.content for r in results])
```

---

## Document Flow

Step-by-step document flow:

```
1. Django QuerySet
   Assistant.get_rag_queryset(silo_id)
   └─> Returns: QuerySet of your models

2. Convert to RagDocuments
   Assistant.get_rag_documents(silo_id)
   └─> Converts QuerySet → List[RagDocument]

3. Create RAG Instance
   Assistant.get_rag_pipeline(silo_id)
   └─> Returns: BM25RAG, QdrantRAG, etc.

4. Provider Cache & Warmup
   rag_provider.get_rag_instance(assistant, silo_id)
   ├─> Check cache: {"AssistantName_silo123": rag_instance}
   ├─> If not cached: call warmup() to build index
   └─> Return cached instance

5. Retrieve Documents
   On each user query:
   rag.retrieve(query) → RAGResult
   ├─> Search index
   ├─> Return top-k documents
   └─> Format context for LLM

6. Context Injection
   OpenAIAdapter:
   ├─> Get last user message as query
   ├─> rag.retrieve(query)
   ├─> Format: context = rag.format_context(result)
   └─> Inject: messages[0]["content"] = f"{context}\n\n{system_msg}"

7. Or Tool Calling
   Haystack Agent:
   ├─> rag.as_tool() → ComponentTool
   ├─> Agent decides when to search
   └─> Tool calls rag.retrieve(query) directly
```

---

## Configuration

### BM25Config

```python
from django_ai_sdk.rags import BM25Config

config = BM25Config(
    top_k=5,           # Number of documents to retrieve
    k1=1.5,            # Term frequency saturation
    b=0.75,            # Length normalization
    document_threshold=0.0,  # Minimum score threshold
)
```

**Parameters:**
- `top_k`: How many documents to retrieve (default: 5)
- `k1`: BM25 term frequency parameter (default: 1.5)
- `b`: BM25 length normalization (default: 0.75)
- `document_threshold`: Minimum relevance score (default: 0.0)

### RAGConfig (Base)

```python
from django_ai_sdk.rags import RAGConfig

config = RAGConfig(
    top_k=3,
    embedder_model="intfloat/multilingual-e5-large-instruct",
    document_threshold=0.7,
)
```

### QdrantBM25HybridRAGConfig

```python
from django_ai_sdk.rags.haystack import QdrantBM25HybridRAGConfig

config = QdrantBM25HybridRAGConfig(
    top_k=5,
    use_bm25=True,
    use_embeddings=True,
    bm25_weight=0.3,           # Weight for BM25 in RRF
    embedding_weight=0.7,      # Weight for embeddings in RRF
    on_disk=False,             # Store Qdrant index on disk
)
```

---

## Integration Modes

### Mode 1: Context Injection (OpenAI)

Automatically injects retrieved context into system message.

```python
class PirateOpenAIAssistant(Assistant):
    rag_provider = BaseRAGProvider()
    
    async def get_rag_pipeline(self, silo_id=None):
        return BM25RAG(documents=docs, config=BM25Config(top_k=5))
    
    async def get_pipeline_adapter(self, thread_id=None):
        rag = await self.rag_provider.get_rag_instance(self, None)
        
        return OpenAIAdapter(
            client=AsyncOpenAI(),
            model=self.model,
            instructions=self.get_instructions(),
            rag_pipeline=rag,  # ← Context injection
            store=True,
            storage_adapter=await self.get_storage_adapter(thread_id),
        )
```

**How it works:**
1. User sends: "What is the pirate code?"
2. Adapter retrieves: Top 5 documents about pirate code
3. Injects into prompt: `"Context: [documents]\n\nQuestion: What is the pirate code?"`
4. OpenAI generates response using context

### Mode 2: Tool Calling (Haystack/Agents)

Exposes RAG as a callable tool the AI can use.

```python
class PirateBasicAssistant(Assistant):
    rag_provider = HaystackRAGProvider()
    
    async def get_rag_pipeline(self, silo_id=None):
        return QdrantBM25HybridRAG(documents=docs)
    
    async def get_pipeline_adapter(self, thread_id=None):
        rag = await self.rag_provider.get_rag_instance(self, None)
        
        # Create tool
        search_tool = await self.rag_provider.build_tool(rag, generator)
        
        # Agent uses tool
        tool_agent = ToolAgent(
            config=ToolAgentConfig(
                model=self.model,
                system_prompt=self.get_system_prompt(),
                tools=[search_tool],  # ← Tool calling
            ),
            generator=generator,
        )
        
        return HaystackAdapter(
            pipeline=tool_agent.pipeline(),
            generator_component=generator,
        )
```

**How it works:**
1. User sends: "What is the pirate code?"
2. Agent decides: "I need to search for this"
3. Calls tool: `search_documents(query="pirate code")`
4. Tool retrieves documents
5. Agent generates response using results

---

## Examples

### Example 1: OpenAI with BM25 RAG

```python
from django.conf import settings
from haystack import Document
from openai import AsyncOpenAI

from django_ai_sdk import Assistant
from django_ai_sdk.adapters.openai import OpenAIAdapter
from django_ai_sdk.rags import BaseRAGProvider, BM25RAG, BM25Config
from django_ai_sdk.storage.memory import MemoryStorageAdapter

class PirateOpenAIAssistant(Assistant):
    """OpenAI assistant with BM25 RAG."""
    
    name = "Pirate with RAG"
    model = settings.AI_SDK_DEFAULT_MODEL
    instructions = [
        "You are a knowledgeable pirate AI assistant.",
        "Use the provided context to answer questions accurately.",
    ]
    
    rag_provider = BaseRAGProvider()
    storage_adapter = MemoryStorageAdapter
    
    def _get_example_documents(self):
        """Return documents for RAG."""
        return [
            Document(
                id="pirate_code",
                content="The Pirate Code includes: Every man has a vote...",
                meta={"topic": "rules"}
            ),
            Document(
                id="ship_commands",
                content="Common commands: 'Avast ye!' means stop...",
                meta={"topic": "commands"}
            ),
            Document(
                id="treasure_map",
                content="X marks the spot on the old island...",
                meta={"topic": "treasure"}
            ),
        ]
    
    async def get_rag_pipeline(self, silo_id=None):
        """Create BM25 RAG pipeline."""
        documents = self._get_example_documents()
        return BM25RAG(
            documents=documents,
            config=BM25Config(top_k=5)
        )
    
    async def get_pipeline_adapter(self, thread_id=None):
        """Create OpenAI adapter with RAG."""
        storage_adapter = await self.get_storage_adapter(thread_id)
        
        # Get RAG from provider (cached)
        rag = await self.rag_provider.get_rag_instance(self, None)
        
        return OpenAIAdapter(
            client=AsyncOpenAI(api_key=settings.OPENAI_API_KEY),
            instructions=self.get_instructions(),
            model=self.get_model(),
            store=True,
            storage_adapter=storage_adapter,
            rag_pipeline=rag,  # Context injection mode
        )
```

### Example 2: Haystack with Qdrant Hybrid RAG

```python
from django_ai_sdk import Assistant
from django_ai_sdk.adapters.haystack import HaystackAdapter
from django_ai_sdk.rags.haystack import HaystackRAGProvider, QdrantBM25HybridRAG
from django_ai_sdk.storage.db import DbStorageAdapter

class PirateBasicAssistant(Assistant):
    """Haystack assistant with hybrid RAG."""
    
    name = "Basic Pirate"
    model = "gpt-4o-mini"
    instructions = ["You are a pirate AI with hybrid search."]
    
    rag_provider = HaystackRAGProvider()
    storage_adapter = DbStorageAdapter
    
    async def get_rag_pipeline(self, silo_id=None):
        documents = await self.get_rag_documents(silo_id)
        return QdrantBM25HybridRAG(documents=documents)
    
    async def get_pipeline_adapter(self, thread_id=None):
        storage_adapter = await self.get_storage_adapter(thread_id)
        
        # Get RAG
        rag = await self.rag_provider.get_rag_instance(self, None)
        
        # Build tool
        search_tool = await self.rag_provider.build_tool(rag, generator)
        
        # Create tool agent
        tool_agent = ToolAgent(
            config=ToolAgentConfig(
                model=self.get_model(),
                system_prompt=self.get_system_prompt(),
                tools=[search_tool],
            ),
            generator=generator,
        )
        
        return HaystackAdapter(
            pipeline=tool_agent.pipeline(),
            generator_component=generator,
            storage_adapter=storage_adapter,
        )
```

### Example 3: Custom RAG Implementation

```python
from django_ai_sdk.rags import BaseRAGAdapter, RAGConfig, RAGResult, RAGSource

class MyCustomRAG(BaseRAGAdapter):
    """Custom RAG using your own search engine."""
    
    def __init__(self, documents, config=None):
        super().__init__(config=config or RAGConfig())
        self.documents = documents
        self._index = None
        self._is_warmed_up = False
    
    def warmup(self) -> None:
        """Build search index."""
        if self._is_warmed_up:
            return
        
        # Your indexing logic here
        self._index = build_custom_index(self.documents)
        self._is_warmed_up = True
    
    @property
    def needs_warmup(self) -> bool:
        return not self._is_warmed_up
    
    async def retrieve(self, query: str) -> RAGResult:
        """Search documents."""
        if self.needs_warmup:
            self.warmup()
        
        # Your search logic
        results = self._index.search(query, top_k=self.config.top_k)
        
        return RAGResult(
            documents=[{"content": r.content} for r in results],
            context="\n\n".join([r.content for r in results]),
            sources=[
                RAGSource(id=r.id, content=r.content, metadata=r.metadata)
                for r in results
            ],
            query=query,
        )

# Usage
class MyAssistant(Assistant):
    rag_provider = BaseRAGProvider()
    
    async def get_rag_pipeline(self, silo_id=None):
        documents = await self.get_rag_documents(silo_id)
        return MyCustomRAG(documents=documents)
```

---

## Best Practices

### 1. Always Use Provider

```python
# Good - uses caching
rag = await self.rag_provider.get_rag_instance(self, silo_id)

# Bad - bypasses caching, rebuilds index every time
rag = await self.get_rag_pipeline(silo_id)
```

### 2. Handle Missing RAG Gracefully

```python
async def get_pipeline_adapter(self, thread_id=None):
    rag = None
    if self.rag_provider:
        rag = await self.rag_provider.get_rag_instance(self, None)
    
    return Adapter(rag_pipeline=rag)  # None is okay
```

### 3. Implement warmup() Correctly

```python
def warmup(self) -> None:
    if self._is_warmed_up:
        return  # Avoid rebuilding
    
    # Build index
    self._index = build_index(self.documents)
    self._is_warmed_up = True  # Required!
```

### 4. Clear Cache When Documents Change

```python
# After adding/updating documents
assistant.rag_provider.clear_cache()

# Or reindex completely
await assistant.rag_provider.reindex(assistant, silo_id)
```

### 5. Use Appropriate RAG Type

| Use Case | Recommended RAG |
|----------|----------------|
| Small docs, no GPU | BM25RAG |
| Large docs, GPU available | QdrantBM25HybridRAG |
| Custom search engine | Custom BaseRAGAdapter |
| Haystack pipelines | HaystackRAGProvider |

### 6. Configure top_k Appropriately

```python
# Too few - might miss context
BM25Config(top_k=1)

# Good balance
BM25Config(top_k=5)

# Too many - might overload context window
BM25Config(top_k=50)
```

---

## Next Steps

- See [Architecture Guide](architecture/) for core concepts
- Check [Adapters](adapters/) for backend integration
- Review [Storage](storage/) for persistence patterns
- Check [Testing](testing/) for test examples
