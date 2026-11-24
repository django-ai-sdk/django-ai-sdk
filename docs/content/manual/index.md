---
title: Developer Manual
sidebar:
  order: 100
---

Internal documentation for contributors and advanced users: how the SDK works, its extension points, and how to test it.

If you're building an application with the SDK, start with the [Quick Start](../quickstart.md) and [Guides](../index.md) instead.

## Dev Setup

### Clone and install

```bash
git clone https://github.com/django-ai-sdk/django-ai-sdk.git
cd django-ai-sdk
make setup      # runtime + demo extras only
make setup-all  # every extra, including the torch-based transformers stack
```

### Run tests

```bash
make test
```

### Regenerate docs diagrams

```bash
make docs-graphs
```

## Repository Layout

```
django_ai_sdk/
├── agent.py              # Agent coordinator class
├── common.py             # ChatMessage, StreamWriter, MessageChunk
├── events.py             # 13 normalized StreamEvent types
├── responses.py          # stream_response()
├── permissions.py        # BasePermission, built-ins, Operation enum
├── adapters/             # Stream, Run
├── agents/               # registry, services, runtime, config, ToolAgent + subagent engine
├── protocols/            # Vercel + OpenAI handlers
├── storage/              # memory + db adapters, ThreadService
├── rags/                 # BM25, ChromaDB, Qdrant hybrid, provider
├── integrations/         # mcp, github, linear, notion, weather
├── conversation/         # Thread + Message models
├── memories/             # MemoryService, Entry, documents
├── files/                # processors, transforms, pipelines
├── workflows/            # definitions, runs, listeners
├── durable/              # workflow runs, triggers and tasks on DBOS (opt-in app)
├── artifacts/            # ArtifactSchema
├── citations/            # CitationRegistry, formatter
├── suggestions/          # DefaultSuggestionGenerator
├── tracing/              # opt-in Haystack spans persisted to the ORM
└── management/commands/  # warmup_rag
```

## Core Concepts
::::cards
:::card[System Architecture]{href="architecture.md"}
Components, data flow, design patterns
:::
:::card[Agent]{href="agent.md"}
Configuration, lifecycle hooks, as_view()
:::
:::card[Agent Registry]{href="agent-registry.md"}
Registration and stable-ID resolution
:::
:::card[Stream and Run]{href="stream-and-run.md"}
Haystack integration adapters
:::
:::card[Generators]{href="generators.md"}
Per-vendor chat generator factories
:::
:::card[Stream Events]{href="stream-events.md"}
The 13 normalized events, tool calls, errors
:::
:::card[ID Generation]{href="id-generation.md"}
One message_id through SSE, storage, APIs
:::
::::

## Protocol
::::cards
:::card[Protocol Handler]{href="protocol-handler.md"}
Handler interface, built-ins, stream_response
:::
:::card[Protocol Parts]{href="protocol-parts.md"}
Wire format and event → part mapping
:::
::::

## Storage
::::cards
:::card[Storage]{href="storage.md"}
ChatMessage format and adapter API
:::
:::card[ThreadService]{href="thread-service.md"}
Permission-checked thread operations
:::
:::card[Custom Storage Adapters]{href="storage-registry.md"}
Implementing your own backend
:::
::::

## RAG
::::cards
:::card[RAG]{href="rag.md"}
Provider lifecycle and tool-calling integration
:::
:::card[RAG Variants]{href="rag-variants.md"}
BM25, ChromaDB, Qdrant + config
:::
::::

## Operations
::::cards
:::card[Testing]{href="testing.md"}
Strategy, setup, structure, patterns
:::
:::card[Test Tooling]{href="test-tooling.md"}
Fixtures, factories, and mocks
:::
:::card[Tracing]{href="tracing.md"}
Haystack spans in the ORM, token accounting
:::
:::card[CLI]{href="cli.md"}
warmup_rag implementation
:::
::::

## Reference
::::cards
:::card[Settings Reference]{href="settings.md"}
Every AI_SDK_* setting and default
:::
:::card[Permissions]{href="permissions.md"}
Operations, domains, built-in classes
:::
:::card[Memories]{href="memories.md"}
MemoryService, documents, thread linking
:::
:::card[Files]{href="files.md"}
Processors, transforms, upload lifecycle
:::
:::card[Workflows]{href="workflows.md"}
Definition schema, models, runs
:::
:::card[Webhooks]{href="webhooks.md"}
Inbound events, answered by an agent
:::
::::
