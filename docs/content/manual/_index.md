---
title: Developer Manual
type: docs
weight: 100
---

Internal documentation for contributors and advanced users: how the SDK works, its extension points, and how to test it.

If you're building an application with the SDK, start with the [Quick Start](/quickstart/) and [Guides](/) instead.

## Dev Setup

### Clone and install

```bash
git clone https://github.com/django-ai-sdk/django-ai-sdk.git
cd django-ai-sdk
make setup      # demo/runtime extras only
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
├── agents/               # registry, services, runtime, config
├── protocols/            # Vercel + OpenAI handlers
├── storage/              # memory + db adapters, ThreadService
├── rags/                 # BM25, ChromaDB, Qdrant hybrid, provider
├── integrations/         # mcp, github, linear, notion, weather
├── pipelines/            # Haystack ToolAgent
├── conversation/         # Thread + Message models
├── memories/             # MemoryService, Entry, documents
├── files/                # processors, transforms, pipelines
├── workflows/            # definition, executor, actions
├── automations/          # declaration, schedule, audience, runner
├── artifacts/            # ArtifactSchema
├── citations/            # CitationRegistry, formatter
├── suggestions/          # DefaultSuggestionGenerator
├── tracing/              # opt-in Haystack spans persisted to the ORM
└── management/commands/  # warmup_rag
```

## Core Concepts

{{< cards >}}
  {{< card link="architecture/" title="System Architecture" icon="cube" subtitle="Components, data flow, design patterns" >}}
  {{< card link="agent/" title="Agent" icon="user" subtitle="Configuration, lifecycle hooks, as_view()" >}}
  {{< card link="agent-registry/" title="Agent Registry" icon="finger-print" subtitle="Registration and stable-ID resolution" >}}
  {{< card link="stream-and-run/" title="Stream and Run" icon="lightning-bolt" subtitle="Haystack integration adapters" >}}
  {{< card link="generators/" title="Generators" icon="chip" subtitle="Per-vendor chat generator factories" >}}
  {{< card link="stream-events/" title="Stream Events" icon="mail" subtitle="The 13 normalized events, tool calls, errors" >}}
  {{< card link="id-generation/" title="ID Generation" icon="hashtag" subtitle="One message_id through SSE, storage, APIs" >}}
{{< /cards >}}

## Protocol

{{< cards >}}
  {{< card link="protocol-handler/" title="Protocol Handler" icon="globe-alt" subtitle="Handler interface, built-ins, stream_response" >}}
  {{< card link="protocol-parts/" title="Protocol Parts" icon="code" subtitle="Wire format and event → part mapping" >}}
{{< /cards >}}

## Storage

{{< cards >}}
  {{< card link="storage/" title="Storage" icon="database" subtitle="ChatMessage format and adapter API" >}}
  {{< card link="thread-service/" title="ThreadService" icon="clipboard-list" subtitle="Permission-checked thread operations" >}}
  {{< card link="storage-registry/" title="Custom Storage Adapters" icon="collection" subtitle="Implementing your own backend" >}}
{{< /cards >}}

## RAG

{{< cards >}}
  {{< card link="rag/" title="RAG" icon="search" subtitle="Provider lifecycle and tool-calling integration" >}}
  {{< card link="rag-variants/" title="RAG Variants" icon="beaker" subtitle="BM25, ChromaDB, Qdrant + config" >}}
{{< /cards >}}

## Operations

{{< cards >}}
  {{< card link="testing/" title="Testing" icon="check-circle" subtitle="Strategy, setup, structure, patterns" >}}
  {{< card link="test-tooling/" title="Test Tooling" icon="cog" subtitle="Fixtures, factories, and mocks" >}}
  {{< card link="tracing/" title="Tracing" icon="chart-square-bar" subtitle="Haystack spans in the ORM, token accounting" >}}
  {{< card link="cli/" title="CLI" icon="terminal" subtitle="warmup_rag implementation" >}}
{{< /cards >}}

## Reference

{{< cards >}}
  {{< card link="settings/" title="Settings Reference" icon="filter" subtitle="Every AI_SDK_* setting and default" >}}
  {{< card link="permissions/" title="Permissions" icon="shield-check" subtitle="Operations, domains, built-in classes" >}}
  {{< card link="memories/" title="Memories" icon="folder" subtitle="MemoryService, documents, thread linking" >}}
  {{< card link="files/" title="Files" icon="document" subtitle="Processors, transforms, upload lifecycle" >}}
  {{< card link="workflows/" title="Workflows" icon="server" subtitle="Definition schema, models, executor" >}}
  {{< card link="automations/" title="Automations" icon="clock" subtitle="Schedule, audience, lease, checks" >}}
{{< /cards >}}
