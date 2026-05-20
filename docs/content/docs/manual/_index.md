---
title: Developer Manual
type: docs
weight: 100
---

This manual is for developers who want to understand Django AI SDK's internals, extend its capabilities, or contribute to the project.

If you're building an application with the SDK, start with the [Quick Start](/docs/quickstart/) and [Guides](/docs/) instead.

---

## What You'll Find Here

- **Architecture** — How components interact internally
- **Extension Points** — Where and how to add custom functionality  
- **Design Decisions** — Why we built it this way
- **Contribution Guide** — How to develop PRs effectively

---

## Quick Navigation

- **[Architecture](architecture/)** — Component interactions and data flow
- **[Adapters](adapters/)** — Adding new AI backends
- **[CLI](cli/)** — Command line interface implementation
- **[RAG](rag/)** — Knowledge retrieval internals
- **[Storage](storage/)** — Persistence layer architecture
- **[Storage Registry](storage-registry/)** — Cross-storage adapter registry
- **[Testing](testing/)** — Test patterns for contributors
- **[Protocol Parts](protocol-parts/)** — Complete protocol reference

---

## Getting Started with Development

```bash
# Clone the repo
git clone https://github.com/yourusername/django-ai-sdk.git
cd django-ai-sdk

# Install dependencies
pip install -e ".[dev]"

# Run tests
PYTHONPATH=demo pytest django_ai_sdk/tests -v

# Generate documentation diagrams
python manual/graph.py
```

---

## Architecture at a Glance

```
Request → Assistant.as_view() → Protocol Handler → Adapter → AI Backend
                                         ↓
                                    StreamWriter → Storage
```

The system is built around five core abstractions:

1. **Assistant** — Orchestration layer, configures components
2. **Protocol Handler** — Message format conversion (Vercel, custom)
3. **Adapter** — AI backend integration (OpenAI, Haystack, etc.)
4. **Storage** — Conversation persistence abstraction
5. **RAG** — Knowledge retrieval and context injection

Each abstraction has a base class defining the interface. Implementation details are in subclasses.

---

## Design Principles

### Single ID Source
UUIDs generated once in the adapter flow through the entire system:
- SSE stream (frontend tracking)
- Storage (database)
- API endpoints (operations)

**Why**: Prevents ID mismatch bugs.

### Protocol-Agnostic Core
Internal `ChatMessage` format is separate from wire protocols.

**Why**: Enables supporting multiple protocols without changing assistant code.

### Async-First
All streaming operations use async generators.

**Why**: Non-blocking I/O with Django's ASGI support.

---

## Contributing

We welcome contributions! Areas where help is needed:

- New adapter implementations
- Additional RAG providers
- Protocol handler extensions
- Documentation improvements
- Bug fixes and testing

See individual sections for implementation details.
