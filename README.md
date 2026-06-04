# Django AI SDK

A Django SDK for building AI-powered applications with support for multiple LLM providers, RAG (Retrieval-Augmented Generation), and streaming responses.

## Project Status: Read This First

This is an **early preview**. We're actively iterating on the API and learning from real usage. Here's what that means for you:

- **Expect breaking changes**: APIs will shift as we find better patterns.
- **Migrations might be reset**: Don't rely on database schema stability between versions.
- **Not for production**: Use this for experimentation, prototypes, and side projects. Keep critical workloads elsewhere.
- **Watch the repo**: Things change quickly. Star & watch to stay in the loop.
- **Your feedback shapes the SDK**: Break things, open issues, tell us what hurts.

We'd love to have you along for the ride, just keep your seatbelt on.


## Install

```bash
pip install django-ai-sdk
```

Or with uv:

```bash
uv add django-ai-sdk
```

## Quick Start

### 1. Add to INSTALLED_APPS

```python
# settings.py
INSTALLED_APPS = [
    ...
    "django_ai_sdk",
]
```

Then run `python manage.py migrate`.

### 2. Define your assistant

```python
# assistants.py
from django_ai_sdk import Assistant

class HelpDeskAssistant(Assistant):
    name = "Help Desk"
    model = "gpt-4o"
    instructions = "You are a helpful support assistant."
```

### 3. Return a streaming response

```python
# views.py
from .assistants import HelpDeskAssistant

assistant = HelpDeskAssistant()

@router.post("/chat")
async def chat(request, payload: ChatRequest):
    return await assistant.as_view(
        payload.messages,
        thread_id=payload.thread_id,
    )
```

## Features

- **RAG Pipelines**: BM25, ChromaDB, and Qdrant hybrid search with query expansion.
- **Streaming Responses**: Built-in SSE streaming. Works with Vercel AI SDK protocol.
- **Conversation Storage**: Automatic message persistence. Thread-based history out of the box.
- **Tool Calling**: MCP, memory, and custom tools — all managed by your Assistant.
- **Reindexing**: Hot-reload documents. Cached embeddings with simple refresh API.

## Documentation

Full documentation and examples: [github.com/django-ai-sdk/django-ai-sdk](https://github.com/django-ai-sdk/django-ai-sdk)
