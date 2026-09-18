# Django AI SDK

Build AI agents in Django. Batteries included.

## Project Status: Read This First

This is a **beta**. The API may still change as we find better patterns.
Here's what that means for you:

- **API may still evolve**: some interfaces will shift as we find better patterns.
- **Not for production yet**: use this for experimentation, prototypes, and side projects. Keep critical workloads elsewhere until we hit stable.
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

### 2. Define your agent

```python
# agents.py
from django_ai_sdk import Agent
from django_ai_sdk.adapters.base import Stream
from django_ai_sdk.generators import openai_responses_chat
from haystack import Pipeline


class HelpDeskAgent(Agent):
    name = "Help Desk"
    model = "gpt-5-mini"
    instructions = "You are a helpful support agent."
    llm = openai_responses_chat

    async def get_pipeline_adapter(self, thread_id=None, user=None):
        storage_adapter = await self.get_storage_adapter(thread_id)
        generator = self.get_llm()
        return Stream(
            pipeline=Pipeline(),
            generator=generator,
            storage_adapter=storage_adapter,
        )
```

For non-streaming tasks (title generation, structured output), use `Run` instead:

```python
from django_ai_sdk.adapters.base import Run

    async def get_run_adapter(self, thread_id=None, user=None):
        return Run(generator=self.get_llm())
```

### 3. Return a streaming response

```python
# views.py
from .agents import HelpDeskAgent

agent = HelpDeskAgent()


@router.post("/chat")
async def chat(request, payload: ChatRequest):
    return await agent.as_view(
        payload.messages,
        thread_id=payload.thread_id,
    )
```

## Features

- **Agents**: A Django `Agent` class you can run from a view, a task, or the API,
  streaming or not, with tool calling built in.
- **Workflows**: Multi-step, definition-driven runs with persisted steps and actions,
  so a process can span multiple agent calls with a record of what happened.
- **Subagents**: Agents can delegate to other agents as tools, so one agent can
  orchestrate specialists instead of doing everything itself.
- **Tool Calling**: MCP, memory, and custom tools, all managed by your Agent.
- **Streaming Responses**: Built-in SSE streaming. Works with Vercel AI SDK protocol.
- **Conversation Storage**: Automatic message persistence. Thread-based history out of the box.
- **RAG Pipelines**: BM25, ChromaDB, and Qdrant hybrid search with query expansion,
  for when an agent needs to ground itself in your documents.
- **Artifacts**: 16 structured UI types (tables, plans, approval cards, code blocks, and more)
  submitted by the LLM via tool calls.
- **File Processing**: Document upload with pipeline-based processing (text, CSV, JSON,
  DOCX, PPTX, XLSX. Extraction transforms for metadata embedding.
- **Integrations**: Third-party tools as self-registering Django apps, with caching,
  circuit breaking and OAuth built in.
- **Tracing**: Opt-in Haystack tracing persisted to the ORM, with per-thread and
  per-message token accounting.
- **Reindexing**: Hot-reload documents. Cached embeddings with simple refresh API.

## Documentation

Full documentation and examples: [github.com/django-ai-sdk/django-ai-sdk](https://github.com/django-ai-sdk/django-ai-sdk)
