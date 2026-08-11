---
title: Agent
type: docs
weight: 102
---

Deep dive into the `Agent` class: what it configures and which hooks the SDK calls.

{{< callout type="info" >}}
Building an app? See the [Agents guide](/docs/agents/). This page is the internal reference.
{{< /callout >}}

## Responsibilities

The `Agent` is the coordinator. It:

- Declares AI personality: `name`, `model`, `instructions`
- Picks a protocol handler and a storage adapter
- Provides the adapter hooks `Stream`/`Run` are built from
- Exposes lifecycle hooks for RAG warmup, citations, and suggestions

## Configuration

Key class attributes (full table in the [Agents guide](/docs/agents/)):

| Variable | Default | Purpose |
| --- | --- | --- |
| `name` | None | Display name |
| `model` | `None` | Model identifier passed to the generator |
| `instructions` | built-in prompt | System prompt, built with `prompt()` |
| `protocol` | `VercelProtocolHandler` | Wire-format handler class |
| `storage_adapter` | `MemoryStorageAdapter` | Fallback storage class |
| `tools` | `[]` | Class-level tool providers |
| `integrations` | `[]` | Integration names whose tools reach this agent |
| `artifacts` | `[]` | `ArtifactSchema` subclasses exposed as submission tools |
| `rag_provider` | `None` | `RAGProvider` instance for RAG |
| `memories` | `[]` | Default connected memories |
| `max_history` | `None` | Cap on messages sent to the model |
| `response_format` | `None` | Pydantic model for structured `run()` output |
| `title_generation` | `True` | Auto-generate thread titles |
| `permissions` | `[AllowAll]` | Permission classes gating operations |
| `warmup_on_init` | `False` | Warm up RAG on instantiation |
| `hidden` / `abstract` | `False` | Registry behavior flags |

`Agent.__init__` instantiates `self.protocol()` into `self.protocol_handler`.

## Adapter Hooks

The SDK calls two abstract hooks; you implement them:

| Hook | Returns | Used by |
| --- | --- | --- |
| `get_pipeline_adapter(thread_id, user)` | `Stream` | `Agent.as_view()` (streaming chat) |
| `get_run_adapter(thread_id, user)` | `Run` | `Agent.run()` (titles, extraction, jobs) |

```python
async def get_pipeline_adapter(self, thread_id=None, user=None):
    generator = OpenAIChatGenerator(...)
    tool_agent = ToolAgent(
        config=ToolAgentConfig(
            model=self.get_model(),
            system_prompt=self.get_system_prompt(),
            tools=await self.get_tools(thread_id=thread_id or "", user=user),
        ),
        generator=generator,
    )
    return Stream(
        pipeline=tool_agent.pipeline(),
        generator=generator,
        storage_adapter=await self.get_storage_adapter(thread_id),
    )
```

A worker-only agent (`hidden = True`, never in chat) can leave `get_pipeline_adapter()` unimplemented.

## Storage Resolution

`get_storage_adapter(thread_id)` queries all registered adapters for the thread and returns a bound instance, falling back to the agent's `storage_adapter`. See [Storage](../storage/).

## Tool Assembly

`get_tools(thread_id, user)` combines, in order:

1. Class-level `tools` providers (each called with `thread_id`/`user` kwargs)
2. Integration tools (`integrations` list)
3. Artifact submission tools (`artifacts`)

`get_rag_tools(thread_id, user)` appends one retrieval tool per active memory when `rag_provider` is set. See [RAG](../rag/).

## RAG Lifecycle Hooks

Classmethods that delegate to `rag_provider` (no-op without a provider):

| Hook | Effect |
| --- | --- |
| `MyAgent.warmup(agent, memory_id)` | Pre-build and cache pipelines |
| `MyAgent.reindex(agent, memory_id, force_rebuild)` | Drop cache entry and rebuild |
| `MyAgent.clear_rag_cache(agent)` | Drop all cached instances |

## Citations & Suggestions

| Hook | Returns |
| --- | --- |
| `get_citation_registry()` | Fresh per-turn `CitationRegistry` |
| `get_citation_formatter()` | From `citation_formatter_class` |
| `get_suggestion_generator()` | `self.suggestion_generator(agent=self)` |

## Entry Point

`as_view(protocol_messages, thread_id=None, user=None)`:

1. Checks `CHAT` permissions
2. Converts protocol messages via `self.protocol_handler.to_chat_messages()`
3. Stores the newest user message when a thread is active
4. Builds the pipeline adapter and returns a `StreamingHttpResponse`

```python
response = await agent.as_view(payload.messages, thread_id=..., user=request.user)
```

Next: [Agent Registry](../agent-registry/), registration and stable-ID resolution.
