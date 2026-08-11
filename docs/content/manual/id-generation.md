---
title: ID Generation
type: docs
weight: 106
---

How a single `message_id` flows from generation → SSE → storage → later operations.

![ID Generation Flow](/images/graphs/id_generation.png)

The `message_id` is generated **once** in `Stream.stream()`:

```
Stream.stream()
    ↓
message_id = str(uuid.uuid4())  # Generate once
    ↓
yield MessageStartEvent(message_id)  # SSE: {"messageId": "..."}
    ↓
StreamWriter(message_id=message_id)  # Storage callback
    ↓
ChatMessage(id=message_id)          # Persisted message
    ↓
Rating / delete / restore endpoints use the same ID
```

This is what lets the frontend track the message it renders while the backend rates, deletes, or restores the same record.

## StreamWriter

When `store=True` and a `storage_adapter` is provided, `Stream` creates a `StreamWriter` bound to the `message_id`. As chunks stream in, `stream_writer.add_chunk(...)` accumulates them:

```python
writer = StreamWriter(message_id=message_id, role="assistant")
writer.add_chunk(MessageChunk(type="text", content="Hello"))
writer.add_chunk(MessageChunk(type="text", content=" world"))
```

`MessageChunk` types: `text`, `reasoning`, `tool_call_start`, `tool_input`, `tool_output`, `error`.

## Finalize

On completion, `Stream.get_final_message()`:

1. Persists citation sources (reference fields only: `index`, `title`, `source_id`, `memory_id`, `page_number`; content is resolved fresh from the store).
2. Calls `stream_writer.finalize("stop")`, which invokes `storage_adapter.storage_callback(...)` to save the message.

The result is available as `stream.message_result`.

## Citations & Suggestions Wiring

- With a `citation_registry`, `Stream` emits a `SourceEvent` per registered source after a retrieval tool runs, in cumulative-index order. `source_id` is `doc_id` or `doc_id:chunk_id`.
- With `suggestion_generator`, `Stream` generates follow-up questions after the reply using the last six messages and the response. Generation runs under `AI_SDK_SUGGESTION_TIMEOUT` (default 5s); timeouts and errors are logged and skipped, never fatal.

Next: [Protocol Handler](../protocol-handler/), converting events to wire format.
