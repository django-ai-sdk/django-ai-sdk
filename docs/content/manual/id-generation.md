---
title: ID Generation
sidebar:
  order: 106
---

How one `message_id` goes from generation to the SSE stream, to storage and to later operations.

::image[ID Generation Flow]{src="../images/graphs/id_generation.svg" caption="One message_id, from generation to the SSE stream, storage and later operations."}

`Stream.stream()` generates the `message_id` **once**:

::::trace[One message_id, start to finish]
:::step[Stream.stream() generates the message_id once]{icon="code"}
`message_id = str(uuid.uuid4())`
:::

:::step[MessageStartEvent sends it over SSE]{icon="globe"}
`yield MessageStartEvent(message_id)`, on the wire as `{"messageId": "..."}`
:::

:::step[StreamWriter is bound to it for the storage callback]{icon="file-text"}
`StreamWriter(message_id=message_id)`
:::

:::step[The persisted ChatMessage carries it]{icon="file-text"}
`ChatMessage(id=message_id)`
:::

:::step[Rating, delete and restore endpoints use the same ID]{icon="wrench"}
:::
::::

:::details[The same flow as text]

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
:::

So the frontend tracks the message it renders, and the backend rates, deletes or restores the same record.

## StreamWriter

When `store=True` and you pass a `storage_adapter`, `Stream` creates a `StreamWriter` bound to the `message_id`. `stream_writer.add_chunk(...)` collects the chunks as they stream in:

```python
writer = StreamWriter(message_id=message_id, role="assistant")
writer.add_chunk(MessageChunk(type="text", content="Hello"))
writer.add_chunk(MessageChunk(type="text", content=" world"))
```

`MessageChunk` types: `text`, `reasoning`, `tool_call_start`, `tool_input`, `tool_output`, `error`.

## Finalize

On completion, `Stream.get_final_message()`:

:::steps
1. Persists citation sources. It stores only the reference fields `index`, `title`, `source_id`, `memory_id` and `page_number`, and resolves the content from the store when it is read.
2. Calls `stream_writer.finalize("stop")`, which invokes `storage_adapter.storage_callback(...)` to save the message.
:::

The result is in `stream.message_result`.

## Citations and suggestions wiring

- With a `citation_registry`, `Stream` emits a `SourceEvent` per registered source after a retrieval tool runs, in cumulative-index order. `source_id` is `doc_id` or `doc_id:chunk_id`.
- With a `suggestion_generator`, `Stream` generates follow-up questions after the reply from the last six messages and the response. Generation runs under `AI_SDK_SUGGESTION_TIMEOUT`, 5 seconds by default. Timeouts and errors are logged and skipped, and never fail the request.

Next: [Protocol Handler](protocol-handler.md), which converts events to the wire format.
