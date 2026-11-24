---
title: Stream Events
sidebar:
  order: 105
---

The normalized events that `Stream.stream()` yields, and how tool calls and errors reach the client.

All events live in `django_ai_sdk.events` and subclass `StreamEvent`.

## Reference

| Event | When | Key fields |
| --- | --- | --- |
| `MessageStartEvent` | Stream begins | `message_id` |
| `TextChunkEvent` | Text token | `content` |
| `ReasoningChunkEvent` | Reasoning token | `content` |
| `DataEvent` | Arbitrary structured data | `data_type`, `data` |
| `DataEvent` (`oversight`) | An [oversight](oversight.md#client-notices) rule acted at the gate, the tool stage or finish | `data_type="oversight"`, `data`: `turn_id`, `stage`, `outcome`, and `message`, `replace` or `input` as they apply |
| `ToolCallStartEvent` | Tool call begins | `tool_call_id`, `tool_name` |
| `ToolInputChunkEvent` | Incremental tool args | `tool_call_id`, `input_chunk` |
| `ToolInputCompleteEvent` | Tool args ready | `tool_call_id`, `tool_name`, `tool_input` |
| `ToolOutputEvent` | Tool result | `tool_call_id`, `tool_output` |
| `SourceEvent` | RAG citation | `index`, `title`, `content`, `source_id` |
| `SuggestionEvent` | Follow-up questions | `suggestions` |
| `TextEndEvent` | The answer's text is complete; citations may still follow | (none) |
| `MessageEndEvent` | Message complete | `finish_reason` (`content-filter` when oversight blocked the question or withheld the answer) |
| `ErrorEvent` | Error occurred | `error_code`, `ref` (see [Error Handling](../serving/errors.md)) |
| `StreamEndEvent` | Stream terminated | (none) |

## Handling events

```python
async for event in stream.stream(messages):
    match event:
        case MessageStartEvent():
            print(f"Message {event.message_id} started")
        case TextChunkEvent():
            print(event.content, end="")
        case ToolCallStartEvent():
            print(f"Using: {event.tool_name}")
        case ToolInputCompleteEvent():
            print(f"Input: {event.tool_input}")
        case ToolOutputEvent():
            print(f"Output: {event.tool_output}")
        case MessageEndEvent():
            print(f"Finish reason: {event.finish_reason}")
        case ErrorEvent():
            print(f"Error {event.error_code} (ref {event.ref})")
        case StreamEndEvent():
            print("Done")
```

## Tool call handling

Tool calls and results can arrive as streaming chunks and in the final pipeline result. `Stream` handles both:

- Emits `ToolCallStartEvent` / `ToolInputCompleteEvent` / `ToolOutputEvent` from chunks during streaming.
- Runs `get_pipeline_result()` after the pipeline completes to capture tool calls that only appear in the final messages. It converts them to `MessageChunk`s on the `StreamWriter`, so they persist.

Two helpers process tool data. `parse_tool_input()` JSON-decodes tool arguments and falls back to the raw string. `parse_tool_output()` makes Haystack results JSON-serializable.

## Error handling

`Stream.stream()` handles failures as follows:

- A pipeline failure produces an `ErrorEvent` followed by `StreamEndEvent`. The message is persisted with `finish_reason="error"`.
- An unexpected exception also yields an `ErrorEvent`. It carries only the error code and a `ref`. The SDK logs the exception text with that ref and stores it for admins.
- On cancellation, `pipeline_task.cancel()` runs in `finally` and the partial message is persisted with `finish_reason="cancelled"`.
- `MessageEndEvent` and `StreamEndEvent` always end the stream.

Next: [ID Generation](id-generation.md), how one `message_id` passes through the system.
