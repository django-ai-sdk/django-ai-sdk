---
title: Stream Events
type: docs
weight: 105
---

The normalized events `Stream.stream()` yields, and how tool calls and errors surface.

All events live in `django_ai_sdk.events` and subclass `StreamEvent`.

## Reference

| Event | When | Key fields |
| --- | --- | --- |
| `MessageStartEvent` | Stream begins | `message_id` |
| `TextChunkEvent` | Text token | `content` |
| `ReasoningChunkEvent` | Reasoning token | `content` |
| `DataEvent` | Arbitrary structured data | `data_type`, `data` |
| `ToolCallStartEvent` | Tool call begins | `tool_call_id`, `tool_name` |
| `ToolInputChunkEvent` | Incremental tool args | `tool_call_id`, `input_chunk` |
| `ToolInputCompleteEvent` | Tool args ready | `tool_call_id`, `tool_name`, `tool_input` |
| `ToolOutputEvent` | Tool result | `tool_call_id`, `tool_output` |
| `SourceEvent` | RAG citation | `index`, `title`, `content`, `source_id` |
| `SuggestionEvent` | Follow-up questions | `suggestions` |
| `MessageEndEvent` | Message complete | `finish_reason` |
| `ErrorEvent` | Error occurred | `error_message`, `error_code` |
| `StreamEndEvent` | Stream terminated | (none) |

## Handling Events

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
            print(f"Error: {event.error_message}")
        case StreamEndEvent():
            print("Done")
```

## Tool Call Handling

Tool calls and results can arrive both as streaming chunks and in the final pipeline result. `Stream`:

- Emits `ToolCallStartEvent` / `ToolInputCompleteEvent` / `ToolOutputEvent` from chunks during streaming.
- Runs `get_pipeline_result()` after the pipeline completes to capture tool calls that only appear in the final messages, converting them to `MessageChunk`s on the `StreamWriter` so they persist.

Helpers: `parse_tool_input()` JSON-decodes tool arguments (falling back to the raw string); `parse_tool_output()` makes Haystack results JSON-serializable.

## Error Handling

`Stream.stream()` is defensive:

- Pipeline failures produce an `ErrorEvent` (with the message persisted with `finish_reason="error"`) followed by `StreamEndEvent`.
- Unexpected exceptions yield an `ErrorEvent` with the exception type and message.
- Cancellation (`pipeline_task.cancel()` in `finally`) persists the partial message with `finish_reason="cancelled"`.
- `MessageEndEvent` and `StreamEndEvent` always terminate the stream cleanly.

Next: [ID Generation](../id-generation/), how the `message_id` flows through the system.
