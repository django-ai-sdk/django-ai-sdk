---
title: Protocol Parts
type: docs
weight: 108
---

Wire-format reference for protocol handlers. Primarily for contributors working on handlers or adapters.

Protocol handlers convert internal `StreamEvent`s into wire-format parts. `VercelProtocolHandler` implements the **Vercel AI SDK Data Stream Protocol**; `OpenAIProtocolHandler` implements the OpenAI chat-completions streaming format. All Vercel parts are Pydantic models in `django_ai_sdk.protocols.vercel`, serialized as SSE `data:` lines with `exclude_none=True, by_alias=True`.

## Message lifecycle

| Part | When | JSON |
| --- | --- | --- |
| `MessageStartPart` | Stream begins | `{"type":"start","messageId":"msg-123"}` |
| `FinishPart` | Message complete | `{"type":"finish","finishReason":"stop"}` |
| `DonePart` | Stream terminated | `data: [DONE]` |

## Text blocks

| Part | When | JSON |
| --- | --- | --- |
| `TextStartPart` | Text block begins | `{"type":"text-start","id":"text-456"}` |
| `TextDeltaPart` | Text content | `{"type":"text-delta","id":"text-456","delta":"Hello"}` |
| `TextEndPart` | Text block ends | `{"type":"text-end","id":"text-456"}` |

The `id` is generated once per block when the first text chunk arrives, so every delta in a block shares an `id`.

## Reasoning blocks

For models that expose thinking:

| Part | When | JSON |
| --- | --- | --- |
| `ReasoningStartPart` | Reasoning begins | `{"type":"reasoning-start","id":"reason-789"}` |
| `ReasoningDeltaPart` | Reasoning content | `{"type":"reasoning-delta","id":"reason-789","delta":"Let me think..."}` |
| `ReasoningEndPart` | Reasoning ends | `{"type":"reasoning-end","id":"reason-789"}` |

A reasoning block opens on the first `reasoning_chunk` event and closes when text starts or the message ends.

## Tool calls

| Part | When | JSON |
| --- | --- | --- |
| `ToolInputStartPart` | Tool call begins | `{"type":"tool-input-start","toolCallId":"tool-1","toolName":"weather"}` |
| `ToolInputDeltaPart` | Incremental args | `{"type":"tool-input-delta","toolCallId":"tool-1","inputTextDelta":"{\"city\":\"Lon"}` |
| `ToolInputAvailablePart` | Args complete | `{"type":"tool-input-available","toolCallId":"tool-1","toolName":"weather","input":{"city":"London"}}` |
| `ToolOutputAvailablePart` | Tool result | `{"type":"tool-output-available","toolCallId":"tool-1","output":{"temp":"15C"}}` |

`ToolOutputAvailablePart.output` is unwrapped from Haystack's `{result, origin, error}` envelope so the frontend receives the actual return value.

## Sources and files

| Part | When | JSON |
| --- | --- | --- |
| `SourceUrlPart` | External URL | `{"type":"source-url","sourceId":"url-1","url":"https://..."}` |
| `SourceDocumentPart` | RAG document | `{"type":"source-document","sourceId":"doc-1","mediaType":"file","title":"Guide","providerMetadata":{"citation":{"index":1}}}` |
| `FilePart` | File attachment | `{"type":"file","url":"...","mediaType":"image/png"}` |

`SourceDocumentPart` carries the citation index in `providerMetadata.citation.index` so clients map inline citations to sources by index rather than emission order.

## Steps and control

| Part | When | JSON |
| --- | --- | --- |
| `StartStepPart` | Step begins | `{"type":"start-step"}` |
| `FinishStepPart` | Step ends | `{"type":"finish-step"}` |
| `ErrorPart` | Error occurred | `{"type":"error","errorText":"API timeout"}` |
| `AbortPart` | Stream aborted | `{"type":"abort","reason":"..."}` |

## Custom data

| Part | When | JSON |
| --- | --- | --- |
| `DataPart` | Custom structured data | `{"type":"data-suggestions","data":{"suggestions":["..."]}}` |

`DataPart.type` is dynamic and must start with `data-`. Suggestions are emitted as `data-suggestions`.

## Event → Part Mapping

`VercelProtocolHandler.handle_stream()` maps events to parts:

| Event | Part(s) |
| --- | --- |
| `MessageStartEvent` | `MessageStartPart` (requires `message_id`) |
| `ReasoningChunkEvent` | `ReasoningStartPart` (once) + `ReasoningDeltaPart` per chunk |
| `TextChunkEvent` | `TextStartPart` (once) + `TextDeltaPart` per chunk |
| `ToolCallStartEvent` | `ToolInputStartPart` |
| `ToolInputCompleteEvent` | `ToolInputAvailablePart` |
| `ToolOutputEvent` | `ToolOutputAvailablePart` |
| `DataEvent` | `DataPart(type="data-{data_type}")` |
| `SuggestionEvent` | `DataPart(type="data-suggestions")` |
| `SourceEvent` | `SourceDocumentPart` with citation index |
| `ErrorEvent` | `ErrorPart` |
| `MessageEndEvent` | close open reasoning/text blocks + `FinishPart` |
| `StreamEndEvent` | `DonePart` |

If the stream is interrupted, `sse()` emits closing `ReasoningEndPart` / `TextEndPart` for any open blocks before terminating.

## SSE Format

`format_sse()` (in `django_ai_sdk.protocols.utils`) wraps each payload:

```
data: {"type":"start","messageId":"msg-abc123"}

data: {"type":"text-start","id":"text-456"}

data: {"type":"text-delta","id":"text-456","delta":"Hello"}

data: {"type":"text-end","id":"text-456"}

data: {"type":"finish","finishReason":"stop"}

data: [DONE]
```

Rules: each part on its own `data:` line (JSON with `ensure_ascii=False`); `\n\n` separates events; `[DONE]` terminates the stream.

## Message Object Format

`from_chat_messages()` converts stored `ChatMessage`s back into the Vercel message shape (used for history endpoints):

```json
{
  "id": "msg-123",
  "role": "assistant",
  "parts": [
    {"type": "text", "text": "Hello!"},
    {"type": "source-document", "sourceId": "doc-1", "mediaType": "file",
     "title": "Guide", "providerMetadata": {"citation": {"index": 1}}}
  ],
  "finish_reason": "stop",
  "tool_calls": [],
  "processing_time_ms": 1250,
  "has_errors": false,
  "feedbacks": [],
  "created_at": "..."
}
```

## OpenAI Protocol Handler

`OpenAIProtocolHandler` (in `django_ai_sdk.protocols.openai`) emits OpenAI-compatible `chat.completion.chunk` SSE lines instead of Data Stream Protocol parts, buffering text and tool calls into delta objects and terminating with `data: [DONE]`.

## References

- [Vercel AI SDK Data Stream Protocol](https://ai-sdk.dev/docs/ai-sdk-ui/streaming-protocol)
- [Stream and Run](../stream-and-run/): event emission
- [Protocol Handler](../protocol-handler/): the handler interface
