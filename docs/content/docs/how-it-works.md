---
title: How It Works
type: docs
prev: protocols-and-adapters
weight: 6
---

This page explains what happens under the hood when a request flows through the SDK. You don't need to know this to use it, but it helps if you're extending or debugging things.

## Request Lifecycle

When a client sends a chat message, here's what happens:

```
1. Django view receives POST with messages
2. View calls: assistant.as_view(payload.messages, thread_id=...)
3. Protocol handler converts messages:  VercelProtocolHandler.to_chat_messages()
4. If thread_id provided:
   - Creates DbStorageAdapter for the thread
   - Stores the user message immediately
5. Creates adapter:  get_pipeline_adapter(storage_adapter=...)
6. Calls stream_response(adapter, messages, protocol_handler)
7. Returns StreamingHttpResponse with SSE headers
8. Client starts receiving events as they stream
```

Inside the streaming response:

```
9.  protocol_handler.sse(adapter, messages) starts iterating
10. adapter.stream(messages) yields StreamEvent objects
11. Protocol handler converts each event to a StreamChunk (Vercel format)
12. Each chunk is serialized as SSE: "data: {...}\n\n"
13. StreamWriter aggregates chunks into a complete ChatMessage
14. On completion, StreamWriter.finalize() stores the assistant message
15. Stream terminates with "data: [DONE]\n\n"
```

## ChatMessage

`ChatMessage` is the single message type that flows through the whole system. It starts simple (just `role` and `content`) and gets richer as it moves through the pipeline:

```python
from django_ai_sdk.common import ChatMessage

# Input: simple
user_msg = ChatMessage(role="user", content="Tell me a joke")

# After streaming: rich with metadata
assistant_msg = ChatMessage(
    role="assistant",
    content="Why don't pirates shower before...",
    model="gpt-4",
    finish_reason="stop",
    processing_time_ms=1250,
    adapter_type="openai",
    tool_calls=[...],    # if tools were used
    errors=[...],        # if anything went wrong
)
```

There's no separate "input message" vs "storage message" type. The same `ChatMessage` class handles both -- optional fields default to empty/zero.

## Streaming Events

Adapters don't produce protocol-specific output. They yield normalized `StreamEvent` objects that any protocol handler can consume:

```python
# Text streaming
MessageStartEvent(message_id="msg_abc")
TextChunkEvent(content="Hello")
TextChunkEvent(content=" world!")
MessageEndEvent(finish_reason="stop")
StreamEndEvent()

# Tool usage
ToolCallStartEvent(tool_call_id="tool_1", tool_name="weather")
ToolInputCompleteEvent(tool_call_id="tool_1", tool_name="weather", tool_input={"city": "London"})
ToolOutputEvent(tool_call_id="tool_1", tool_output={"temp": "15C"})

# Errors
ErrorEvent(error_message="API timeout")
```

The full set of events: `MessageStartEvent`, `TextChunkEvent`, `ReasoningChunkEvent`, `ToolCallStartEvent`, `ToolInputChunkEvent`, `ToolInputCompleteEvent`, `ToolOutputEvent`, `MessageEndEvent`, `ErrorEvent`, `StreamEndEvent`.

## StreamWriter

When `store=True` (the default for `OpenAIAdapter`), the adapter creates a `StreamWriter` that aggregates streaming chunks into a complete `ChatMessage`:

```python
from django_ai_sdk.common import StreamWriter, MessageChunk
import uuid

writer = StreamWriter(
    adapter_type="openai",
    message_id=str(uuid.uuid4()),  # Required: unique ID for the message
    model="gpt-4",
    storage_callback=storage_adapter.storage_callback,
)

# During streaming, chunks are added:
writer.add_chunk(MessageChunk(type="text", content="Hello"))
writer.add_chunk(MessageChunk(type="text", content=" world"))

# After streaming completes:
complete_message = await writer.finalize("stop")
# -> Stores the complete ChatMessage via the callback
```

`MessageChunk` types: `text`, `reasoning`, `tool_call_start`, `tool_input`, `tool_output`, `error`.

## Storage

### Models

The `django_ai_sdk.conversation` app provides two models:

- **Thread** -- a conversation container. Has `id` (UUID), `title`, `assistant_id`, `model`, `metadata`, timestamps.
- **Message** -- a single message in a thread. Has `id` (UUID), `thread` FK, `result` (JSONField with the serialized `ChatMessage`), `created_at`.

### How Storage Gets Wired

You don't manually store messages. When you pass `thread_id` to `as_view()`:

1. The SDK creates a `DbStorageAdapter(thread_id)`.
2. The user's latest message is stored immediately via `storage_adapter.store_chat_message()`.
3. The storage adapter is passed to `get_pipeline_adapter(storage_adapter=...)`.
4. The adapter sets up a `StreamWriter` with `storage_callback=storage_adapter.storage_callback`.
5. When streaming finishes, `StreamWriter.finalize()` calls the callback, which stores the assistant message.

If you don't pass `thread_id`, no storage happens. Messages stream through and are gone.

### Custom Storage

If you don't want database storage, implement `BaseStorageAdapter`:

```python
from django_ai_sdk.storage.base import BaseStorageAdapter

class FileStorageAdapter(BaseStorageAdapter):
    async def get_thread(self):
        ...
    async def storage_callback(self, chat_message):
        # Store the assistant's completed message however you want
        ...
    async def store_chat_message(self, chat_message):
        # Store individual messages (called for user messages)
        ...
```

Then pass it when creating your assistant:

```python
assistant = MyAssistant(storage_adapter=FileStorageAdapter(thread_id))
```

## The `stream_response()` Function

This is the low-level function that `as_view()` calls. You normally don't use it directly, but it's there if you need custom control:

```python
from django_ai_sdk.responses import stream_response

async def my_custom_view(request):
    adapter = MyAdapter(...)
    messages = [ChatMessage(role="user", content="Hello")]

    return await stream_response(
        adapter=adapter,
        messages=messages,
        protocol_handler=VercelProtocolHandler(),
    )
```

It creates a `StreamingHttpResponse` with `content_type="text/event-stream"` and sets CORS and Vercel protocol headers.
