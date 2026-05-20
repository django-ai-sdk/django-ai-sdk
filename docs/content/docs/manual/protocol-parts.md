---
title: Protocol Parts Reference
type: docs
weight: 108
---

Complete reference for all Vercel AI SDK Data Stream Protocol parts supported by the Django AI SDK. This is primarily for contributors working on protocol handlers or adapters.

## Overview

The protocol handler converts internal `StreamEvent` objects to wire-format protocol parts. This page documents all parts the SDK supports, including those not commonly used by application developers.

## Protocol Part Types

### Message Parts

Control the message lifecycle:

| Part | When | JSON Structure |
|------|------|----------------|
| `MessageStartPart` | Stream begins | `{"type":"start","messageId":"msg-123"}` |
| `FinishPart` | Message complete | `{"type":"finish","finishReason":"stop"}` |
| `DonePart` | Stream terminated | `data: [DONE]` |

### Text Parts

Stream text content:

| Part | When | JSON Structure |
|------|------|----------------|
| `TextStartPart` | Text block begins | `{"type":"text-start","id":"text-456"}` |
| `TextDeltaPart` | Text content chunk | `{"type":"text-delta","id":"text-456","delta":"Hello"}` |
| `TextEndPart` | Text block ends | `{"type":"text-end","id":"text-456"}` |

### Reasoning Parts

For models that expose thinking (o1, o3-mini, DeepSeek):

| Part | When | JSON Structure |
|------|------|----------------|
| `ReasoningStartPart` | Reasoning begins | `{"type":"reasoning-start","id":"reason-789"}` |
| `ReasoningDeltaPart` | Reasoning content | `{"type":"reasoning-delta","id":"reason-789","delta":"Let me think..."}` |
| `ReasoningEndPart` | Reasoning ends | `{"type":"reasoning-end","id":"reason-789"}` |

### Tool Parts

Function calling lifecycle:

| Part | When | JSON Structure |
|------|------|----------------|
| `ToolInputStartPart` | Tool call begins | `{"type":"tool-input-start","toolCallId":"tool-1","toolName":"weather"}` |
| `ToolInputDeltaPart` | Incremental args | `{"type":"tool-input-delta","toolCallId":"tool-1","delta":"{\"city\":\"Lon"}` |
| `ToolInputAvailablePart` | Args complete | `{"type":"tool-input-available","toolCallId":"tool-1","toolName":"weather","input":{"city":"London"}}` |
| `ToolOutputAvailablePart` | Tool result | `{"type":"tool-output-available","toolCallId":"tool-1","output":{"temp":"15C"}}` |

### Source Parts

Document and file references:

| Part | When | JSON Structure |
|------|------|----------------|
| `SourceUrlPart` | External URL | `{"type":"source-url","id":"url-1","url":"https://...","title":"API Docs"}` |
| `SourceDocumentPart` | Document ref | `{"type":"source-document","id":"doc-1","documentId":"doc-uuid","title":"Guide"}` |
| `FilePart` | File attachment | `{"type":"file","data":"base64...","mimeType":"image/png"}` |

### Control Parts

Stream control:

| Part | When | JSON Structure |
|------|------|----------------|
| `StartStepPart` | Step begins | `{"type":"start-step","id":"step-1"}` |
| `FinishStepPart` | Step ends | `{"type":"finish-step","id":"step-1"}` |
| `ErrorPart` | Error occurred | `{"type":"error","errorText":"API timeout"}` |
| `AbortPart` | Stream aborted | `{"type":"abort"}` |

### Data Parts

Custom structured data:

| Part | When | JSON Structure |
|------|------|----------------|
| `DataPart` | Custom data | `{"type":"data","value":{"rag_sources":[...]}}` |

## Implementation Details

### Protocol Handler Interface

All protocol handlers implement `BaseProtocolHandler`:

```python
from django_ai_sdk.protocols.base import BaseProtocolHandler

class CustomProtocolHandler(BaseProtocolHandler):
    def to_chat_messages(self, protocol_messages: list) -> list[ChatMessage]:
        """Convert protocol messages to internal format."""
        return [
            ChatMessage(role=msg["role"], content=msg["content"])
            for msg in protocol_messages
        ]
    
    async def sse(self, adapter, messages) -> AsyncGenerator[bytes, None]:
        """Stream events as SSE."""
        async for event in adapter.stream(messages):
            part = self._event_to_part(event)
            yield f"data: {part.json()}\n\n".encode()
        yield b"data: [DONE]\n\n"
    
    def from_chat_messages(self, chat_messages: list[ChatMessage]) -> list[dict]:
        """Convert internal messages back to protocol format."""
        return [
            {
                "role": msg.role,
                "content": msg.content,
                "id": msg.id,
            }
            for msg in chat_messages
        ]
```

### Event to Part Mapping

The Vercel protocol handler maps events to parts:

```python
def _event_to_part(self, event: StreamEvent) -> StreamPart:
    mapping = {
        MessageStartEvent: lambda e: MessageStartPart(message_id=e.message_id),
        TextChunkEvent: lambda e: TextDeltaPart(id=self._current_text_id, delta=e.content),
        ToolCallStartEvent: lambda e: ToolInputStartPart(
            tool_call_id=e.tool_call_id,
            tool_name=e.tool_name
        ),
        ToolInputCompleteEvent: lambda e: ToolInputAvailablePart(
            tool_call_id=e.tool_call_id,
            tool_name=e.tool_name,
            input=e.tool_input
        ),
        ToolOutputEvent: lambda e: ToolOutputAvailablePart(
            tool_call_id=e.tool_call_id,
            output=e.tool_output
        ),
        DataEvent: lambda e: DataPart(value=e.data),
        # ... etc
    }
    
    converter = mapping.get(type(event))
    if converter:
        return converter(event)
    
    raise ValueError(f"Unknown event type: {type(event)}")
```

### DataEvent Usage

`DataEvent` transmits custom structured data through the stream:

```python
# In adapter - emit RAG sources
async for event in self.stream(messages):
    if isinstance(event, DataEvent):
        # Custom data part: {"type":"data","value":{"rag_retrieval":...}}
        yield event
```

**Common uses:**
- RAG source references
- Debug information
- Metadata attachments
- Application-specific data

Example from OpenAIAdapter:

```python
# After RAG retrieval
yield DataEvent(data={
    "rag_retrieval": {
        "query": query,
        "document_count": len(documents),
        "sources": [
            {"id": doc.id, "score": doc.score}
            for doc in documents
        ]
    }
})
```

## SSE Format

All parts are serialized as Server-Sent Events:

```
# Message start
data: {"type":"start","messageId":"msg-abc123"}

# Text chunk
data: {"type":"text-delta","id":"text-456","delta":"Hello"}

# Tool call
data: {"type":"tool-input-start","toolCallId":"tool-1","toolName":"search"}

# Custom data
data: {"type":"data","value":{"key":"value"}}

# Stream end
data: [DONE]
```

**Format rules:**
- Each part on its own line prefixed with `data: `
- Double newline (`\n\n`) separates events
- `[DONE]` marker terminates stream
- Parts are JSON-encoded

## Part Schemas

### MessageStartPart

```python
class MessageStartPart(StreamPart):
    type: Literal["start"] = "start"
    message_id: str  # UUID for tracking
```

### TextDeltaPart

```python
class TextDeltaPart(StreamPart):
    type: Literal["text-delta"] = "text-delta"
    id: str  # Text block ID (consistent for all chunks in block)
    delta: str  # Incremental content
```

### ToolInputAvailablePart

```python
class ToolInputAvailablePart(StreamPart):
    type: Literal["tool-input-available"] = "tool-input-available"
    tool_call_id: str
    tool_name: str
    input: dict  # Complete tool arguments
```

### DataPart

```python
class DataPart(StreamPart):
    type: Literal["data"] = "data"
    value: dict  # Arbitrary JSON-serializable data
```

## Frontend Compatibility

The SDK's protocol output is compatible with `@ai-sdk/react`:

```typescript
// React hook receives parts as-is
const { messages } = useChat();

// Parts converted to message structure
// Tool calls available in message.toolInvocations
// Custom data accessible via message.annotations
```

## Adding New Parts

To extend the protocol with custom parts:

1. **Define the part schema:**

```python
from pydantic import BaseModel
from typing import Literal

class MyCustomPart(BaseModel):
    type: Literal["my-custom"] = "my-custom"
    custom_field: str
```

2. **Add event type:**

```python
@dataclass
class MyCustomEvent(StreamEvent):
    event_type: str = "my_custom"
    custom_field: str
```

3. **Map event to part in protocol handler:**

```python
def _event_to_part(self, event: StreamEvent) -> StreamPart:
    if isinstance(event, MyCustomEvent):
        return MyCustomPart(custom_field=event.custom_field)
    # ... existing mappings
```

4. **Emit from adapter:**

```python
async def stream(self, messages):
    # ... 
    yield MyCustomEvent(custom_field="value")
```

## Best Practices

### 1. Use Standard Parts When Possible

```python
# Good - use standard parts
yield ToolInputAvailablePart(...)

# Avoid - custom parts for standard concepts
yield DataPart(value={"tool_call": ...})  # Use ToolInputAvailablePart instead
```

### 2. Keep DataPart Lightweight

```python
# Good - summary data
yield DataEvent(data={"sources_count": 5})

# Avoid - full documents
yield DataEvent(data={"full_document": very_large_doc})
```

### 3. Consistent ID Generation

```python
# All parts for same entity share ID
text_id = str(uuid.uuid4())
yield TextStartPart(id=text_id)
yield TextDeltaPart(id=text_id, delta="Hello")
yield TextEndPart(id=text_id)
```

## Testing Protocol Output

```python
import pytest

@pytest.mark.asyncio
async def test_protocol_parts():
    handler = VercelProtocolHandler()
    adapter = MockAdapter()
    
    parts = []
    async for chunk in handler.sse(adapter, []):
        parts.append(chunk.decode())
    
    # Verify structure
    assert 'data: {"type":"start"' in parts[0]
    assert 'data: {"type":"finish"' in parts[-2]
    assert 'data: [DONE]' in parts[-1]
```

## References

- [Vercel AI SDK Data Stream Protocol](https://sdk.vercel.ai/docs/ai-sdk-ui/streaming-data)
- [Architecture Guide](architecture/) — Component interactions
- [Adapters Guide](adapters/) — Event emission
