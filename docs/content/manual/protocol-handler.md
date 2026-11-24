---
title: Protocol Handler
sidebar:
  order: 107
---

The interface that converts internal events into your frontend's wire format.

:::note
Usage first? See the [Protocols guide](../serving/protocols.md). This page is the API reference.
:::

## Interface

`BaseProtocolHandler` (`django_ai_sdk.protocols.base`) defines three methods:

| Method | Direction | Purpose |
| --- | --- | --- |
| `to_chat_messages(protocol_messages)` | Inbound | Frontend format to internal `ChatMessage`s |
| `from_chat_messages(chat_messages)` | Outbound | `ChatMessage`s to frontend format, for history and ratings |
| `sse(adapter, messages)` | Stream | Format a streamed run into SSE bytes |

Agents instantiate the handler from the class-level `protocol` attribute. `Agent.__init__` calls `self.protocol()` and stores the result in `self.protocol_handler`.

## VercelProtocolHandler (default)

Implements the [Vercel AI SDK Data Stream Protocol](https://sdk.vercel.ai/docs/ai-sdk-ui/stream-protocol), so any Vercel-compatible frontend, such as `useChat` from `@ai-sdk/react`, works unchanged:

```python
class MyAgent(Agent):
    protocol = VercelProtocolHandler
```

## OpenAIProtocolHandler

Implements the OpenAI Chat Completions streaming format. Use it when your frontend expects OpenAI's protocol:

```python
from django_ai_sdk.protocols.openai import OpenAIProtocolHandler

class MyAgent(Agent):
    protocol = OpenAIProtocolHandler
```

## Custom handler

Subclass `BaseProtocolHandler` and implement the three methods. `sse()` receives `StreamEvent`s and renders them into the stream format your frontend reads:

```python
from django_ai_sdk.protocols.base import BaseProtocolHandler

class MyProtocolHandler(BaseProtocolHandler):
    def to_chat_messages(self, protocol_messages):
        ...

    def from_chat_messages(self, chat_messages):
        ...

    async def sse(self, adapter, messages):
        ...
```

Assign the handler to `protocol` and every view of the agent uses it.

## stream_response

`django_ai_sdk.responses.stream_response` turns any streamable adapter into an SSE `StreamingHttpResponse`:

```python
from django_ai_sdk.responses import stream_response

response = await stream_response(
    adapter,                 # Stream or an async factory returning one
    messages,                # list[ChatMessage]
    protocol_handler,        # e.g. VercelProtocolHandler()
)
```

`agent.as_view()` calls it internally. Call it directly when a custom view builds its own adapter, for example a one-off pipeline per request.

Next: [Protocol Parts](protocol-parts.md), the wire format in detail.
