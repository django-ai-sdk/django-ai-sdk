---
title: Protocol Handler
type: docs
weight: 107
---

The interface that converts internal events to your frontend's wire format.

{{< callout type="info" >}}
Usage first? See the [Protocols guide](/docs/protocols/). This page is the API reference.
{{< /callout >}}

## Interface

`BaseProtocolHandler` (`django_ai_sdk.protocols.base`) defines three methods:

| Method | Direction | Purpose |
| --- | --- | --- |
| `to_chat_messages(protocol_messages)` | Inbound | Frontend format → internal `ChatMessage`s |
| `from_chat_messages(chat_messages)` | Outbound | `ChatMessage`s → frontend format (history, ratings) |
| `sse(adapter, messages)` | Stream | Format a streamed run into SSE bytes |

Agents instantiate the handler from the class-level `protocol` attribute (`Agent.__init__` calls `self.protocol()` → `self.protocol_handler`).

## VercelProtocolHandler (default)

Implements the [Vercel AI SDK Data Stream Protocol](https://sdk.vercel.ai/docs/ai-sdk-ui/stream-protocol), so any Vercel-compatible frontend (`@ai-sdk/react`'s `useChat`) works unchanged:

```python
class MyAgent(Agent):
    protocol = VercelProtocolHandler
```

## OpenAIProtocolHandler

Implements the OpenAI Chat Completions streaming format, useful when your frontend is built around OpenAI's protocol:

```python
from django_ai_sdk.protocols.openai import OpenAIProtocolHandler

class MyAgent(Agent):
    protocol = OpenAIProtocolHandler
```

## Custom Handler

Subclass `BaseProtocolHandler` and implement the three methods. `StreamEvent`s are your inputs for `sse()`: render them into whatever stream format your frontend speaks:

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

Assign the handler to `protocol` and every view picks it up automatically.

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

`agent.as_view()` uses this internally. Use it directly when a custom view assembles its own adapter, for example a one-off pipeline per request.

Next: [Protocol Parts](../protocol-parts/), the wire format in detail.
