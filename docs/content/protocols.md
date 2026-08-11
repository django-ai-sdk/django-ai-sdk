---
title: Protocols
type: docs
weight: 3
---

The SDK separates **how a model runs** from **how messages are formatted on the wire**. Two adapters cover how a model runs; protocol handlers cover the wire format.

- **`Stream`**: streaming chat, built on a Haystack `Pipeline`
- **`Run`**: non-streaming calls with optional structured output
- **Protocol handlers**: convert protocol messages (e.g. Vercel AI SDK) to internal `ChatMessage`s and back

{{< callout type="info" >}}
Contributor? Full API references live in the [Developer Manual](/manual/): [Stream and Run](/manual/stream-and-run/), [Stream Events](/manual/stream-events/), [Protocol Handler](/manual/protocol-handler/), [Protocol Parts](/manual/protocol-parts/).
{{< /callout >}}

```
                    +--------------------- agent.py ---------------------+
Client (SSE) <----> | VercelProtocolHandler |  ChatMessage  |  Stream    |----> Haystack Pipeline
                    | OpenAIProtocolHandler |               |  Run       |----> generator
                    +------------------------------------------------------+
```

## The Intermediate Format: `ChatMessage`

Regardless of protocol, agents talk internally in `django_ai_sdk.common.ChatMessage`. Protocol handlers translate on both sides of the boundary:

- `to_chat_messages(protocol_messages)`: inbound
- `from_chat_messages(chat_messages)`: outbound (history, ratings)
- `sse(adapter, messages)`: format a streamed run into SSE bytes

---

## Stream

`Stream` wraps a Haystack pipeline and emits normalized streaming events. It's the return value of `get_pipeline_adapter()`.

```python
from django_ai_sdk.adapters.base import Stream

stream = Stream(
    pipeline=pipeline,            # haystack.Pipeline (required)
    generator=generator,          # OpenAIChatGenerator (required)
    store=True,                   # persist the assistant message
    storage_adapter=storage,      # where to persist it
    citation_registry=registry,   # citation numbering (RAG)
    suggestion_generator=gen,     # follow-up questions
)
```

`Stream` requires an actual `haystack.Pipeline`: a bare generator isn't enough. The typical pipeline is built by `ToolAgent`:

```python
from django_ai_sdk.pipelines.haystack import ToolAgent, ToolAgentConfig

tool_agent = ToolAgent(
    config=ToolAgentConfig(
        model=self.get_model(),
        system_prompt=self.get_system_prompt(),
        tools=tools,  # list[haystack.Tool]
    ),
    generator=generator,
)
pipeline = tool_agent.pipeline()
```

Any Haystack pipeline works: the only requirement is that it accepts `messages` and streams `StreamingChunk`s through a callback. If the pipeline's first component is a Haystack `Agent`, `Stream` detects it and runs it directly (handling tool loops), otherwise it runs the pipeline as-is.

### Streaming lifecycle

`await stream.stream(messages)` is an async generator of `StreamEvent`s:

1. **`MessageStartEvent`**: carries the `message_id`, generated **once** here as a UUID
2. **`TextChunkEvent`**: one per chunk from the model
3. **`ToolCallStartEvent`** / **`ToolInputCompleteEvent`** / **`ToolOutputEvent`**: tool calls as they happen
4. **`SourceEvent`**: RAG sources, streamed with citation indices
5. **`MessageEndEvent`**: after the message is finalized
6. **`SuggestionEvent`**: optional follow-up suggestions
7. **`StreamEndEvent`**: the run is done
8. **`ErrorEvent`**: pipeline failures, instead of a hanging stream

Events carry structured data, so protocol handlers render them into their own format.

{{< callout type="info" >}}
When `store=True` and a `storage_adapter` is set, a `StreamWriter` accumulates chunks into a `ChatMessage` and persists it with the same `message_id` the frontend saw: a single ID through SSE, storage, and later operations like rating or deleting the message.
{{< /callout >}}

### Running a Stream directly

You don't usually call `stream()` yourself: `agent.as_view()` does. But it's a plain async generator, so it works anywhere:

```python
async for event in stream.stream(messages):
    if isinstance(event, TextChunkEvent):
        print(event.content)
```

---

## Run

`Run` wraps a generator for **non-streaming** calls. It's the return value of `get_run_adapter()` and powers `agent.run()`: title generation, structured extraction, background jobs.

```python
from django_ai_sdk.adapters.base import Run

run = Run(generator=generator)
reply = await run.run(messages)                # -> str | None
```

### Structured output

Pass a Pydantic model as `response_format` and the reply comes back as an instance:

```python
from pydantic import BaseModel

class Movie(BaseModel):
    title: str
    year: int

movie = await run.run(messages, response_format=Movie)
```

`Run` sends the model's JSON schema as a `response_format` on the OpenAI-compatible generator, then validates the reply. The generator must support JSON-schema responses (any `OpenAIChatGenerator`-compatible endpoint does).

---

## Protocol Handlers

Protocol handlers translate between your frontend's wire format and internal `ChatMessage`s. `Agent.as_view()` uses `agent.protocol_handler`, set from the class-level `protocol` attribute.

### VercelProtocolHandler (default)

Agents default to `VercelProtocolHandler`, which implements the [Vercel AI SDK Data Stream Protocol](https://sdk.vercel.ai/docs/ai-sdk-ui/stream-protocol). Any Vercel-compatible frontend (`@ai-sdk/react`, `useChat`) works without backend changes.

Inbound (`to_chat_messages`) parses Vercel `Message` objects; outbound (`from_chat_messages`) renders `ChatMessage`s back to that shape for history loads. Streams are emitted as SSE:

```
data: {"type":"start","messageId":"msg_abc123"}
data: {"type":"text-start","id":"text_001"}
data: {"type":"text-delta","id":"text_001","delta":"Hello"}
data: {"type":"tool-input-start","toolCallId":"call_1","toolName":"get_today"}
data: {"type":"finish","finishReason":"stop"}
data: [DONE]
```

```python
class MyAgent(Agent):
    protocol = VercelProtocolHandler
```

### OpenAIProtocolHandler

`OpenAIProtocolHandler` implements the OpenAI Chat Completions streaming format, useful when your frontend is built around OpenAI's protocol instead of Vercel's.

```python
from django_ai_sdk.protocols.openai import OpenAIProtocolHandler

class MyAgent(Agent):
    protocol = OpenAIProtocolHandler
```

### Custom handlers

Subclass `BaseProtocolHandler` and implement the three methods:

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

`StreamEvent`s are your inputs for `sse()`: render them into whatever stream format your frontend speaks. Assign the handler to `protocol` on your agent and every view picks it up automatically.

---

## `stream_response`

`django_ai_sdk.responses.stream_response` is the generic helper that turns any streamable adapter into an SSE `StreamingHttpResponse`:

```python
from django_ai_sdk.responses import stream_response

response = await stream_response(
    adapter,                 # Stream or an async factory returning one
    messages,                # list[ChatMessage]
    protocol_handler,        # e.g. VercelProtocolHandler()
)
```

`agent.as_view()` uses this internally. Use it directly when you want a custom view that assembles its own adapter, for example an endpoint that builds a one-off pipeline per request.
