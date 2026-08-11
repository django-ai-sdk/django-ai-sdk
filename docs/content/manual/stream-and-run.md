---
title: Stream and Run
type: docs
weight: 104
---

The two adapters an `Agent` uses to talk to a Haystack model: `Stream` (streaming chat) and `Run` (non-streaming calls).

{{< callout type="info" >}}
Usage first? See the [Protocols guide](/docs/protocols/). This page is the API reference.
{{< /callout >}}

![Adapter Flow](/images/graphs/adapter_flow.png)

| Adapter | Purpose | Powered by |
| --- | --- | --- |
| `Stream` | Streaming chat | A `haystack.Pipeline` (usually a `ToolAgent`) |
| `Run` | Non-streaming calls | A generator with `.run()` |

An agent exposes one of each via `get_pipeline_adapter()` and `get_run_adapter()`.

## Run

`Run` wraps a generator for single-shot calls. It converts internal `ChatMessage`s to Haystack messages, runs the generator, and returns one reply.

```python
Run(
    generator: Any,        # Haystack chat generator with .run()
    model: str | None = None,
    instructions: str | None = None,
)
```

```python
response = await run.run(
    messages: list[ChatMessage],
    system_prompt: str | None = None,
    response_format: type[T] | None = None,
)  # -> T | str | None
```

- Without `response_format`, returns the reply text as a `str`.
- With `response_format`, passes the Pydantic model's JSON schema as the generation's `response_format` and returns a validated instance.

```python
from pydantic import BaseModel

class Movie(BaseModel):
    title: str
    year: int

class MyAgent(Agent):
    async def get_run_adapter(self, thread_id=None, user=None):
        generator = OpenAIChatGenerator(...)
        return Run(generator=generator)

    # agent.run(messages, response_format=Movie) -> Movie
    # agent.run(messages) -> str
```

`Run.get_messages()` keeps only `user` and `assistant` messages and converts them to Haystack `ChatMessage`s; a `system_prompt` passed to `run()` is prepended as a system message.

## Stream

`Stream` wraps a Haystack pipeline and normalizes its streaming output into `StreamEvent`s.

```python
Stream(
    pipeline: Pipeline,                          # haystack.Pipeline instance
    generator: Any,                              # generator used for streaming
    store: bool = True,                          # persist the reply
    storage_adapter: BaseStorageAdapter | None = None,
    citation_registry: CitationRegistry | None = None,
    suggestion_generator: SuggestionGenerator | None = None,
)
```

{{< callout type="warning" >}}
`pipeline` must be a real `haystack.Pipeline`: anything else raises `TypeError`. A `ToolAgent` exposes its pipeline via `tool_agent.pipeline()`.
{{< /callout >}}

### Pipeline vs Agent Component

If the pipeline's first component is a Haystack `Agent`, `Stream` calls `agent.run_async(...)` directly with the streaming callback. Otherwise it runs `pipeline.run_async({"messages": ...})`. Both paths stream chunks through a queue that `Stream.get_events()` consumes as events.

### `stream()`

`stream()` is an async generator:

```python
async for event in stream.stream(messages: list[ChatMessage]):
    ...  # handle StreamEvent
```

Order of operations:

1. Convert `ChatMessage`s to Haystack messages (`merge_messages` controls consecutive-message merging).
2. Generate the `message_id` UUID exactly once.
3. Create a `StreamWriter` (when `store` and a `storage_adapter` are set).
4. Attach the streaming callback to the generator, warm it up if it has `warm_up`.
5. Yield `MessageStartEvent`, live events, `MessageEndEvent`, optional `SuggestionEvent`, then `StreamEndEvent`.

## Examples

### Stream a ToolAgent

```python
stream = Stream(
    pipeline=tool_agent.pipeline(),
    generator=generator,
    storage_adapter=storage,
    citation_registry=await agent.get_citation_registry(),
    suggestion_generator=agent.suggestion_generator,
)
```

### Stream a Bare Pipeline

```python
pipeline = Pipeline()
pipeline.add_component("prompt", ChatPromptBuilder(template=[...]))
pipeline.add_component("llm", generator)
pipeline.connect("prompt.prompt", "llm.messages")

stream = Stream(pipeline=pipeline, generator=generator)
async for event in stream.stream(messages):
    ...
```

Next: [Stream Events](../stream-events/), what `stream()` yields.
