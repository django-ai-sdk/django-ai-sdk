---
title: Stream and Run
sidebar:
  order: 104
---

The two adapters an `Agent` uses to talk to a Haystack model: `Stream` (streaming chat) and `Run` (non-streaming calls).

:::note
For usage, see the [Protocols guide](../serving/protocols.md). This page is the API reference.
:::

::image[Adapter Flow]{src="../images/graphs/adapter_flow.svg" caption="An agent's two adapters: Stream and Run."}

| Adapter | Purpose | Powered by |
| --- | --- | --- |
| `Stream` | Streaming chat | A `haystack.Pipeline` (usually a `ToolAgent`) |
| `Run` | Non-streaming calls | A generator with `.run()` |

An agent returns one of each from `get_pipeline_adapter()` and `get_run_adapter()`.

## Run

`Run` wraps a generator for single-shot calls. It converts internal `ChatMessage`s to Haystack messages, runs the generator, and returns one reply.

```python
Run(
    generator: Any,        # Haystack chat generator with .run(), e.g. self.get_llm()
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
        return Run(generator=self.get_llm())

    # agent.run(messages, response_format=Movie) -> Movie
    # agent.run(messages) -> str
```

`Run.get_messages()` keeps only `user` and `assistant` messages and converts them to Haystack `ChatMessage`s. A `system_prompt` passed to `run()` goes first as a system message.

:::details[Earlier shapes of Run]
`Run` wraps a plain generator and has never taken tools. Two earlier shapes
added them, `Run(agent=prebuilt_agent)` and then `Run(generator, tools=[...])`.
Both were deleted because no code called them. The second also built an
*ungoverned* Haystack Agent from a generic step-cap setting instead of the
calling agent's own limits. A subagent class run headless would have lost its
own `max_agent_steps` without any error. [Headless tool runs](#headless-tool-runs)
below describes the shape that replaced them.
:::

## Headless tool runs

::image[Headless Tool Run Flow]{src="../images/graphs/headless_tool_run_flow.svg" caption="agent.run(tools=True): one call, one final answer."}

An `Agent` can run its own tools to completion **without streaming**. There is
no `Stream`, no SSE connection and no client reading chunks. One call returns
one final answer. Use it for a workflow step, a scheduled job or a management
command: code that needs tool access while nobody watches a live response.

```python
answer = await agent.run(messages, tools=True)
```

`Agent.run()` takes the headless path only when both conditions hold:

```python
resolved = self.response_format if response_format is self._UNSET else response_format
with_tools = tools and resolved is None
```

- **`tools` is `True`.** This is opt-in. Resolving tools calls `self.get_tools()`, which reaches every configured integration. A one-shot call, such as title generation or structured extraction, pays that cost only when it asks for tools.
- **`resolved is None`.** No `response_format` is set. The SDK does not support structured output and a tool loop together, see [Structured output from a tool loop](generators.md#structured-output-from-a-tool-loop). If both are set, `response_format` wins without a warning and tools are never resolved.

`get_run_adapter()` must return a `Run`. Oversight runs as hooks on its Haystack Agent, so another adapter would run unchecked, and `Agent.run()` raises `TypeError` for it. `Agent.run()` then hands the loop to `Run`, which runs it as a Haystack Agent:

```python
return await adapter.run(
    sent,
    rendered_prompt,
    response_format=resolved,
    tools=await self.get_tools(thread_id=thread_id or "", user=user) if with_tools else None,
    max_agent_steps=self.max_agent_steps if with_tools else None,
    hooks=default_hooks(self),
    request=TurnRequest(agent=self, channel=channel, user=user, thread_id=thread_id, ...),
)
```

`build_subagent()` uses **the same assembly** to turn a subagent into a
Haystack `Agent` for streamed delegation, see [Subagent
Delegation](agent.md#subagent-delegation). `default_hooks(self)` and
`max_agent_steps=self.max_agent_steps` come from the calling agent, not from a
generic constant. A class with `max_agent_steps = 4` gets 4 steps when it is
delegated to live and when it runs headless. The assembly exists in one
function, `default_hooks()`.

The return value is the last message's text. There is no digest and no
synthesis pass when the step cap cuts the loop short. `subagent_response()`
does that for the coordinator and subagent handoff, where a lost result means
a lost answer to the user in the middle of a conversation. A headless run has
no ongoing conversation, so the caller gets the last message's text whether
the loop finished or not.

:::caution
No shipped code calls `agent.run(tools=True)` yet, in `django_ai_sdk` or in
the [studio demo](https://github.com/django-ai-sdk/studio). Tests cover the mechanism, but there is no example to copy.
A workflow step that needs tool access is the likely first caller.
:::

Under an [oversight](oversight.md#agents-and-runs) policy, `Agent.run()` is one
turn. Oversight runs inside the Haystack Agent as hooks from
`django_ai_sdk.oversight.plugin`, and `run_with_oversight` adds them on every run.
The `TurnRequest` says whose turn it is. `run_with_oversight` opens its turn, then:

::::trace[An overseen run]
:::step[GateQuestionHook]{icon="eye"}
Judges the history and the question before the first model call.
:::

:::step[JudgeToolCallHook]{icon="wrench"}
Checks every tool call before it runs.
:::

:::step[MaskToolResultHook]{icon="text-cursor-input"}
Masks every result before the model reads it.
:::

:::step[JudgeAnswerHook]{icon="scale"}
Judges what the loop wrote before the answer is returned.
:::
::::

A refusal raises `django_ai_sdk.errors.Blocked`.

[Limitations](../oversight/oversight.md#limitations) lists what oversight does not
guard, such as `stream_response()` from your own view or an adapter other than `Stream`.

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

:::caution
`pipeline` must be a `haystack.Pipeline`. Anything else raises `TypeError`. A `ToolAgent` returns its pipeline from `tool_agent.pipeline()`.
:::

### Pipeline vs agent component

If the pipeline's first component is a Haystack `Agent`, `Stream` calls `agent.run_async(...)` with the streaming callback. Otherwise it runs `pipeline.run_async({"messages": ...})`. Both paths put chunks on a queue, and `Stream.get_events()` reads them as events.

### `stream()`

`stream()` is an async generator:

```python
async for event in stream.stream(messages: list[ChatMessage]):
    ...  # handle StreamEvent
```

`stream()` runs these steps in order:

::::trace[stream()]
:::step[Convert ChatMessages to Haystack messages]{icon="braces"}
`merge_messages` controls whether consecutive messages merge.
:::

:::step[Generate the message_id UUID exactly once]{icon="key-round"}
:::

:::step[Create a StreamWriter]{icon="database"}
When `store` and a `storage_adapter` are set.
:::

:::step[Attach the streaming callback to the generator]{icon="zap"}
And call `warm_up` if the generator has it.
:::

:::step[Yield the events]{icon="message-square-text"}
`MessageStartEvent`, live events, `MessageEndEvent`, optional `SuggestionEvent`, then `StreamEndEvent`.
:::
::::

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

### Stream a bare pipeline

```python
pipeline = Pipeline()
pipeline.add_component("prompt", ChatPromptBuilder(template=[...]))
pipeline.add_component("llm", generator)
pipeline.connect("prompt.prompt", "llm.messages")

stream = Stream(pipeline=pipeline, generator=generator)
async for event in stream.stream(messages):
    ...
```

Next: [Stream Events](stream-events.md), what `stream()` yields.
