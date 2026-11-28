---
title: Tracing
sidebar:
  order: 122
---

The tracing app stores every Haystack span of a run as a `Trace` row: the agent, each step, each LLM call and each tool call. Each row links to the `Thread` and `Message` it was produced for. Tracing is opt-in, as in Haystack itself. Installing the SDK changes nothing until you add the app.

## Enabling

```python
INSTALLED_APPS = [
    "django_ai_sdk",
    "django_ai_sdk.tracing",
]
```

Adding the app is the only switch. The app ships the `Trace` model and its migration, and its `AppConfig.ready()` enables the tracer at process startup.

:::details[Enable the tracer without the AppConfig]
To enable the tracer yourself instead of through the AppConfig:

```python
from haystack import tracing
from django_ai_sdk.tracing.tracer import DefaultTracer

tracing.enable_tracing(DefaultTracer())
```

`enable_tracing()` swaps a field on Haystack's process-global `ProxyTracer`, and pipelines resolve that field at run time. You can call it after your pipelines are built. Call it once at startup, never per request. The last call wins.
:::

## What a span records

| Column | Notes |
| --- | --- |
| `operation_name` | The Haystack operation, e.g. `haystack.agent.step.llm`. |
| `started_at` / `ended_at` / `duration_ms` | Wall-clock bounds, plus a monotonic duration. |
| `parent` | Self-FK; the tree of one run. |
| `thread` / `message` | Correlation, `db_constraint=False` (see below). |
| `model_name`, `prompt_tokens`, `completion_tokens`, `total_tokens` | Harvested from the LLM call's metadata. |
| `tags` | The span's Haystack tags, coerced to JSON. |

The table has indexes on `(operation_name, started_at)` and `(thread, started_at)`. The FK columns carry Django's own indexes.

### Content and sensitive data

The tracer stores prompts and replies only when `HAYSTACK_CONTENT_TRACING_ENABLED=true`. That setting puts the content of queries, documents and answers into the `tags` column. Treat that data as sensitive. Deleting a `Thread` or `Message` cascades to its traces.

The tracer records token counts and `model_name` either way. Haystack reports usage through the same content tags it gates. `TelemetrySpan.set_content_tag()` therefore harvests the numbers before it delegates to the base implementation, which decides whether to keep the payload itself.

:::caution
Haystack reads that environment variable **once**, when `haystack.tracing` is first imported (`ProxyTracer.__init__`). It must be a real environment variable, set before the process starts. Assigning it in `settings.py` does nothing, and changing it needs a restart.
:::

### Dropping bulky tags

Some tags hold static configuration, not observability data, and they are large. `haystack.agent.tools` serializes every tool definition in full and repeats it byte for byte on every run. Drop such tags by key:

```python
AI_SDK_TRACING_EXCLUDED_TAGS = [
    "haystack.agent.tools",
    "haystack.agent.state_schema",
]
```

The setting takes exact keys only, no wildcards. The check runs before tag coercion, so an excluded tag costs nothing to serialize. It applies to content tags too, so you can keep content tracing on and drop one payload. Excluding a content key still records its token counts.

## Correlation

The SDK correlates runs it streams. `Stream.stream()` binds the assistant `Message` it mints and the `Thread` of its storage adapter. It binds them around the `asyncio.create_task` that starts the pipeline, so the task's context captures the contextvar.

For pipelines you run yourself:

```python
from django_ai_sdk.tracing import bind

with bind(thread_id=thread.id, message_id=message.id):
    await pipeline.run_async({"messages": messages})
```

`bind()` coerces ids to `uuid.UUID` and skips ids that don't parse, without an error. Correlation is observability, so it must never fail the request that carries it. A nested bind merges with the enclosing one.

The tracer reads the contextvar **once, on the root span**. Child spans inherit the ids from their parent span object. Correlation therefore survives when Haystack runs sync components in `asyncio.to_thread` workers, where contextvar propagation is unreliable.

## How writes work

Haystack's `Tracer.trace` is a sync context manager even under `Pipeline.run_async`. Haystack enters it directly on the event loop thread, so a plain ORM call there would raise `SynchronousOnlyOperation`. The tracer buffers spans instead of writing each one:

- `Trace` uses a client-generated UUID primary key, so a child span knows its `parent_id` before anything is written.
- The tracer buffers every span in a tree on its root span. The root's exit writes the whole tree in **one** `bulk_create`: `abulk_create` scheduled on the loop when one is running, inline otherwise. Rows are in creation order, so parents precede children and the self-FK holds inside the single statement.
- The tracer logs a failed write and never raises it. Tracing must not break the run it observes.

This has two consequences.

:::note[Spans appear only once the run completes.]
A hung or still-running pipeline has no rows yet. The table is not a live view.
:::

**The write is fire-and-forget on the loop.** When you need a barrier, in tests or at shutdown, use `await aflush()`:

```python
from django_ai_sdk.tracing import aflush

await aflush()
```

The `thread` and `message` foreign keys are declared `db_constraint=False`. The tracer writes a run's spans when the run ends, which is *before* the streaming adapter persists the assistant `Message` they belong to. Deletes still cascade, because Django collects them in the ORM, not in the database.

## Token accounting

The tracer records usage on every span that wraps an LLM call. It records it **again** on the agent's own `haystack.agent.run` span, aggregated from Haystack's `token_usage` state. A naive `Sum` over the token columns therefore counts every call twice. For an Agent running inside a Pipeline it counts three times, because the component span carries the same rollup.

The rule is to count **leaf spans only**. Leaf spans are exactly the LLM calls in every execution path and never a rollup, because a rollup has the spans it aggregates as children:

| Span | Children | Tokens | Counted |
| --- | --- | --- | --- |
| `haystack.agent.run` | yes | rollup | no |
| `haystack.agent.step` | yes | null | no |
| `haystack.agent.step.llm` | no | per call | **yes** |
| `haystack.component.run` (chat generator) | no | per call | **yes** |
| `haystack.component.run` (wrapping an Agent) | yes | rollup | no |
| `haystack.agent.step.tool` | no | null | yes, contributes 0 |

`TraceQuerySet.llm_calls()` implements this rule, and both the service and the manager's `token_usage()` use it. The rollup row is still there when you want a run total from a single read.

### Provider caveat

The Responses API reports usage on a streamed response without being asked, so the default `openai_responses_chat` needs no configuration. The tracer accepts its `input_tokens`/`output_tokens` names next to the `prompt_tokens`/`completion_tokens` of Chat Completions. Chat Completions omits usage from a *streamed* response unless the request sets `stream_options.include_usage`. The adapters do not add it, because the option is invalid on a non-streaming call. Declare it on the agent instead, as described in [Generators](generators.md#token-usage-while-streaming). A generator that reports no usage leaves the token columns null.

## TraceService

`TraceService` is the permission-checked entry point. A trace belongs to its thread, so every method resolves the thread through `ThreadService`. That enforces `VIEW_THREAD` and raises `PermissionDenied`. A method raises `ValueError` when the thread or message doesn't exist. Traces have no permission domain of their own.

```python
from django_ai_sdk.tracing.services import TraceService

await TraceService.thread_traces(thread_id, user=user, message_id=None,
                                 operation_name=None, limit=100, offset=0)
await TraceService.message_traces(message_id, user=user)
await TraceService.thread_token_usage(thread_id, user=user)
await TraceService.message_token_usage(message_id, user=user)
```

Both trace listings paginate with `limit`/`offset`. Pass `limit=None` to return every span instead of the newest 100.

Message-scoped calls resolve the owning thread themselves, so a caller with only a message id doesn't need its thread. All four methods return pydantic schemas from `django_ai_sdk.tracing.schemas`: `TraceOut` and `TokenUsage`.

For sync contexts such as DRF class-based views, import the module-level aliases instead:

```python
from django_ai_sdk.tracing.services import thread_traces, thread_token_usage
```

## Manager

Use the manager for direct ORM work, where you do your own access control:

```python
Trace.objects.for_thread(thread.id).token_usage()
# {'prompt_tokens': 1841, 'completion_tokens': 317, 'total_tokens': 2158}

await Trace.objects.for_message(message.id).atoken_usage()

Trace.objects.for_message(message.id).roots()       # the top span of the run
Trace.objects.llm_calls()                           # one row per LLM call
```

Every helper is a `QuerySet` method, so it chains after any filter:

```python
from django.db.models import Sum
from django.db.models.functions import TruncDay

Trace.objects.llm_calls().exclude(total_tokens__isnull=True).annotate(
    day=TruncDay("started_at")
).values("day", "model_name").annotate(tokens=Sum("total_tokens")).order_by("day")
```

## Subagents in a trace

A coordinator and each of its subagents run the same Haystack `Agent`. Both emit `haystack.agent.run`, so a raw tree cannot tell them apart. The SDK therefore wraps every subagent run in one named span and tags it with the subagent's id and display name:

```
haystack.agent.run                    the coordinator
└─ django_ai_sdk.subagent.run         tags: django_ai_sdk.subagent.id = "7f3a2c11-..."
   │                                        django_ai_sdk.subagent.name = "Research Planner"
   └─ haystack.agent.run              the subagent's own loop
      ├─ haystack.agent.step.llm
      └─ haystack.component.run       its tools
```

Nesting comes from the active-span contextvar, so it holds across the `ComponentTool` boundary without extra wiring.

The tracer copies both tags onto indexed `Trace` columns, `agent_id` and `agent_name`, so you don't have to read them from `tags`. **Use `agent_id` for lookups.** It is the subagent's `Agent.agent_id`: a UUID derived from the class's module path, or from the `AgentSettings` row for a `RuntimeAgent`. It stays the same when the display name changes. `agent_name` is a label only. Two subagent classes can share one name, and grouping by it would merge their totals without warning.

`subagent_usage()` returns what each subagent cost, keyed by id:

```python
Trace.objects.for_thread(thread.id).subagent_usage()
# {'7f3a2c11-...': {'agent_name': 'Research Planner', 'prompt_tokens': 8210,
#                    'completion_tokens': 1932, 'total_tokens': 10142},
#  '9c04e8aa-...': {'agent_name': 'Fact Checker', 'prompt_tokens': 412, ...}}

Trace.objects.subagent_ids()                      # ['7f3a2c11-...', '9c04e8aa-...']
Trace.objects.subagent_names()                    # ['Fact Checker', 'Research Planner'], for display only
Trace.objects.subagents("7f3a2c11-...")           # just that subagent's wrapper spans
```

`subagent_usage()` walks each subagent's whole subtree, because the wrapper span carries no tokens. Its LLM calls are descendants. A nested subagent counts under its own id, not under the subagent that delegated to it. The helper excludes a span with no `agent_id`, because there is no stable key to group it under. Such a span was recorded before the column existed or never got tagged. Every helper chains after any filter, like the rest of the queryset, and has an async twin with an `a` prefix.

The service layer returns the per-subagent usage in the same call as the plain total, without a separate round trip. `thread_token_usage()` and `message_token_usage()` set `TokenUsage.by_subagent`, keyed by id, with `agent_name` on each entry for display. They compute it from the same queryset as the top-level totals and enforce the thread's permissions once for both:

```python
usage = await TraceService.thread_token_usage(thread.id, user=request.user)
usage.total_tokens        # 10142: the whole thread, subagents included
usage.by_subagent
# {'7f3a2c11-...': TokenUsage(agent_name='Research Planner', prompt_tokens=8210,
#                              completion_tokens=1932, total_tokens=10142)}
```

The totals count a subagent's tokens twice on purpose: once in its own `by_subagent` entry and once in the outer total. The spans are the same, and the two numbers answer two different questions.

Next: [Settings Reference](settings.md), for the tracing settings and every other `AI_SDK_*` value.
