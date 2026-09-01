---
title: Tracing
type: docs
weight: 122
---

Every Haystack span of a run — the agent, each step, each LLM call, each tool call — persisted as a `Trace` row and linked to the `Thread` and `Message` it was produced for. Opt-in, the way Haystack itself treats tracing: installing the SDK changes nothing until you add the app.

## Enabling

```python
INSTALLED_APPS = [
    "django_ai_sdk",
    "django_ai_sdk.tracing",
]
```

That is the whole switch. The app ships the `Trace` model plus its migration, and its `AppConfig.ready()` enables the tracer at process startup.

Manual control, if you would rather not use the AppConfig:

```python
from haystack import tracing
from django_ai_sdk.tracing.tracer import DefaultTracer

tracing.enable_tracing(DefaultTracer())
```

`enable_tracing()` swaps a field on Haystack's process-global `ProxyTracer`, which pipelines resolve at run time — so it may be called after your pipelines are built, but it must be called once at startup, never per request. Last call wins.

## What a span records

| Column | Notes |
| --- | --- |
| `operation_name` | The Haystack operation, e.g. `haystack.agent.step.llm`. |
| `started_at` / `ended_at` / `duration_ms` | Wall-clock bounds, plus a monotonic duration. |
| `parent` | Self-FK; the tree of one run. |
| `thread` / `message` | Correlation, `db_constraint=False` (see below). |
| `model_name`, `prompt_tokens`, `completion_tokens`, `total_tokens` | Harvested from the LLM call's metadata. |
| `tags` | The span's Haystack tags, coerced to JSON. |

Indexed on `(operation_name, started_at)` and `(thread, started_at)`; the FK columns carry Django's own indexes.

### Content and sensitive data

Prompts and replies are **not** stored unless `HAYSTACK_CONTENT_TRACING_ENABLED=true`, which puts the content of queries, documents and answers into the `tags` column. Treat that data as sensitive, and note that deleting a `Thread` or `Message` cascades to its traces.

Token counts and `model_name` are recorded either way. Haystack reports usage through the same content tags it gates, so `TelemetrySpan.set_content_tag()` harvests the numbers before delegating to the base implementation, which decides whether the payload itself is kept.

That environment variable is read **once**, when `haystack.tracing` is first imported (`ProxyTracer.__init__`). It must be a real environment variable set before the process starts; assigning it in `settings.py` does nothing, and changing it needs a restart.

### Dropping bulky tags

Some tags are static configuration rather than observability data, and they are large: `haystack.agent.tools` serializes every tool definition in full, and repeats byte-for-byte on every run. Drop them by key:

```python
AI_SDK_TRACING_EXCLUDED_TAGS = [
    "haystack.agent.tools",
    "haystack.agent.state_schema",
]
```

Exact keys only, no wildcards. The check runs before tag coercion, so an excluded tag costs nothing to serialize. It applies to content tags too — you can keep content tracing on while dropping one payload — and excluding a content key still records its token counts.

## Correlation

Runs streamed through the SDK are correlated automatically: `Stream.stream()` binds the assistant `Message` it mints and the `Thread` its storage adapter is bound to, around the `asyncio.create_task` that starts the pipeline, so the contextvar is captured in the task's context.

For pipelines you run yourself:

```python
from django_ai_sdk.tracing import bind

with bind(thread_id=thread.id, message_id=message.id):
    await pipeline.run_async({"messages": messages})
```

Ids are coerced to `uuid.UUID` and silently skipped when they don't parse — correlation is observability and must never fail the request carrying it. Nested binds merge with the enclosing one.

The contextvar is read **once, on the root span**. Child spans inherit the ids from their parent span object, so correlation survives Haystack running sync components in `asyncio.to_thread` workers, where contextvar propagation is not something to rely on.

## How writes work

Haystack's `Tracer.trace` is a sync context manager even under `Pipeline.run_async` — Haystack enters it directly on the event loop thread — so a plain ORM call there would raise `SynchronousOnlyOperation`. Rather than writing per span, the tracer buffers:

- `Trace` uses a client-generated UUID primary key, so a child span knows its `parent_id` before anything is written.
- Every span in a tree is buffered on its root span, and the root's exit writes the whole tree in **one** `bulk_create` — `abulk_create` scheduled on the loop when one is running, inline otherwise. Rows are in creation order, so parents precede children and the self-FK holds inside the single statement.
- A failed write is logged, never raised. Tracing must not break the run it observes.

Two consequences worth knowing:

**Spans appear only once the run completes.** A hung or still-running pipeline has no rows yet; this is not a live view.

**The write is fire-and-forget on the loop.** `await aflush()` is the barrier when you need one — tests, or shutdown:

```python
from django_ai_sdk.tracing import aflush

await aflush()
```

The `thread` and `message` foreign keys are declared `db_constraint=False` because a run's spans are written when the run ends, which is *before* the streaming adapter persists the assistant `Message` they were correlated with. Deletes still cascade: Django collects those in the ORM, not the database.

## Token accounting

Usage is recorded on every span that wraps an LLM call **and** aggregated again onto the agent's own `haystack.agent.run` span, from Haystack's `token_usage` state. So a naive `Sum` over the token columns double-counts every call — triple, for an Agent running inside a Pipeline, whose component span carries the same rollup.

The rule is to count **leaf spans only**, which are exactly the LLM calls in every execution path and never a rollup — a rollup by definition has the spans it aggregates as children:

| Span | Children | Tokens | Counted |
| --- | --- | --- | --- |
| `haystack.agent.run` | yes | rollup | no |
| `haystack.agent.step` | yes | null | no |
| `haystack.agent.step.llm` | no | per call | **yes** |
| `haystack.component.run` (chat generator) | no | per call | **yes** |
| `haystack.component.run` (wrapping an Agent) | yes | rollup | no |
| `haystack.agent.step.tool` | no | null | yes, contributes 0 |

`TraceQuerySet.llm_calls()` implements it, and both the service and the manager's `token_usage()` already count that way. The rollup row stays available when you want a run total from a single read.

### Provider caveat

OpenAI omits usage entirely from a *streamed* response unless `stream_options.include_usage` is requested. The `Stream` adapter sets that default on OpenAI-compatible generators at construction; a custom streaming generator that reports no usage leaves the token columns null.

## TraceService

The permission-checked entry point. A trace belongs to its thread, so every method resolves the thread through `ThreadService` — enforcing `VIEW_THREAD` and raising `PermissionDenied` — and raises `ValueError` when the thread or message doesn't exist. There is no separate trace permission domain.

```python
from django_ai_sdk.tracing.services import TraceService

await TraceService.thread_traces(thread_id, user=user, message_id=None,
                                 operation_name=None, limit=100, offset=0)
await TraceService.message_traces(message_id, user=user)
await TraceService.thread_token_usage(thread_id, user=user)
await TraceService.message_token_usage(message_id, user=user)
```

Message-scoped calls resolve the owning thread themselves, so a caller holding only a message id doesn't need to know its thread. All four return pydantic schemas from `django_ai_sdk.tracing.schemas` — `TraceOut` and `TokenUsage`.

For sync contexts such as DRF class-based views, import the module-level aliases instead:

```python
from django_ai_sdk.tracing.services import thread_traces, thread_token_usage
```

## Manager

For direct ORM work, where you are doing your own access control:

```python
Trace.objects.for_thread(thread.id).token_usage()
# {'prompt_tokens': 1841, 'completion_tokens': 317, 'total_tokens': 2158}

await Trace.objects.for_message(message.id).atoken_usage()

Trace.objects.for_message(message.id).roots()       # the top span of the run
Trace.objects.llm_calls()                           # one row per LLM call
```

Every helper is a `QuerySet` method, so they chain after any filter:

```python
from django.db.models import Sum
from django.db.models.functions import TruncDay

Trace.objects.llm_calls().exclude(total_tokens__isnull=True).annotate(
    day=TruncDay("started_at")
).values("day", "model_name").annotate(tokens=Sum("total_tokens")).order_by("day")
```

Next: [Settings Reference](../settings/), for the tracing settings and every other `AI_SDK_*` value.
