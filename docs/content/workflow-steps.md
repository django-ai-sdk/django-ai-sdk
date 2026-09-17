---
title: Step Workflows
type: docs
weight: 55
---

A real pipeline is not only agent steps: it fetches a file, converts it, calls an API, validates what came back, writes rows. This page is how you declare those in Python, so a host does not orchestrate around the engine and lose the record of what happened.

There is one runner. A [stored `WorkflowDefinition`](/manual/workflows/) is compiled into the same `Step` objects this page declares and walked by the same `run_steps`, so both read back from one record. What differs is who authors it and what it may reach: a definition is JSON written at runtime, held to the step types the deployment registered; a pipeline is code, free to do work no model is involved in.

A step is a named unit of work that reads the run's state and returns a value. The runner does not interpret what it returns.

## Two namespaces

The run's state has two halves, and nothing crosses between them:

| | |
| --- | --- |
| `ctx.input(name)` | What the caller supplied. Fixed before the first step, never written to. |
| `ctx.step(name)` | What an earlier step produced, keyed by **that step's own name**. |

A step's result is filed under its `name`. There is no `output_key` to choose and no shared namespace to collide in, so an input and a step may go by the same word without either being renamed.

## Declaring steps

Steps run in the order you declare them. `requires` names the steps this one reads: it runs once every one of them has completed.

```python
from django_ai_sdk.workflows import AgentStep, OnError, Step, StepOutcome


class ReadFileStep(Step):
    name = "text"
    on_error = OnError.FAIL

    async def run(self, ctx):
        document = ctx.input("document")
        text = await read_the_file(document)
        if not text.strip():
            return StepOutcome(status="failed", detail="the file yielded no text")
        return StepOutcome(output=text, detail=f"{len(text)} characters")


class SummariseStep(AgentStep):
    name = "summary"
    requires = ("text",)
    on_error = OnError.CONTINUE

    agent = SummaryAgent
    schema = Summary

    async def user_message(self, ctx):
        return ctx.step("text")


PIPELINE = (ReadFileStep(), SummariseStep())
```

A step with several values to hand on returns one structured object; a step whose real result is the rows it wrote returns nothing and the record simply says it completed.

## Running

```python
from django_ai_sdk.workflows import run_steps

outcomes = await run_steps(
    PIPELINE,
    inputs={"document": document},
    principal=request.user,
)
```

`inputs` seeds `ctx.input`. `principal` is the user an `AgentStep` runs as. `outcomes` maps every step's name to its `StepOutcome`.

## Agent steps

`AgentStep` is one class whether the agent is named by import or by id, so a definition's agent step is this class configured rather than a subclass of it:

| | |
| --- | --- |
| `agent` | An `Agent` subclass, for a step written in code. |
| `agent_id` | An id resolved through `AgentService`, for a step compiled from JSON. |
| `schema` | A pydantic model, passed as the agent's `response_format`. Nested output is expressible. A result that does not come back as that model fails the step. |
| `instructions` | A system prompt replacing the agent's own; empty keeps the agent's. |
| `history` | Names of the inputs to send as the conversation — the author names them, per step, whether the step comes from code or JSON. A `str` input is one user turn; a `list` is a conversation of messages, message dicts, or strings. |

It sends the inputs named in `history` as the transcript, then one turn. Override the async `user_message(ctx)` to write that turn, or `messages(ctx)` to write the whole conversation; both may query. By default the turn is what the steps in `requires` produced, each under a `[name]` heading. A `None` result is reported as `failed`, not stored.

The transcript is **named, not guessed**. An input is data until a step names it as `history`, so a list of role-shaped rows that happens to be data is never mistaken for a prompt, and a workflow with two agent steps can give each a different conversation. What a history field holds is the author's choice: a `str` arrives as one user turn, a message dict as one message, a list item-by-item.

## Failure

`on_error` decides what a step's failure does to the rest of the run, and applies whether the step raised or returned `status="failed"`:

| | |
| --- | --- |
| `OnError.FAIL` | The run stops. Every step that will now not run is recorded as `skipped`, then the exception propagates — `StepFailed` when the step reported failure rather than raising. |
| `OnError.CONTINUE` | The failure is recorded and the rest of the run proceeds. |

A failed step produces nothing, and that is the whole cascade: a step whose required step produced nothing is recorded `skipped`, naming what it waited for. So a failure reaches exactly what read it and nothing else. To *react* to a failure — notify someone, file a ticket, write a fallback — use an [action](#actions); that is what they are for.

Returning a failure rather than raising lets the step describe it: the detail reaches the record, and the output the step had already built is still on the outcome.

## Preconditions

`skip_when` covers what the names cannot say: a step having completed proves it produced something, never that the value is usable.

```python
    async def skip_when(self, ctx):
        if ctx.input("document").category != "invoice":
            return "not an invoice"
        return ""
```

It is async, because a precondition is usually a database read. A skipped step never starts, and its reason is recorded.

## Actions

Everything that watches a run rather than doing its work is an action, and there is one kind of object for all of it — recording the run's rows, telling someone a step finished, delivering the result when it ends.

```python
from django_ai_sdk.workflows import WorkflowAction


class NotifyAction(WorkflowAction):
    description = "Tell the run's user a step finished"

    async def on_step_end(self, ctx, step, outcome):
        await notify(ctx.principal, f"{step.name}: {outcome.status}")
```

| Callback | When |
| --- | --- |
| `on_run_start(ctx)` | Before the first step is considered. |
| `on_step_start(ctx, step)` | A step is about to run. Not called for a step that is skipped. |
| `on_step_end(ctx, step, outcome)` | A step settled — completed, failed or skipped. |
| `on_run_end(ctx, error)` | The run is over. `error` is what ended it, or `None`. |

Every callback is a no-op by default, so an action implements only the moments it cares about.

**An action cannot fail the run.** Steps do the work; actions watch it. An action that raises is that action's own bug — it is logged against its class name and the run carries on, so a notification nobody received does not turn a workflow whose every step completed into a failed one, and one broken action does not stop the actions after it. The single exception is `StepAlreadyRunning`, below.

**Where it is attached decides what it sees.** An action on the workflow gets all four callbacks, for the run and for every step in it. An action on one step gets the two step callbacks, for that step alone:

```python
await run_steps(PIPELINE, actions=[NotifyAction()])          # every step
SummariseStep.actions = (NotifyAction(),)                    # this step only
```

An action whose rows are also a claim refuses a duplicate delivery by raising `StepAlreadyRunning` — the one exception the runner lets through rather than logging. The queued entry point treats it as the ordinary case, and neither the recorder nor the executor stamps the run row, because the delivery that holds the run is still working on it.

## Recording a run

The runner knows things no step can: that a step was blocked by an upstream failure, that it was skipped, that it crashed before writing anything. `RunRecorder` is the action that writes them down, and the executor always attaches it.

```python
from django_ai_sdk.workflows import RunRecorder, WorkflowRun

run = await WorkflowRun.objects.acreate(status=WorkflowRun.Status.RUNNING)
await run_steps(PIPELINE, inputs={"document": document}, actions=[RunRecorder(run, PIPELINE)])
```

`RunRecorder` writes one `WorkflowRunStep` row per step, and takes the declared pipeline because that is what fixes each row's order. A host that already keeps a step table — because a review screen queries one — adds an action of its own instead of writing the same fact twice under two retention policies.

## Resume

Dispatch is at-least-once, so the same pipeline can be handed to a worker twice. A re-dispatch resumes rather than repeating: pass what already finished as `completed=`, and those steps are trusted rather than run again. They get no action call either, so a resume does not re-date the row it is trusting.

```python
await run_steps(PIPELINE, completed={"text": stored_text})
```

`WorkflowExecutor` does this for you, reading the run's own completed rows.

## Interpreting a step's output

The runner never inspects `output`. What a step returns means whatever the host decides — a value the next step reads, a payload to file, a list of things a person should check. That interpretation belongs in the host's own code, in its own vocabulary:

```python
outcomes = await run_steps(PIPELINE, inputs={"document": document}, actions=[recorder])

if outcomes["extract"].status == "completed":
    my_app.record_issues(document, outcomes["extract"].output)
```

## What a step may read

`requires` names steps declared **earlier** in the pipeline. A name that is not one of them is refused with `ImproperlyConfigured` before the first step runs, so a misspelled `requires` — or a step declared ahead of the one it reads — fails up front rather than handing the step `None`:

```python
await run_steps(PIPELINE)
# ImproperlyConfigured: Step 'extract' requires ['extracted_txt'], which no earlier
# step produces. Declared by this point: ['document', 'extracted_text'].
```

Inputs need no `requires`: they are their own namespace, present from the start, and a definition that declares `input_fields` has them checked before the run is even queued.

## Declaring a workflow

Expose the step class under a key, then name that key from a definition. The registry
is the gate: a type that is not listed cannot be composed by a runtime author.

```python
# settings.py
AI_SDK_WORKFLOW_STEPS = {"ocr": "myapp.steps.OcrStep"}
```

```python
# myapp/workflows.py — autodiscovered on startup
from django_ai_sdk.workflows import FieldDefinition, StepDefinition, WorkflowDefinition, register

register(
    WorkflowDefinition(
        name="process-document",
        input_fields={"document": FieldDefinition(type="str", description="Document id")},
        steps=[
            StepDefinition(type="ocr", name="text"),
            StepDefinition(
                type="agent",
                name="triage",
                agent_id=TriageAgent().agent_id,
                requires=["text"],
            ),
        ],
    )
)
```

The definition owns the graph — `name`, `requires`, `on_error`, `actions` — and the class
supplies the behaviour. Class attributes on a registered `Step` are not defaults for
those fields.

A run's inputs cross a queue as JSON, so pass the id of a row rather than the row and
let the step resolve it:

```python
run = await WorkflowService.run_by_id(workflow_id, inputs={"document": str(document.pk)}, user=user)
```

The queued entry point logs a `StepFailed` or a `StepAlreadyRunning` rather than
re-raising: both are already recorded by the time they reach it. Anything else reaches
the queue as a real failure.

## Limitations

- **A queued workflow has to be declared at module level.** `enqueue` names it by import path, so a class nested inside another class or a function cannot be resolved by the worker.
- **No retry, and no re-run of one step.** A failed step is re-run only by dispatching the whole pipeline again, which resumes past whatever already completed.
- **No timeout of its own.** A model call is bounded by [`AI_SDK_LLM_TIMEOUT`](/manual/settings/) in the generator, but a step doing its own HTTP or subprocess work has to bound its own client. `django_tasks` has no timeout either.
- **A worker that dies mid-step leaves the row open.** Nothing fires once the process is gone, so a host that must recover from a killed worker needs its own sweep over rows still marked running.
- **Sequential.** Steps run one at a time, in declared order, even where two of them read nothing from each other.
- **Order is yours to get right.** A step declared ahead of the one it reads is refused, not reordered.
- **An `AgentStep` runs the CHAT permission check** on the principal, the same way a JSON agent step does. A code pipeline is not a permission bypass.
- **`RunRecorder` is an action, so it is isolated like any other.** A database error while writing a step row is logged rather than fatal, which means a run can finish with a row missing from its record. The trade is deliberate: the alternative is letting any action fail the work.
