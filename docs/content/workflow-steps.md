---
title: Step Workflows
type: docs
weight: 55
---

A real pipeline is not only agent steps: it fetches a file, converts it, calls an API, validates what came back, writes rows. This page is how you declare those in Python, so a host does not orchestrate around the engine and lose the record of what happened.

There is one runner. A [stored `WorkflowDefinition`](/manual/workflows/) is compiled into the same `Step` objects this page declares and walked by the same `run_steps`, so both read back from one record. What differs is who authors it and what it may reach: a definition is JSON written at runtime, held to the step types the deployment registered; a pipeline is code, free to do work no model is involved in.

A step is a named unit of work that reads the run's state and returns a value. The runner does not interpret what it returns.

## Declaring steps

Steps run in the order you declare them. `requires` and `provides` name entries in the run's state: a step runs once every name it requires is there, and what it returns is filed under the name it provides, for the steps after it.

```python
from django_ai_sdk.workflows import AgentStep, OnError, Step, StepOutcome


class ReadFileStep(Step):
    name = "read"
    requires = ("document",)
    provides = "text"
    on_error = OnError.FAIL

    async def run(self, ctx):
        document = ctx.get("document")
        text = await read_the_file(document)
        if not text.strip():
            return StepOutcome(status="failed", detail="the file yielded no text")
        return StepOutcome(output=text, detail=f"{len(text)} characters")


class SummariseStep(AgentStep):
    name = "summarise"
    requires = ("text",)
    provides = "summary"
    on_error = OnError.CONTINUE

    agent = SummaryAgent
    schema = Summary

    async def user_message(self, ctx):
        return ctx.get("text")


PIPELINE = (ReadFileStep(), SummariseStep())
```

A step declaring a name in `provides` returns that value directly; a step declaring none returns whatever it likes and the runner publishes nothing — a step whose real result is the rows it wrote is exactly that case. A step with several values to hand on returns one structured object under one name.

## Running

```python
from django_ai_sdk.workflows import run_steps

outcomes = await run_steps(
    PIPELINE,
    inputs={"document": document},
    principal=request.user,
)
```

`inputs` seeds the run's state; steps read it and their own predecessors' output through `ctx.get(name)`. `principal` is the user an `AgentStep` runs as. `outcomes` maps every step's name to its `StepOutcome`.

## Failure

`on_error` decides what a step's failure does to the rest of the run, and applies whether the step raised or returned `status="failed"`:

| | |
| --- | --- |
| `OnError.FAIL` | The run stops. Every step that will now not run is recorded as `skipped`, then the exception propagates — `StepFailed` when the step reported failure rather than raising. |
| `OnError.CONTINUE` | The failure is recorded and the rest of the run proceeds. |

Without an `error_key`, the failed step publishes nothing, and that is the whole cascade: a step whose required name is missing is recorded `skipped`, naming what it waited for. So a failure reaches exactly what read its `provides` and nothing else. Set `error_key` when a later step should handle the failure — the runner publishes `{step, error}` under that name, and the executor stores it on `WorkflowRun.outputs`.

Returning a failure rather than raising lets the step describe it — the detail reaches the record, and the output the step had already built is still on the outcome.


## Preconditions

`skip_when` covers what the names cannot say: a name being on the table proves it was produced, never that its value is usable.

```python
    async def skip_when(self, ctx):
        if ctx.get("document").category != "invoice":
            return "not an invoice"
        return ""
```

It is async, because a precondition is usually a database read. A skipped step never opens, and its reason is recorded.

## Resume

Dispatch is at-least-once, so the same pipeline can be handed to a worker twice. A re-dispatch resumes rather than repeating: the sink reports which steps already completed and what they returned, and those are trusted rather than run again.

## Recording a run

The runner knows things no step can: that a step was blocked by an upstream failure, that it was skipped, that it crashed before writing anything, and which steps a resume may trust. A sink is where it writes them.

```python
from django_ai_sdk.workflows import WorkflowRun, WorkflowRunStepSink

run = await WorkflowRun.objects.acreate(status=WorkflowRun.Status.RUNNING)
await run_steps(PIPELINE, inputs={"document": document}, sink=WorkflowRunStepSink(run, PIPELINE))
```

`WorkflowRunStepSink` writes one `WorkflowRunStep` row per step, and takes the declared pipeline because that is what fixes each row's order. A host that already keeps a step table — because a review screen queries one — passes its own sink instead of writing the same fact twice under two retention policies:

```python
class DocumentSink:
    def __init__(self, document):
        self.document = document

    async def completed(self):
        """Steps already finished, name to stored output, for resume."""
        ...

    async def open(self, step): ...
    async def close(self, step, outcome): ...
    async def fail(self, step, exc): ...
```

A sink is constructed by the host, so it is bound to whatever it records against and two workflows in one project can each keep their own. `open` fires before a step runs, `close` with its outcome after, `fail` with the exception when it raised. Omit `sink=` entirely and nothing is recorded.

`open` may raise to refuse the run. It is called outside the runner's own error handling, so the exception propagates to the caller instead of being recorded as a step failure — which is what a sink whose rows are also a claim needs, when dispatch is at-least-once and a second worker reaches the same subject.

## Declaring a workflow

Expose the step class under a key, then name that key from a definition. The registry
is the gate: a type that is not listed cannot be composed by a runtime author.

```python
# settings.py
AI_SDK_WORKFLOW_STEPS = {"ocr": "myapp.steps.OcrStep"}
```

```python
# myapp/workflows.py — autodiscovered on startup
from django_ai_sdk.workflows import WorkflowDefinition, WorkflowStep, register

register(
    WorkflowDefinition(
        name="process-document",
        steps=[
            WorkflowStep(type="ocr", name="read", output_key="text", requires=["document"]),
            WorkflowStep(
                type="agent",
                name="triage",
                agent_id=TriageAgent().agent_id,
                output_key="triage",
                requires=["text"],
            ),
        ],
    )
)
```

The definition owns the graph — `name`, `requires`, `output_key`, `on_error`,
`error_key` — and the class supplies the behaviour. Class attributes on a registered
`Step` are not defaults for those fields.

A run's inputs cross a queue as JSON, so pass the id of a row rather than the row and
let the step resolve it:

```python
run = await WorkflowService.run_by_id(workflow_id, inputs={"document": str(document.pk)}, user=user)
```

The queued entry point logs a `StepFailed` or a `StepAlreadyRunning` rather than
re-raising: both are already recorded by the time they reach it. Anything else reaches
the queue as a real failure.

## Interpreting a step's output

The runner never inspects `output`. What a step returns means whatever the host decides — a value the next step reads, a payload to file, a list of things a person should check. That interpretation belongs in the host's own code, in its own vocabulary:

```python
outcomes = await run_steps(PIPELINE, inputs={"document": document}, sink=sink)

if outcomes["extract"].status == "completed":
    my_app.record_issues(document, outcomes["extract"].output)
```

## What a step may read

`requires` names entries in the run's state. A name is resolved from one of two places, checked before the first step runs:

- the `provides` of a step declared **earlier** in the pipeline
- a key of the `inputs` mapping passed to `run_steps`

A name in neither is refused with `ImproperlyConfigured` naming the step and the name, so a misspelled `requires` — or a step declared ahead of the one that produces its input — fails before the first step runs rather than handing the step `None`. A `provides` that collides with an `inputs` key is refused for the same reason: one name, one source.

```python
await run_steps(PIPELINE, inputs={"document": document})
# ImproperlyConfigured: Step 'extract' requires ['extracted_txt'], which the run's inputs
# do not carry and no earlier step provides. Available by this point: ['document',
# 'extracted_text'].
```

## What a resumed run reads

A step the sink reports as completed is not run again; what it recorded is published into the run's state for the steps after it, and it gets no sink call, so a resume does not re-date the row it is trusting.

A sink that stores no output — one whose steps write their own columns and read them back — returns `None`, and its steps read their persisted state instead of the run's.

## Limitations

- **A queued workflow has to be declared at module level.** `enqueue` names it by import path, so a class nested inside another class or a function cannot be resolved by the worker.
- **No retry, and no re-run of one step.** A failed step is re-run only by dispatching the whole pipeline again, which resumes past whatever already completed.
- **No timeout of its own.** A model call is bounded by [`AI_SDK_LLM_TIMEOUT`](/manual/settings/) in the generator, but a step doing its own HTTP or subprocess work has to bound its own client. `django_tasks` has no timeout either.
- **A worker that dies mid-step leaves the row open.** Nothing fires once the process is gone, so a host that must recover from a killed worker needs its own sweep over rows still marked running.
- **Sequential.** Steps run one at a time, in declared order, even where two of them read nothing from each other.
- **Order is yours to get right.** A step declared ahead of the one producing its input is refused, not reordered.
- **An `AgentStep` runs the CHAT permission check** on the principal, the same way a JSON agent step does. A code pipeline is not a permission bypass.
