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
    on_error = OnError.STOP

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
    user=request.user,
)
```

`inputs` seeds `ctx.input`. `user` is the user an `AgentStep` runs as. `outcomes` maps every step's name to its `StepOutcome`.

## Agent steps

`AgentStep` is one class whether the agent is named by import or by id, so a definition's agent step is this class configured rather than a subclass of it:

| | |
| --- | --- |
| `agent` | An `Agent` subclass, for a step written in code. |
| `agent_id` | An id resolved through `AgentService`, for a step compiled from JSON. |
| `schema` | A pydantic model, passed as the agent's `response_format`. Nested output is expressible. A result that does not come back as that model fails the step. |
| `instructions` | A system prompt replacing the agent's own; empty keeps the agent's. Code steps only: a JSON definition cannot set it. |
| `history` | Names of the inputs to send as the conversation — the author names them, per step, whether the step comes from code or JSON. A `str` input is one user turn; a `list` is a conversation of messages, message dicts, or strings. Only user and assistant turns: a `system` message fails the step. |

It sends the inputs named in `history` as the transcript, then one turn. Override the async `user_message(ctx)` to write that turn, or `messages(ctx)` to write the whole conversation; both may query. By default the turn is what the steps in `requires` produced, each under a `[name]` heading. A `None` result is reported as `failed`, not stored.

A step's call is a turn under the agent's [oversight](/oversight/) policy, recorded on the channel `workflow:<key>:<step>`. The key is the run's `workflow_key`, which the server sets when it opens the run: the stored workflow's slug, the registered workflow's name, or `adhoc` for any other definition. A definition's own `name` never picks it, so a definition cannot borrow the channel a policy chooses by, and no stored or registered workflow may be called `adhoc`. The `history` inputs are not what the server stored: the gate judges them on every run. A question the policy refuses or an answer it withholds fails the step (`refused by oversight: ...` or `withheld by oversight: ...`), so `on_error` decides what that costs; a masked structured result is rebuilt from its masked JSON. A step that overrides `run()` and calls the agent itself passes `channel=self.channel(ctx)` and handles `django_ai_sdk.errors.Blocked`.

The transcript is **named, not guessed**. An input is data until a step names it as `history`, so a list of role-shaped rows that happens to be data is never mistaken for a prompt, and a workflow with two agent steps can give each a different conversation. What a history field holds is the author's choice: a `str` arrives as one user turn, a message dict as one message, a list item-by-item.

## Failure

`on_error` decides what a step's failure does to the rest of the run, and applies whether the step raised or returned `status="failed"`:

| | |
| --- | --- |
| `OnError.STOP` | The run stops. Every step that will now not run is recorded as `skipped`, then the exception propagates — `WorkflowStopped` when the step reported failure rather than raising. |
| `OnError.CONTINUE` | The failure is recorded and the rest of the run proceeds. |

A failed step produces nothing, and that is the whole cascade: a step whose required step did not complete is recorded `skipped`, naming what it waited for and whether that failed or was skipped. So a failure reaches exactly what read it and nothing else. To *react* to a failure — notify someone, file a ticket, write a fallback — use a [listener](#listeners); that is what they are for.

Returning a failure rather than raising lets the step describe it: the detail reaches the record, and the output the step had already built is still on the outcome.

## Preconditions

`skip_if` covers what the names cannot say: a step having completed proves it produced something, never that the value is usable.

```python
    async def skip_if(self, ctx):
        if ctx.input("document").category != "invoice":
            return "not an invoice"
        return ""
```

It is async, because a precondition is usually a database read. A skipped step never starts, and its reason is recorded.

## Listeners

Everything that watches a run rather than doing its work is a listener, and there is one kind of object for all of it — recording the run's rows, telling someone a step finished, delivering the result when it ends.

```python
from django_ai_sdk.workflows import RunListener


class NotifyListener(RunListener):
    description = "Tell the run's user a step finished"

    async def on_step_end(self, ctx, step, outcome):
        await notify(ctx.user, f"{step.name}: {outcome.status}")
```

| Callback | When |
| --- | --- |
| `on_run_start(ctx)` | Before the first step is considered. |
| `on_step_start(ctx, step)` | A step is about to run. Not called for a step that is skipped. |
| `on_step_end(ctx, step, outcome)` | A step settled — completed, failed or skipped. |
| `on_run_end(ctx, error)` | The run is over. `error` is what ended it, or `None`. |

Every callback is a no-op by default, so a listener implements only the moments it cares about.

**A listener cannot fail the run.** Steps do the work; listeners watch it. A listener that raises is that listener's own bug — it is logged against its class name and the run carries on, so a notification nobody received does not turn a workflow whose every step completed into a failed one, and one broken listener does not stop the listeners after it.

**Where it is attached decides what it sees.** A listener on the workflow gets all four callbacks, for the run and for every step in it. A listener on one step gets the two step callbacks, for that step alone:

```python
await run_steps(PIPELINE, listeners=[NotifyListener()])          # every step
SummariseStep.listeners = (NotifyListener(),)                    # this step only
```

## Recording a run

The runner knows things no step can: that a step was blocked by an upstream failure, that it was skipped, that it crashed before writing anything. `RunRecorder` writes them down, and every queued run walks with one.

```python
from django_ai_sdk.workflows.runs import RunRecorder
from django_ai_sdk.workflows.models import WorkflowRun

run = await WorkflowRun.objects.acreate(status=WorkflowRun.Status.RUNNING)
await run_steps(PIPELINE, inputs={"document": document}, recorder=RunRecorder(run, PIPELINE))
```

`RunRecorder` writes one `StepRun` row per step, and takes the declared pipeline because that is what fixes each row's order. It is passed as `recorder=`, not among the listeners, because it is the run's own record: it is told about each step first, and an error it raises (a database that is down, a run that is no longer this worker's) stops the run instead of being logged. A step does not run without its row. It also refreshes the run's `heartbeat_at` at every step, which is what `workflows_sweep` reads. A host that already keeps a step table — because a review screen queries one — adds a listener of its own instead of writing the same fact twice under two retention policies.

## Resume

A pipeline you run yourself resumes by passing what already finished as `completed=`: those steps are trusted rather than run again. They get no listener call either, so a resume does not re-date the row it is trusting.

```python
await run_steps(PIPELINE, completed={"text": stored_text})
```

A queued run does this for you. Dispatch is at-least-once, so the same run can reach a worker twice: the worker claims the run first (pending to running, in one conditional update), and a second delivery finds nothing to claim. A run that failed, or that `workflows_sweep` marked interrupted after its worker died, is queued again with `WorkflowService.resume(run_id)`; the worker that claims it reads the run's completed rows and continues from there.

## Interpreting a step's output

The runner never inspects `output`. What a step returns means whatever the host decides — a value the next step reads, a payload to file, a list of things a person should check. That interpretation belongs in the host's own code, in its own vocabulary:

```python
outcomes = await run_steps(PIPELINE, inputs={"document": document}, recorder=recorder)

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
AI_SDK_WORKFLOWS = {"STEPS": {"ocr": "myapp.steps.OcrStep"}}
```

```python
# myapp/workflows.py — autodiscovered on startup
from django_ai_sdk.workflows import FieldDefinition, StepDefinition, WorkflowDefinition, register_workflow

register_workflow(
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

The definition owns the graph — `name`, `requires`, `on_error`, `listeners` — and the class
supplies the behaviour. Class attributes on a registered `Step` are not defaults for
those fields.

A registered step that needs settings — which language to read, how many pages —
takes them from its definition's `config`, and says what it accepts in
`validate_config`. The definition is checked when it is stored or registered, so a bad
config is refused there rather than in the worker. A step that does not override it
accepts no config at all.

```python
from pydantic import BaseModel


class OcrConfig(BaseModel):
    language: str = "nld"


class OcrStep(Step):
    @classmethod
    def validate_config(cls, config):
        OcrConfig.model_validate(config)  # a ValidationError is a ValueError

    async def run(self, ctx):
        language = OcrConfig.model_validate(self.config).language
        ...


StepDefinition(type="ocr", name="text", config={"language": "eng"})
```

A run's inputs cross a queue as JSON, so pass the id of a row rather than the row and
let the step resolve it:

```python
run = await WorkflowService.start("process-document", inputs={"document": str(document.pk)}, user=user)
```

The queued entry point logs a `WorkflowStopped` or a `RunSuperseded` rather than
re-raising: the run's rows already say what happened. Anything else reaches the queue
as a real failure.

## Limitations

Workflows are a beta: retries, timeouts, parallel steps, human approval, the sweep, run data and snapshot semantics are listed once, under [Beta limitations](/manual/workflows/#beta-limitations). What concerns writing steps in particular:

- **No re-run of one step.** A failed step runs again only when the whole run is resumed, which continues past whatever already completed.
- **A step does not beat while it runs.** `workflows_sweep --stale-after` must be longer than your slowest step, or a live run is marked `interrupted` (its worker then stops at its next step).
- **Order is yours to get right.** A step declared ahead of the one it reads is refused, not reordered.
- **An `AgentStep` runs the CHAT permission check** on the user, the same way a JSON agent step does. A code pipeline is not a permission bypass.
- **A pipeline you pass to `run_steps` yourself is not queued**, recorded or swept: only a `WorkflowRun` is.
