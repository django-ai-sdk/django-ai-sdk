---
title: Workflows
type: docs
weight: 121
---

The workflow engine orchestrates multi-step agent tasks: sequential agent steps that pass results forward, then optional side-effect actions. The [Views and Routing guide](/views-and-routing/#workflows) covers the public API; this page documents the definition schema, models, and executor.

## The Definition

```python
from django_ai_sdk.workflows import WorkflowDefinition, WorkflowStep, WorkflowAction

workflow = WorkflowDefinition(
    name="Summarize and alert",
    steps=[
        WorkflowStep(
            name="summarize",
            agent_id="summarizer",
            output_key="summary",
        ),
        WorkflowStep(
            name="classify",
            agent_id="classifier",
            requires=["summary"],                # read step 1's output
            output_key="priority",
            output_fields={                     # structured output
                "priority": {"type": "str", "description": "high|low"},
                "score": {"type": "float"},
            },
        ),
    ],
    actions=[WorkflowAction(type="thread_message", input_key="summary")],
)
```

### Step semantics

Each `WorkflowStep` runs its `agent_id` via `agent.run()` (non-streaming). Its result is stored under `output_key` in the run's outputs.

| Field | Purpose |
| --- | --- |
| `type` | `"agent"`, or a key of [`AI_SDK_WORKFLOW_STEPS`](/manual/settings/) naming a registered `Step` class. |
| `name` | Display name for the step record; defaults to `output_key`. |
| `agent_id` | Agent to run (resolved through `AgentService`). Required for `type: "agent"`, refused otherwise. |
| `output_key` | The run-state name the result is filed under. Required for an agent step; a registered step may omit it when its result is the rows it wrote. |
| `requires` | Run-state names this step reads: an earlier step's `output_key` / `error_key`, or a key of the run's inputs. Each is passed to an agent step as a `[name]` user message. A name with no source is refused before any step runs. Definitions stored with the older `input_key` are rewritten into `requires` by migration `0005`; the schema no longer accepts the key. |
| `on_error` | `"fail"` stops the run; `"continue"` records the failure and runs the rest. |
| `error_key` | When `on_error` is `"continue"` and the step fails, publish `{step, error}` under this run-state name so a later step (or action) can require it. Must differ from `output_key`. |
| `system_prompt_override` | Optional system prompt for this step. |
| `output_fields` | When set, the agent runs with structured output: a dynamic Pydantic model is built from the `{name: {type, description}}` map (`type` ∈ `str` / `int` / `float` / `bool`). A result that does not come back as that model fails the step. |

Both kinds compile to `Step` objects and run through the same runner, so every run reads back from one record.

### Registered step types

A definition can only name a `Step` class the deployment exposed:

```python
# settings.py
AI_SDK_WORKFLOW_STEPS = {
    "gather_thread": "apps.agents.steps.GatherStep",
}
```

The definition names and wires it — `name`, `requires`, `output_key`, `on_error` — and the class supplies the behaviour. A key whose path will not import is left out with a warning rather than breaking the site, and a definition naming an unregistered type is refused with `ImproperlyConfigured` listing what is registered.

An agent step — whether a code `AgentStep` or a JSON `type: "agent"` step — is permission-checked (`Operation.CHAT`) against the run's principal. A registered Python step is never permission-checked — the registry is a deployment-level decision. Payload object ids in step inputs are **not** ACL-checked by the engine; hosts that need object auth must enforce it inside the step or refuse to expose those types at runtime.

### Three registries

| Registry | Setting / API | Holds |
| --- | --- | --- |
| Step types | `AI_SDK_WORKFLOW_STEPS` | Python `Step` classes a JSON definition may compose |
| Workflow names | `workflows.register` / `WorkflowSettings.slug` | Runnable definitions, keyed by slug |
| Actions | `AI_SDK_WORKFLOW_ACTIONS` (+ built-in `thread_message`) | Post-run side effects |

### Actions

`WorkflowAction(type, input_key)` runs after all steps complete. The `type` is looked up in the action registry; the action receives `(payload, ActionContext)` where `ActionContext` carries `user`, `agent_id`, and `source`. Unknown types and missing inputs are skipped with a warning.

## Models

| Model | Purpose |
| --- | --- |
| `WorkflowSettings` | A persisted, named workflow: `name`, `slug` (registry key, shared with code declarations), `definition` (JSON), `active`, `created_by`. |
| `WorkflowRun` | One execution: status `pending` / `running` / `completed` / `failed`, `workflow_definition` snapshot, `inputs` (run-state seed, including `messages` for chat-shaped runs), `outputs` (completed `provides` and failed `error_key` payloads), `error`, `task_id`, `user`. |
| `WorkflowRunStep` | Per-step progress: `sequence`, `step_name`, `output_key`, `output`, status `pending` / `running` / `completed` / `failed` / `skipped`, `detail` / `error`, timestamps. |

A `WorkflowSettings.to_workflow_definition()` round-trips the stored JSON.

## Running

```python
from django_ai_sdk.workflows import WorkflowService

# Ad-hoc run (inline definition, no persisted record)
run = await WorkflowService.run(workflow, messages, user=request.user)
run = await WorkflowService.run(workflow, inputs={"document": doc}, user=request.user)

# Persisted workflows
record = await WorkflowService.create("My workflow", workflow, user=request.user)
await WorkflowService.update(workflow_id, name=..., workflow=..., active=...)
await WorkflowService.delete(workflow_id)
await WorkflowService.get(workflow_id)
await WorkflowService.list_workflows(active_only=True)

# Run a persisted workflow. Pass run_id only to resume that row (reuses its
# stored inputs; completed runs short-circuit). Omit run_id for a fresh run.
run = await WorkflowService.run_by_id(
    workflow_id, messages, inputs={"file": ref}, user=request.user, run_id=None
)

# Run history
runs = await WorkflowService.list_runs(workflow_id)
run = await WorkflowService.get_run(run_id)      # prefetches steps
```

A definition is checked when it is written, not only when it runs: `create` and `update` refuse one that could not execute (unknown step or action type, a duplicate `output_key` or step name, `error_key` without `on_error="continue"`, or either key shadowing the reserved `messages` input).

## Execution Model

`WorkflowService.run()` creates a `WorkflowRun` in `pending`, then `WorkflowExecutor.enqueue()` schedules the `execute_workflow` background task (`django_tasks`), which runs `WorkflowExecutor.run()`. Whether that happens in a worker or inline in the calling request depends on the configured `TASKS` backend - see [Background Tasks](/manual/settings/#background-tasks).

`WorkflowExecutor.run()`:

1. Seeds `WorkflowRun.inputs` via `normalize_workflow_inputs` (messages always JSON-safe).
2. Marks the run `running` (or returns early for an already-completed run; the operation is idempotent).
3. Compiles the definition to `Step` objects and runs them through one `StepRunner` (JSON and code share this path).
4. **Replays** completed steps from a previous attempt into outputs, so a resumed run continues where it left off.
5. Runs registered actions with `ActionContext`.
6. Marks the run `completed` with the full outputs; a step exception marks the run (and its step) `failed` and propagates.

### Known limits

- Sequential only — no fan-out / join.
- No per-step timeout or automatic retry.
- A killed worker can leave a step row `running`; the SDK does not sweep it.
- No object-level ACL on payload PKs (document ids, etc.).
- In-turn multi-agent chat without a durable record belongs to **subagents**, not workflows.

## Custom Actions

```python
from django_ai_sdk.workflows.actions import ActionContext, BaseAction

class ConsoleLogAction:
    description = "Log the payload"

    async def execute(self, payload, context: ActionContext) -> None:
        print(context.source, payload)
```

```python
# settings.py
AI_SDK_WORKFLOW_ACTIONS = {
    "console_log": "apps.agents.actions.ConsoleLogAction",
}
```

A key whose path will not import is left out with a warning rather than breaking the site, and a definition naming an action that is not registered is refused when it is created.

`WorkflowService.list_actions()` returns `[{key, description}]` from the registry. Built-in: `thread_message`.

## Step Workflows

`WorkflowExecutor` compiles a definition into `Step` objects and calls the `run_steps` walker. The [Step Workflows guide](/workflow-steps/) covers writing step classes; this section is the reference for stored definitions, runs, and actions.

### Step

| Attribute | Default | Purpose |
| --- | --- | --- |
| `name` | `""` | Log key and sink key; required, unique within a pipeline. |
| `requires` | `()` | Run-state names the step reads. Each must be a key of the run's `inputs` or provided by an **earlier** step, or the pipeline is refused before anything runs. |
| `provides` | `""` | The one name this step's output is filed under, or `""` to publish nothing. One name, one producer, and not a name the run's `inputs` already carry. |
| `on_error` | `OnError.FAIL` | `FAIL` aborts the run; `CONTINUE` records the failure and runs the rest. |
| `error_key` | `""` | When `CONTINUE` fails the step, publish `{step, error}` under this name so a handler can `requires` it (and so it lands in `WorkflowRun.outputs`). Empty means the step publishes nothing on failure. |

`skip_when(ctx) -> str` returns a reason to skip or `""` to run. A skipped step is recorded with its reason and publishes nothing, so whatever reads its output skips in turn. `run(ctx) -> StepOutcome` is the work.

`AgentStep` adds `agent` (an `Agent` subclass) and `schema` (a real pydantic model, so nested output is expressible); it implements `run` as one structured call, with async `system_prompt(ctx)` and `user_message(ctx)` hooks, which may query. A `None` result is reported as `failed`, not stored.

### StepContext and StepOutcome

| `StepContext` | Purpose |
| --- | --- |
| `inputs` | What the caller passed to `run_steps`. |
| `outputs` | What earlier steps produced, keyed by the names they declared in `provides`. |
| `principal` | The user the steps act as; forwarded to `AgentStep` calls. |

`ctx.get(name, default)` reads either half, outputs first.

| `StepOutcome` | Purpose |
| --- | --- |
| `output` | The step's value, filed under the name it declares in `provides` and ignored when it declares none. |
| `status` | `completed` / `failed` / `skipped`. Anything but `completed` blocks dependents. |
| `detail` | One line for a log or a timeline. |

### Running

`run_steps(steps, *, inputs=None, principal=None, sink=None)` validates the pipeline, runs the steps **in the order they were declared**, and returns `{step_name: StepOutcome}`. A step runs once every name in its `requires` is on the run's state; one that is not there yet makes the step `skipped`, naming what it waited for. A re-dispatch resumes, skipping the steps the sink reports as already completed. Without `sink=` nothing is recorded.

### Models

`WorkflowRunStep` carries `RUNNING` and `SKIPPED` statuses and a `detail` column holding the one line a step's outcome reported. `sequence` is the step's position in the declared pipeline, and `("run", "sequence")` is its identity — a skipped step never opens, so `started_at` cannot order a run.

A step that never ran is a `skipped` row naming what it was waiting on, not an absence.

### Sinks

| `StepSink` | Called |
| --- | --- |
| `begin()` | Before the first step is considered; default sink marks the run `running`. |
| `completed()` | Before running, for resume: steps already finished, name to stored output. |
| `open(step)` | Before a step runs. |
| `close(step, outcome)` | After a step settles, skips included. A skipped step never opened, so `close` has to create the row. |
| `fail(step, exc)` | When a step raised. |
| `end(outcomes, error)` | After the walk; default sink stamps `failed` when the walker aborted. |

Exactly one of `close` and `fail` settles a step: a step that raised gets `fail` and no `close`; one that returned `status="failed"` gets `close` and no `fail`. A step replayed from `completed()` gets neither, so a resume does not re-date the row it is trusting.

The default, `WorkflowRunStepSink(run, steps)`, writes one `WorkflowRunStep` row per step; it takes the declared pipeline because that is what fixes each row's `sequence` and `output_key`.

```python
run = await WorkflowRun.objects.acreate(status=WorkflowRun.Status.RUNNING)
await run_steps(PIPELINE, inputs={"document": document}, sink=WorkflowRunStepSink(run, PIPELINE))
```

Pass a sink of your own instead when the host already keeps a step table, so one fact is not stored twice under two retention policies. It needs the four methods above and nothing else — no base class, no registration:

```python
class DocumentSink:
    def __init__(self, document):
        self.document = document

    async def completed(self):
        return {
            row.step: None
            async for row in ExtractionStep.objects.filter(
                document=self.document, status="completed"
            )
        }

    async def open(self, step): ...
    async def close(self, step, outcome): ...
    async def fail(self, step, exc): ...
```

`open` may raise to refuse the run. It is called outside the runner's error handling, so the exception reaches the caller instead of being recorded as a step failure — which is what a sink whose rows are also a claim needs, when dispatch is at-least-once. Raise `StepAlreadyRunning` for that, and the queued entry point treats it as the ordinary case rather than a failure.

