---
title: Workflows
type: docs
weight: 121
---

The workflow engine orchestrates multi-step agent tasks: steps that pass results forward, and hooks that watch the run. The [Views and Routing guide](/views-and-routing/#workflows) covers the public API; this page documents the definition schema, models, and executor.

## The Definition

```python
from django_ai_sdk.workflows import FieldDefinition, HookDefinition, StepDefinition, WorkflowDefinition

workflow = WorkflowDefinition(
    name="summarize-and-alert",
    input_fields={
        "history": FieldDefinition(type="list", description="The conversation so far"),
        "document": FieldDefinition(type="str", description="Document id"),
        "options": FieldDefinition(
            type="object",
            fields={
                "tone": FieldDefinition(
                    type="str", enum=["brief", "thorough"], required=False, default="brief"
                ),
            },
        ),
    },
    steps=[
        StepDefinition(
            name="summary",
            agent_id="summarizer",
            history=["history"],
        ),
        StepDefinition(
            name="priority",
            agent_id="classifier",
            requires=["summary"],               # read the step named "summary"
            history=["history"],
            output_fields={                     # structured output
                "priority": FieldDefinition(type="str", enum=["high", "low"]),
                "score": FieldDefinition(type="float"),
            },
        ),
    ],
    hooks=[HookDefinition(type="thread_message", config={"step": "summary"})],
)
```

### Two namespaces

A run's state has two halves, and nothing crosses between them. `ctx.input(name)` is what the caller supplied; `ctx.step(name)` is what an earlier step produced, keyed by **that step's own name**. A step needs no `output_key`, and an input and a step may go by the same word without either being renamed.

### Declared inputs

`input_fields` compiles to a pydantic model, the same way `output_fields` does — one mechanism pointed in two directions. A run's inputs are validated against it once, before the run is queued, so a caller who forgot something hears about it on the call that queued the run rather than three steps in. Undeclared extras are dropped: the schema is the contract.

The shape is the whole contract — structure, enums, defaults — because a declared schema has to survive a dump, cross the queue, and be read back by an editor or an agent. Behaviour does not cross: validators and value constraints (`min_length`, `ge`, …) belong in a step, not in a field.

| `type` | Becomes |
| --- | --- |
| `str` / `int` / `float` / `bool` | The Python type. |
| `list` / `dict` | An open JSON container the host shapes itself. |
| `list` + `items` | A typed list; `items` is another `FieldDefinition`, so elements are checked too. |
| `object` + `fields` | A typed record; `fields` is a `{name: FieldDefinition}` map, recursively. |

A field may also carry `enum` (the values a scalar may take, compiled to `Literal`), `default` (implying `required: false`, with the value used when the caller omits it), and `description` (for the model, the editor, and whoever reads the JSON). Nesting is capped at 8 levels and 32 fields per object, so a stored definition cannot declare a pathology the compiler has to walk.

A workflow written in code may hand the authoring to a pydantic model instead — `WorkflowDefinition(inputs=MyModel, ...)` — which is normalized into the same fields and never rides on the definition, so a stored definition and a code-declared one have one wire format. What pydantic can express as a shape crosses; validators, `default_factory` and `Annotated` constraints do not, and a model carrying any of them is refused rather than half-honored.

```python
from pydantic import BaseModel
from typing import Literal

class DigestInputs(BaseModel):
    thread: str
    tone: Literal["brief", "thorough"] = "brief"
```

There is no `file` type, and no `messages` type either. A file is whatever the host already stores it as — a `str` id or a `dict` reference under whatever key it picks, resolved by the first step that reads it. A conversation is an input like any other, made the transcript by the step that names it (below).

A definition that declares no `input_fields` takes what it is given: a code-authored pipeline is not obliged to describe itself in JSON.

There is no workflow-level `output_fields` either: a run's `outputs` is the map of step outputs keyed by step name, and each step already declares its own shape.

### Step semantics

An agent step runs its `agent_id` via `agent.run()` (non-streaming). Its result is filed under the step's `name`.

| Field | Purpose |
| --- | --- |
| `type` | `"agent"`, or a key of [`AI_SDK_WORKFLOW_STEPS`](/manual/settings/) naming a registered `Step` class. |
| `name` | Required, unique. The key the result is filed under, the key later steps require, and the key the record is written under. |
| `agent_id` | Agent to run (resolved through `AgentService`). Required for `type: "agent"`, refused otherwise. |
| `requires` | Names of **earlier** steps this one reads. Each is passed to an agent step as a `[name]` user message. A name no earlier step declares is refused before any step runs. Inputs are not listed here — they are always available. |
| `history` | Agent steps only. The inputs this step sends as its conversation, in order: a `str` field is one user turn, a `list` field a conversation of messages, message dicts, or strings. The author names them; the engine never infers a transcript from the shape of a value. A name `input_fields` does not declare is refused. |
| `on_error` | `"fail"` stops the run; `"continue"` records the failure and runs the rest. |
| `system_prompt_override` | Optional system prompt for this step. |
| `output_fields` | Agent steps only. The agent runs with structured output: a pydantic model is built from the `{name: FieldDefinition}` map, and a result that does not come back as that model fails the step. A registered step returns what it returns, so declaring them there is refused. |
| `hooks` | Hooks that fire for this step alone. |

Both kinds compile to `Step` objects and run through the same runner, so every run reads back from one record. An agent step compiles to a configured `AgentStep` — there is no second agent-step class for definitions.

### Registered step types

A definition can only name a `Step` class the deployment exposed:

```python
# settings.py
AI_SDK_WORKFLOW_STEPS = {
    "gather_thread": "apps.agents.steps.GatherStep",
}
```

The definition names and wires it — `name`, `requires`, `on_error`, `hooks` — and the class supplies the behaviour. A key whose path will not import is left out with a warning rather than breaking the site, and a definition naming an unregistered type is refused with `ImproperlyConfigured` listing what is registered.

An agent step — whether a code `AgentStep` or a JSON `type: "agent"` step — is permission-checked (`Operation.CHAT`) against the run's principal. A registered Python step is never permission-checked — the registry is a deployment-level decision. Payload object ids in step inputs are **not** ACL-checked by the engine; hosts that need object auth must enforce it inside the step or refuse to expose those types at runtime.

### Three registries

| Registry | Setting / API | Holds |
| --- | --- | --- |
| Step types | `AI_SDK_WORKFLOW_STEPS` | Python `Step` classes a JSON definition may compose |
| Workflow names | `workflows.register` / `WorkflowSettings.slug` | Runnable definitions, keyed by slug |
| Hooks | `AI_SDK_WORKFLOW_HOOKS` | Everything that watches a run rather than doing its work |

### Hooks

One kind of object, attached in two places. `HookDefinition(type, config)` names a key of `AI_SDK_WORKFLOW_HOOKS`; `config` is handed to the hook as it is built, so one registered class serves every definition that names it.

| Attached to | Sees |
| --- | --- |
| `WorkflowDefinition.hooks` | `on_run_start`, `on_step_start`, `on_step_end`, `on_run_end` — the run, and every step in it. |
| `StepDefinition.hooks` | `on_step_start`, `on_step_end` — that step alone. |

Reacting to a failure is a hook's job: a workflow-level hook sees every step that ends, whatever its status. A hook cannot fail the run — steps do the work, hooks watch it — so one that raises is logged against its name and the walk carries on. The single exception is `StepAlreadyRunning`, which is how a hook refuses a duplicate delivery. The package ships no hooks a definition can name — delivering a result somewhere is the host's business. `RunRecorder` is the one built-in, and the executor always attaches it.

A key whose path will not import is left out with a warning rather than breaking the site; a definition naming a hook that is not registered is refused when it is created. `WorkflowService.list_hooks()` returns `[{key, description}]` from the registry.

```python
from django_ai_sdk.workflows import WorkflowHook


class ConsoleLogHook(WorkflowHook):
    description = "Log each step as it settles"

    async def on_step_end(self, ctx, step, outcome):
        print(ctx.workflow, step.name, outcome.status)
```

```python
# settings.py
AI_SDK_WORKFLOW_HOOKS = {
    "console_log": "apps.agents.hooks.ConsoleLogHook",
}
```

## Models

| Model | Purpose |
| --- | --- |
| `WorkflowSettings` | A persisted, named workflow: `name`, `slug` (registry key, shared with code declarations), `definition` (JSON), `active`, `created_by`. |
| `WorkflowRun` | One execution: status `pending` / `running` / `completed` / `failed`, `workflow_definition` snapshot, `inputs` (what the caller supplied), `outputs` (`{step name: output}` for every completed step), `error`, `task_id`, `user`. |
| `WorkflowRunStep` | Per-step progress: `sequence`, `step_name`, `output`, status `pending` / `running` / `completed` / `failed` / `skipped`, `detail` / `error`, timestamps. |

A `WorkflowSettings.to_workflow_definition()` round-trips the stored JSON.

## Running

```python
from django_ai_sdk.workflows import WorkflowService

# Ad-hoc run (inline definition, no persisted record)
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
    workflow_id, inputs={"file": ref}, user=request.user, run_id=None
)

# Run history
runs = await WorkflowService.list_runs(workflow_id)
run = await WorkflowService.get_run(run_id)      # prefetches steps

# The inputs as a JSON Schema, for a form or a tool that composes a run
schema = await WorkflowService.get_inputs_schema(workflow_id, user=request.user)
```

A definition is checked when it is written, not only when it runs: `create` and `update` refuse one that could not execute (an unknown step type or hook, a duplicate step name, a `requires` no earlier step produces, or an `input_fields` / `output_fields` map that pydantic will not build). Inputs are checked on the call that queues the run. Every definition carries a `version` (currently `1`), stored with it, so a future format revision can gate on it rather than guess.

## Execution Model

`WorkflowService.run()` creates a `WorkflowRun` in `pending`, then `WorkflowExecutor.enqueue()` schedules the `execute_workflow` background task (`django_tasks`), which runs `WorkflowExecutor.run()`. Whether that happens in a worker or inline in the calling request depends on the configured `TASKS` backend — see [Background Tasks](/manual/settings/#background-tasks).

`WorkflowExecutor.run()`:

1. Marks the run `running` (or returns early for an already-completed run; the operation is idempotent).
2. Validates the run's inputs against the definition's `input_fields`, when it declares any.
3. Compiles the definition to `Step` objects and hooks, and runs them through `run_steps` (JSON and code share this path).
4. **Replays** completed steps from a previous attempt, so a resumed run continues where it left off and does not re-date the rows it is trusting.
5. Marks the run `completed` with `{step name: output}` for every completed step; a step exception marks the run (and its step) `failed` and propagates.

### Known limits

- Sequential only — no fan-out / join.
- No per-step timeout or automatic retry.
- A killed worker can leave a step row `running`; the SDK does not sweep it.
- No object-level ACL on payload PKs (document ids, etc.).
- Declared schemas are structural: no value constraints (`min_length`, `ge`, …) and no validators. Check those in a step.
- In-turn multi-agent chat without a durable record belongs to **subagents**, not workflows.

## Step Workflows

`WorkflowExecutor` compiles a definition into `Step` objects and calls the `run_steps` walker. The [Step Workflows guide](/workflow-steps/) covers writing step classes and hooks; this section is the reference for the runtime objects.

### Step

| Attribute | Default | Purpose |
| --- | --- | --- |
| `name` | `""` | Required, unique within a pipeline. The key its outcome is recorded under and the key later steps require. |
| `requires` | `()` | Steps declared **earlier** that this one reads. A name that is not one of them is refused before anything runs. |
| `on_error` | `OnError.FAIL` | `FAIL` aborts the run; `CONTINUE` records the failure and runs the rest. |
| `hooks` | `()` | Hooks that fire for this step alone. |

`skip_when(ctx) -> str` returns a reason to skip or `""` to run. A skipped step is recorded with its reason and produces nothing, so whatever reads it skips in turn. `run(ctx) -> StepOutcome` is the work.

`AgentStep` adds `agent` / `agent_id`, `schema` (a real pydantic model, so nested output is expressible), `instructions` and `history` (the inputs to send as the conversation — the step names them; a `str` input is one turn, a list is a conversation); it implements `run` as one structured call, with async `system_prompt(ctx)`, `user_message(ctx)` and `messages(ctx)` hooks, which may query. A `None` result is reported as `failed`, not stored.

### WorkflowContext and StepOutcome

One context, for steps and hooks alike.

| `WorkflowContext` | Purpose |
| --- | --- |
| `inputs` / `input(name)` | What the caller supplied, coerced to the declared `input_fields`. |
| `steps` / `step(name)` | What has run so far, keyed by step name. `step()` returns the output of a **completed** step; `steps[name]` is the whole outcome, a failure's detail included. |
| `principal` | The user the steps act as; forwarded to `AgentStep` calls. |
| `workflow` / `run_id` | The definition's name and the run's id, for hooks that report on the run. |

| `StepOutcome` | Purpose |
| --- | --- |
| `output` | The step's value, filed under the step's own name. |
| `status` | `completed` / `failed` / `skipped`. Anything but `completed` blocks dependents. |
| `detail` | One line for a log or a timeline. |

### Running

`run_steps(steps, *, inputs=None, principal=None, hooks=(), completed=None, workflow="", run_id="")` validates the pipeline, runs the steps **in the order they were declared**, and returns `{step_name: StepOutcome}`. A step runs once every step in its `requires` has completed; one that has not makes the step `skipped`, naming what it waited for. `completed=` replays a previous attempt's finished steps. Without `hooks=` nothing is recorded.

### Models

`WorkflowRunStep` carries `RUNNING` and `SKIPPED` statuses and a `detail` column holding the one line a step's outcome reported. `sequence` is the step's position in the declared pipeline, and `("run", "sequence")` is its identity — a skipped step never starts, so `started_at` cannot order a run.

A step that never ran is a `skipped` row naming what it was waiting on, not an absence.
