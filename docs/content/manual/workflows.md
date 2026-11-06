---
title: Workflows
type: docs
weight: 121
---

The workflow engine orchestrates multi-step agent tasks: steps that pass results forward, and listeners that watch the run. The [Views and Routing guide](/views-and-routing/#workflows) covers the public API; this page documents the definition schema, models, and how a run executes.

## Installing

Workflows are an opt-in app. Add it after the core app and run `migrate`:

```python
INSTALLED_APPS = [..., "django_ai_sdk", "django_ai_sdk.workflows"]
```

On start it reads every installed app's `workflows.py` (after the core app has set up the agents, so a step can name `MyAgent().agent_id`) and registers the [system checks](#system-checks). Without it the SDK has no workflow tables, checks or endpoints: the DRF router leaves `workflows/` out, and the Ninja `get_workflows_router()` is yours not to mount.

The app owns its tables from a single migration, `django_ai_sdk_workflows.0001_initial`. A database from 0.1.x has the workflow tables in an older shape, created by the core app's own `0001`; drop `django_ai_sdk_workflows`, `django_ai_sdk_workflow_runs` and `django_ai_sdk_workflow_run_steps` (and their rows in `django_migrations` for `django_ai_sdk_workflows`) before you migrate. Stored workflows and run history do not carry over.

## The Definition

```python
from django_ai_sdk.workflows import ListenerDefinition, FieldDefinition, StepDefinition, WorkflowDefinition

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
    listeners=[ListenerDefinition(type="thread_message", config={"step": "summary"})],
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

A field may also carry `enum` (the values a scalar may take, compiled to `Literal`), `default` (implying `required: false`, with the value used when the caller omits it), and `description` (for the model, the editor, and whoever reads the JSON). Nesting is capped at 8 levels and 32 fields per object, so a stored definition cannot declare a pathology the compiler has to walk. A whole definition is capped too: at most 50 steps, 20 listeners counted across the workflow and its steps, and 256 KiB as JSON. A definition over a cap is refused where it is parsed, before it is stored or run.

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
| `type` | `"agent"`, or a key of [`AI_SDK_WORKFLOWS["STEPS"]`](/manual/settings/) naming a registered `Step` class. |
| `name` | Required, unique. The key the result is filed under, the key later steps require, and the key the record is written under. |
| `agent_id` | Agent to run (resolved through `AgentService`). Required for `type: "agent"`, refused otherwise. |
| `requires` | Names of **earlier** steps this one reads. Each is passed to an agent step as a `[name]` user message. A name no earlier step declares is refused before any step runs. Inputs are not listed here — they are always available. |
| `history` | Agent steps only. The inputs this step sends as its conversation, in order: a `str` field is one user turn, a `list` field a conversation of messages, message dicts, or strings. The author names them; the engine never infers a transcript from the shape of a value. A name `input_fields` does not declare is refused, and so is a `system` message in it when the step runs: history holds only user and assistant turns. |
| `on_error` | `"stop"` stops the run; `"continue"` records the failure and runs the rest. |
| `output_fields` | Agent steps only. The agent runs with structured output: a pydantic model is built from the `{name: FieldDefinition}` map, and a result that does not come back as that model fails the step. A registered step returns what it returns, so declaring them there is refused. |
| `listeners` | Listeners that fire for this step alone. |
| `config` | Registered steps only. Settings handed to the step as `Step.config`, checked by the class's `validate_config` when the definition is stored or registered. A step takes none unless its class says otherwise. |

A definition cannot replace an agent's system prompt: `system_prompt_override` is refused, because whoever may store or run a definition is not thereby the agent's owner. Change the agent, or write the step in code, where `AgentStep.instructions` and `system_prompt(ctx)` remain.

Both kinds compile to `Step` objects and run through the same runner, so every run reads back from one record. An agent step compiles to a configured `AgentStep` — there is no second agent-step class for definitions.

### Registered step types

A definition can only name a `Step` class the deployment exposed:

```python
# settings.py
AI_SDK_WORKFLOWS = {
    "STEPS": {"gather_thread": "apps.agents.steps.GatherStep"},
}
```

The definition names and wires it — `name`, `requires`, `on_error`, `listeners` — and the class supplies the behaviour. A key whose path will not import is left out with a warning (and system check E002) rather than breaking the site, and a definition naming an unregistered type is refused with `ImproperlyConfigured` listing what is registered.

An agent step — whether a code `AgentStep` or a JSON `type: "agent"` step — is permission-checked (`Operation.CHAT`) against the user the run acts as. A registered Python step is never permission-checked — the registry is a deployment-level decision. Payload object ids in step inputs are **not** ACL-checked by the engine; hosts that need object auth must enforce it inside the step or refuse to expose those types at runtime.

### Three registries

| Registry | Setting / API | Holds |
| --- | --- | --- |
| Step types | `AI_SDK_WORKFLOWS["STEPS"]` | Python `Step` classes a JSON definition may compose |
| Workflow names | `register_workflow` / `Workflow.slug` | Runnable definitions, keyed by slug |
| Listeners | `AI_SDK_WORKFLOWS["LISTENERS"]` | Everything that watches a run rather than doing its work |

### Listeners

One kind of object, attached in two places. `ListenerDefinition(type, config)` names a key of `AI_SDK_WORKFLOWS["LISTENERS"]`; `config` is handed to the listener as it is built, so one registered class serves every definition that names it.

| Attached to | Sees |
| --- | --- |
| `WorkflowDefinition.listeners` | `on_run_start`, `on_step_start`, `on_step_end`, `on_run_end` — the run, and every step in it. |
| `StepDefinition.listeners` | `on_step_start`, `on_step_end` — that step alone. |

Reacting to a failure is a listener's job: a workflow-level listener sees every step that ends, whatever its status. A listener cannot fail the run — steps do the work, listeners watch it — so one that raises is logged against its name and the walk carries on. The package ships no listeners a definition can name — delivering a result somewhere is the host's business. `RunRecorder` is the one built-in: it is not a listener a definition names but the run's own record, passed to `run_steps` as `recorder=`, and an error it raises stops the run (a step does not run without its row).

A key whose path will not import is left out with a warning (and system check E003) rather than breaking the site; a definition naming a listener that is not registered is refused when it is created. `WorkflowService.list_listeners(user=...)` (`GET /workflows/listeners/`) returns `[{key, description}]` from the registry, to whoever may manage workflows.

```python
from django_ai_sdk.workflows import RunListener


class ConsoleLogListener(RunListener):
    description = "Log each step as it settles"

    async def on_step_end(self, ctx, step, outcome):
        print(ctx.workflow, step.name, outcome.status)
```

```python
# settings.py
AI_SDK_WORKFLOWS = {
    "LISTENERS": {"console_log": "apps.agents.listeners.ConsoleLogListener"},
}
```

## Models

| Model | Purpose |
| --- | --- |
| `Workflow` | A persisted, named workflow: `name`, `slug` (registry key, shared with code declarations), `definition` (JSON), `active`, `created_by`. |
| `WorkflowRun` | One execution: status `pending` / `running` / `completed` / `failed` / `interrupted`, `attempt` (how many times a worker claimed it), `heartbeat_at` (when its worker last showed it was alive), `workflow_key` (set by the server: the stored workflow's slug, the registered workflow's name, or `adhoc`), `workflow_definition` snapshot, `inputs` (what the caller supplied), `outputs` (`{step name: output}` for every completed step), `error`, `task_id`, `user`. |
| `StepRun` | Per-step progress: `sequence`, `step_name`, `output`, status `pending` / `running` / `completed` / `failed` / `skipped`, `detail` / `error`, timestamps. |

`registry.definition_of(workflow)` parses and validates the stored JSON. A stored definition is named by its row's slug: `create` and `update` overwrite the `name` it was sent with.

## Running

```python
from django_ai_sdk.workflows.services import WorkflowService

# A definition from your own code (never a client's JSON). A registered workflow
# needs RUN_WORKFLOW; any other definition also needs MANAGE_WORKFLOW.
run = await WorkflowService.run(workflow, inputs={"document": doc}, user=request.user)

# Persisted workflows
record = await WorkflowService.create("My workflow", workflow, user=request.user)
await WorkflowService.update(workflow_id, user=request.user, name=..., workflow=..., active=...)
await WorkflowService.delete(workflow_id, user=request.user)
await WorkflowService.get(workflow_id, user=request.user)
await WorkflowService.list_workflows(user=request.user, active_only=True)

# Start a workflow by its key: a registered workflow's name first, else an
# active stored workflow's slug.
run = await WorkflowService.start("process-document", inputs={"file": ref}, user=request.user)

# Queue a failed or interrupted run again. It reuses the run's own definition and
# inputs, and continues past the steps that completed.
run = await WorkflowService.resume(run_id, user=request.user)

# Run history
runs = await WorkflowService.list_runs(workflow_id, user=request.user)
run = await WorkflowService.get_run(run_id, user=request.user)  # prefetches steps; None if absent

# The inputs as a JSON Schema, for a form or a tool that composes a run
schema = await WorkflowService.get_inputs_schema(workflow_id, user=request.user)
```

A definition is checked when it is written, not only when it runs: `create` and `update` refuse one that could not execute (an unknown step type or listener, a duplicate step name, a `requires` no earlier step produces, or an `input_fields` / `output_fields` map that pydantic will not build). Inputs are checked on the call that queues the run. Every definition carries a `version` (currently `1`), stored with it, so a future format revision can gate on it rather than guess.

## Who may do what

`WorkflowDefaultPermission` is the default for the `workflow` [permission domain](/manual/permissions/):

| Operation | Who |
| --- | --- |
| Start a registered or active stored workflow (`start`, `POST /workflows/{key}/run/`), read a stored workflow's inputs schema | Any authenticated user. |
| Run a registered workflow with `WorkflowService.run` | Any authenticated user. |
| Create, edit, delete or list stored workflows, list the listeners; run any other definition with `WorkflowService.run` | Holders of `django_ai_sdk_workflows.change_workflow`. A holder manages every stored workflow. |
| Read a run (`GET /workflows/runs/{id}/`, or under its stored workflow's path) | The user it ran as, or holders of `django_ai_sdk_workflows.view_workflowrun`. |
| Resume a run (`resume`, `POST /workflows/runs/{id}/resume/`) | The user it ran as. An ad-hoc run (neither stored nor registered) also takes the manage permission, as starting it did. |

`is_staff` grants nothing here; a superuser holds every model permission. Over HTTP only a registered or stored workflow runs, by its key: there is no endpoint that runs a definition sent in the request. `POST /workflows/{key}/run/` answers `{run_id, status}`; follow the run at `GET /workflows/runs/{run_id}/`, which reads any run the caller may read, whether or not a stored workflow started it. Its `steps` carry each step's `status`, `output`, `detail` and `error`. `WorkflowService.run` is for definitions your own code builds. A run, a stored workflow or a resume the caller may not reach reads as absent (404), not forbidden. To change these rules, set `AI_SDK_PERMISSIONS["workflow"]`.

## Execution Model

`WorkflowService.start()` and `run()` create a `WorkflowRun` in `pending` and queue it with `enqueue_run` (`django_ai_sdk.workflows.runs`). It queues the `execute_workflow` background task (`django_tasks`) once the current transaction commits, so a worker never looks for a row that is not there yet; if queueing fails, the run is marked `failed` with the reason. Whether the task runs in a worker or inline in the calling request depends on the configured `TASKS` backend — see [Background Tasks](/manual/settings/#background-tasks).

The run row keeps a copy of the definition (`workflow_definition`) taken when the run was queued, and the worker runs that copy: editing a stored workflow changes later runs, not a queued or resumed one. The step types and listeners it names are resolved by key in the worker, so they must be registered in the worker's settings as well.

In the worker, `execute_run(run_id)`:

1. **Claims** the run with one conditional update: `pending` becomes `running`, `attempt` goes up by one and `heartbeat_at` is set. A second delivery of the same run, or a delivery of a run that already finished, claims nothing and stops there.
2. Validates the run's inputs against the definition's `input_fields`, when it declares any.
3. Compiles the definition to `Step` objects and listeners, and runs them through `run_steps` with a `RunRecorder` (JSON and code share this path).
4. **Replays** completed steps from a previous attempt, so a resumed run continues where it left off and does not re-date the rows it is trusting.
5. Marks the run `completed` with `{step name: output}` for every completed step. A failed step whose `on_error` is `stop` marks the run `failed` either way, but the task ends differently: a step that returned a failed outcome raises `WorkflowStopped`, which the task logs as an outcome; a step that raised lets its own exception fail the task, traceback and all, for your error reporting.

At every step the recorder refreshes `heartbeat_at` and checks that the run is still `running` under the attempt this worker claimed. When it is not (the run was swept, resumed and claimed again), it raises `RunSuperseded` before it writes anything, and the worker stops.

### A worker that dies

A killed worker leaves its run `running` with a heartbeat that stops moving. `workflows_sweep` finds those runs, marks them `interrupted`, and marks the step each one was in `failed`. It also marks `interrupted` a run still `pending` with no `task_id` past `--stale-after`: the process died after the run was committed and before it was queued. A pending run that has a `task_id` is the task backend's to deliver, and the sweep leaves it alone however long it waits:

```bash
manage.py workflows_sweep --stale-after 30m --limit 500 [--dry-run]
```

Schedule it (cron, a periodic task): each run takes the next `--limit` runs, oldest first. Pick a `--stale-after` longer than your slowest step, since a step does not beat while it runs. It shares its options and its pattern with `oversight_sweep`. An interrupted run is not retried by itself: `WorkflowService.resume(run_id)` or `POST /workflows/runs/{id}/resume/` queues it again.

### Beta limitations

Workflows ship as a beta. What the engine does not do yet, in one list:

- **No retries.** A failed step fails its run (or, with `on_error: continue`, is recorded and skipped past). Nothing tries it again by itself: `resume` queues the whole run again, which continues past the steps that completed.
- **No timeouts beyond the model's.** A model call is bounded only by its client; a step doing its own HTTP or subprocess work bounds its own client. `django_tasks` has no task timeout either.
- **Sequential steps only.** Steps run one at a time, in declared order: no fan-out and no join, even where two steps read nothing from each other.
- **Only model calls are overseen.** A step's turn is its model call: an `AgentStep`'s `agent.run()`, or `self.turn()` in a step of your own. A step that calls no model (a count, a fetch, a save) opens no turn and no policy judges it; one that calls a model outside a turn is warned about or refused (`UNGUARDED_MODEL_CALLS`).
- **No human approval.** A step never waits for a person. An oversight rule that blocks the step's turn fails the step; a `Queue` rule files a review item, and the step carries on with the answer it got.
- **The sweep is yours to schedule.** A killed worker, or a run that never reached the queue, is found only by `workflows_sweep`, run by the host's cron or periodic task. Until it runs, the run stays `running` or `pending`.
- **A message the task backend loses is not found.** A run whose task was queued (it has a `task_id`) but never delivered stays `pending`; the sweep does not guess at a backend's backlog. Mark it `interrupted` yourself to resume it.
- **Run data is plaintext and stays.** `inputs`, `outputs` and each `StepRun.output` are stored as plain JSON and kept until you delete the rows. Deleting a user nulls the run's `user` but keeps its inputs and outputs; deleting a thread does not reach them. Pass ids rather than content where you can, and prune old runs yourself.
- **Payload ids are not checked.** A document id or any other key in `inputs` is passed to the steps as given; a step that loads a row by it must check that `ctx.user` may read that row.
- **A run executes the definition captured when it was queued.** Editing or deactivating a stored workflow changes later runs, not one already queued or one that is resumed. The step types and listeners it names are looked up in the worker, so they must be registered there too.
- **Declared schemas are structural.** No value constraints (`min_length`, `ge`, …) and no validators: check those in a step.

In-turn multi-agent chat without a durable record belongs to **subagents**, not workflows.

## System checks

| Id | When |
| --- | --- |
| `ai_sdk.workflows.E001` | A workflow passed to `register_workflow` was refused (no steps, a name that is not a slug, an unknown step type or listener, …). |
| `ai_sdk.workflows.E002` | An `AI_SDK_WORKFLOWS["STEPS"]` entry does not import, or is not a `Step` subclass. |
| `ai_sdk.workflows.E003` | An `AI_SDK_WORKFLOWS["LISTENERS"]` entry does not import, or is not a `RunListener` subclass. |
| `ai_sdk.workflows.W001` | Deploy check (`check --deploy`): the `TASKS` backend runs workflow runs inline in the request that queued them (`ImmediateBackend`), never runs them (`DummyBackend`), or does not load. |

At run time an entry that does not import is left out of its registry with a warning, so a definition naming it is refused; the checks make that visible before the first run.

## Step Workflows

A queued run compiles its definition into `Step` objects and calls the `run_steps` walker. The [Step Workflows guide](/workflow-steps/) covers writing step classes and listeners; this section is the reference for the runtime objects.

### Step

| Attribute | Default | Purpose |
| --- | --- | --- |
| `name` | `""` | Required, unique within a pipeline. The key its outcome is recorded under and the key later steps require. |
| `requires` | `()` | Steps declared **earlier** that this one reads. A name that is not one of them is refused before anything runs. |
| `on_error` | `OnError.STOP` | `STOP` stops the run; `CONTINUE` records the failure and runs the rest. |
| `listeners` | `()` | Listeners that fire for this step alone. |

`config` is what a definition configured a registered step with (empty for a step written in code), and the classmethod `validate_config(config)` raises `ValueError` unless the type accepts it; by default a step accepts no config. `skip_if(ctx) -> str` returns a reason to skip or `""` to run. A skipped step is recorded with its reason and produces nothing, so whatever reads it skips in turn. `run(ctx) -> StepOutcome` is the work. `channel(ctx)` is the channel its turns are recorded on (`workflow:<key>:<step>`), and `turn(ctx, agent_or_policy, input=... | messages=...)` opens one for a model the step calls itself: a step's model calls must come from a turn ([Model calls outside a turn](/oversight/#model-calls-outside-a-turn)).

`AgentStep` adds `agent` / `agent_id`, `schema` (a real pydantic model, so nested output is expressible), `instructions` and `history` (the inputs to send as the conversation — the step names them; a `str` input is one turn, a list is a conversation); it implements `run` as one structured call, with async `system_prompt(ctx)`, `user_message(ctx)` and `messages(ctx)` overrides, which may query. A `None` result is reported as `failed`, not stored.

### WorkflowContext and StepOutcome

One context, for steps and listeners alike.

| `WorkflowContext` | Purpose |
| --- | --- |
| `inputs` / `input(name)` | What the caller supplied, coerced to the declared `input_fields`. |
| `steps` / `step(name)` | What has run so far, keyed by step name. `step()` returns the output of a **completed** step; `steps[name]` is the whole outcome, a failure's detail included. |
| `user` | The user the steps act as; forwarded to `AgentStep` calls. |
| `workflow` / `run_id` | The run's `workflow_key` and its id, for listeners that report on the run. |

| `StepOutcome` | Purpose |
| --- | --- |
| `output` | The step's value, filed under the step's own name. |
| `status` | `completed` / `failed` / `skipped`. Anything but `completed` blocks dependents. |
| `detail` | One line for a log or a timeline. |

### Running

`run_steps(steps, *, inputs=None, user=None, listeners=(), recorder=None, completed=None, workflow="", run_id="")` validates the pipeline, runs the steps **in the order they were declared**, and returns `{step_name: StepOutcome}`. A step runs once every step in its `requires` has completed; one that has not makes the step `skipped`, naming what it waited for. `completed=` replays a previous attempt's finished steps. `recorder=` is told about every step before the listeners are, and its errors stop the run; without it nothing is recorded.

### Models

`StepRun` carries `RUNNING` and `SKIPPED` statuses and a `detail` column holding the one line a step's outcome reported. `sequence` is the step's position in the declared pipeline, and `("run", "sequence")` is its identity — a skipped step never starts, so `started_at` cannot order a run.

A step that never ran is a `skipped` row naming what it was waiting on, not an absence.
