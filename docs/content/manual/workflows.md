---
title: Workflows
type: docs
weight: 121
---

The workflow engine orchestrates multi-step agent tasks: steps that pass results forward, and listeners that watch the run. The [Views and Routing guide](/views-and-routing/#workflows) covers the public API; this page documents the definition schema, models, and how a run executes.

## Installing

Workflows are an opt-in app. Runs execute on DBOS, so it comes with `"django_ai_sdk.durable"` (the `dbos` extra). Add both after the core app, run `migrate`, and run `manage.py ai_sdk_worker`; see [Durable runs on DBOS](#durable-runs-on-dbos):

```python
INSTALLED_APPS = [..., "django_ai_sdk", "django_ai_sdk.workflows", "django_ai_sdk.durable"]
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
| `WorkflowRun` | One execution: status `pending` / `running` / `completed` / `failed` / `cancelled`, `trigger` (what started it: `manual`, `cron:<trigger>` or `event:<trigger>`), `attempt` (how many times a worker claimed it), `heartbeat_at` (when it last made progress: the claim or a step boundary), `workflow_key` (set by the server: the stored workflow's slug, the registered workflow's name, or `adhoc`), `workflow_definition` snapshot, `inputs` (the declared inputs the caller supplied; undeclared keys are dropped), `outputs` (`{step name: output}` for every completed step), `error`, `task_id` (the DBOS workflow id of the current attempt), `user` (who it acts as), `started_by` (the person who asked for it, when one did: the starter of a manual run, or the operator who fired a trigger by hand). |
| `StepRun` | Per-step progress: `sequence`, `step_name`, `output`, status `pending` / `running` / `completed` / `failed` / `skipped` / `waiting` (an approval step), `detail` / `error`, `turn_id` (the oversight turn its model call opened), timestamps. |

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

# Queue a failed or cancelled run again. It reuses the run's own
# definition and inputs, and continues past the steps that completed.
run = await WorkflowService.resume(run_id, user=request.user)

# Stop a pending or running run. The step in flight finishes; none after it starts.
run = await WorkflowService.cancel(run_id, user=request.user)

# Decide an approval step the run waits on. Never on a run that acts as you or that you started.
run = await WorkflowService.decide(run_id, "approve", approved=True, note="ok", user=request.user)

# Triggers: list them with their schedule's state, pause or resume a cron one, fire it now.
triggers = await WorkflowService.list_triggers(user=request.user)
await WorkflowService.set_trigger_paused("nightly-report", "nightly", paused=True, user=request.user)
await WorkflowService.fire_trigger("nightly-report", "nightly", user=request.user)

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
| Cancel a run (`cancel`, `POST /workflows/runs/{id}/cancel/`) | The user it ran as. |
| Decide an approval step (`decide`, `POST /workflows/runs/{id}/steps/{step}/decision/`) | Holders of `django_ai_sdk_workflows.approve_steprun`, and never on a run that acts as them: nobody approves their own run, a superuser included. |
| List triggers (`GET /workflows/triggers/`) | Any authenticated user. |
| Pause, resume or fire a cron trigger (`POST /workflows/triggers/{key}/{trigger}/pause/` and `.../fire/`) | The manage permission. |

`is_staff` grants nothing here; a superuser holds every model permission. Over HTTP only a registered or stored workflow runs, by its key: there is no endpoint that runs a definition sent in the request. `POST /workflows/{key}/run/` answers `{run_id, status}`; follow the run at `GET /workflows/runs/{run_id}/`, which reads any run the caller may read, whether or not a stored workflow started it. Its `steps` carry each step's `status`, `output`, `detail` and `error`. `WorkflowService.run` is for definitions your own code builds. A run, a stored workflow or a resume the caller may not reach reads as absent (404), not forbidden. To change these rules, set `AI_SDK_PERMISSIONS["workflow"]`.

## Execution Model

`WorkflowService.start()` and `run()` create a `WorkflowRun` in `pending` and queue it with `enqueue_run` (`django_ai_sdk.workflows.runs`). It hands the run to DBOS once the current transaction commits, so the worker never looks for a row that is not there yet; if queueing fails, the run is marked `failed` with the reason. A process that dies between the commit and the queueing leaves the run `pending`; the worker queues every pending run again when it starts, which runs nothing twice because the DBOS workflow id is fixed per attempt.

The run row keeps a copy of the definition (`workflow_definition`) taken when the run was queued, and the worker runs that copy: editing a stored workflow changes later runs, not a queued or resumed one. The step types and listeners it names are resolved by key in the worker, so they must be registered in the worker's settings as well.

In the worker, the DBOS workflow calls `execute_run(run_id)`, with each unit checkpointed (below). It:

1. **Claims** the run with one conditional update: `pending` becomes `running`, `attempt` goes up by one and `heartbeat_at` is set. A second delivery of the same run, or a delivery of a run that already finished, claims nothing and stops there. The claim stores the DBOS workflow id as `task_id`, so a workflow whose worker died after the claim but before DBOS recorded it finds the run running under its own id and takes it up again.
2. Validates the run's inputs against the definition's `input_fields`, when it declares any.
3. Compiles the definition to `Step` objects and listeners, and runs them through `run_steps` with a `RunRecorder` (JSON and code share this path).
4. **Replays** completed steps from a previous attempt, so a resumed run continues where it left off and does not re-date the rows it is trusting.
5. Marks the run `completed` with `{step name: output}` for every completed step. A failed step whose `on_error` is `stop` marks the run `failed`.

At every step the recorder refreshes `heartbeat_at` and checks that the run is still `running` under the attempt this worker claimed. When it is not (the run was cancelled, or resumed and claimed again), it raises `RunSuperseded` before it writes anything, and the walk stops. A step whose row this attempt already completed (its worker died after writing it, before DBOS recorded the step) is not run again: the row's outcome is taken as the step's.

Called directly, `execute_run(run_id)` does the same walk in the calling process without checkpoints, as an inline demo endpoint does. Such a run cannot wait: an approval step in it fails at once. In a host's tests, `django_ai_sdk.durable.testing.run_inline(monkeypatch)` in an autouse fixture makes every queued run execute that way, with no DBOS tables needed.

## Durable runs on DBOS

[DBOS](https://docs.dbos.dev) runs every workflow run, in the project's own database (Postgres in production, its own schema `dbos`; a file next to a SQLite database in development). There is no other service: `manage.py ai_sdk_worker` is the worker, and creates or upgrades DBOS's tables as it launches. If a web process may queue a run before the first worker has started, run `manage.py dbos_migrate` in your release step. A host's own DBOS schedules hang on the `durable.signals.worker_launched` signal, sent once DBOS is up.

```python
INSTALLED_APPS = [..., "django_ai_sdk", "django_ai_sdk.workflows", "django_ai_sdk.durable"]
TASKS = {"default": {"BACKEND": "django_ai_sdk.durable.tasks.DBOSTaskBackend"}}
AI_SDK_DBOS = {"VERSION": "1"}  # every key optional; see Settings
```

- **The same walk, checkpointed.** A run is the DBOS workflow `ai_sdk.workflow_run`. The runner hands each unit (the claim, the run's start, each step, the run's end) to DBOS as a step, which records its result. A worker that dies mid-run is recovered by the next one: the units that finished are replayed from their records, and only the step in flight runs again.
- **At least once.** A step runs at least once, not exactly once: the step in flight when a worker dies runs again on recovery, and so does one whose worker died after it finished but before DBOS recorded it. A step with an effect outside the run's own rows (a payment, a booking, a message) makes it idempotent by `self.idempotency_key(ctx)`, `"<run id>:<step name>"`, the same on every recovery and resume: pass it to the API as its idempotency key, or store it with the effect and skip the effect when it is already there.
- **Stale runs.** A run left `running` by a DBOS workflow that no longer runs (it ended, was cancelled outside the SDK, or ran out of recovery attempts) would never close. `fail_stale_runs()` (`django_ai_sdk.workflows.runs`) checks every run whose `heartbeat_at` is older than `AI_SDK_WORKFLOWS["STALE_AFTER_SECONDS"]` (15 minutes) against DBOS, and fails the ones DBOS no longer runs; `resume` takes them up again. A run DBOS still runs is never touched, however old its heartbeat: a long step, an approval waiting for days, a run waiting for recovery. The worker calls it at start and every minute.
- **Ids.** The run's id is its first attempt's DBOS workflow id; a resumed run's attempt is `<run id>:<attempt>`. Queueing an id that exists runs nothing.
- **Pinned version.** `AI_SDK_DBOS["VERSION"]` is fixed rather than a hash of the code, so a deploy recovers the runs the last one left. Each run carries its definition, so only a change to step code that an unfinished run cannot replay calls for a new version.
- **JSON, not pickle.** DBOS stores inputs, step results and errors as JSON (`portable_json`); loading them runs no code.
- **Cancel.** `cancel` marks the run `cancelled` and tells DBOS, which stops it before its next step. The step in flight finishes and its outcome is recorded, so a resume does not do it again. A run waiting for an approval is woken first, since DBOS's cancel alone leaves it waiting. `resume` takes it up again.
- **One worker at a time.** Without DBOS Conductor, a worker recovers every unfinished run of its executor (`"local"`) at launch, so two at once would both run them. `ai_sdk_worker` takes a Postgres advisory lock first: a second worker waits as a standby until the first stops, which also makes a deploy that starts the new worker before stopping the old one safe. Concurrency comes from the queues, not from more workers.
- **No cap on the `workflows` queue.** A run waiting for an approval counts against its queue's `worker_concurrency` for as long as it waits, so a cap there lets a few waiting approvals stop every other run. Bound model spend in the agents, not in the queue.
- **One queue system.** `DBOSTaskBackend` runs every other `django_tasks` task (oversight's reflection and notifications, document processing) as a DBOS workflow of one step on the queue its `queue_name` names, so `db_worker` is not needed. `AI_SDK_DBOS["QUEUES"]` declares the queues and their concurrency.

Web processes never launch DBOS: they queue, send decisions and manage schedules through a `DBOSClient` on the same tables.

## Approval steps

The built-in step type `approval` waits for a person:

```json
{"type": "approval", "name": "approve", "requires": ["check"],
 "config": {"prompt": "Book this invoice?", "timeout_seconds": 86400}}
```

The step is set `waiting` with the prompt as its `detail`, and the run waits durably (`DBOS.recv`) across restarts and deploys. A decision comes from `WorkflowService.decide` or `POST /workflows/runs/{id}/steps/{step}/decision/` (`{"approved": true, "note": "..."}`). It takes the `approve_steprun` permission and is refused on a run that acts as the approver or that they started (such as a trigger they fired by hand that runs as a bot). The first decision wins, and a decision that comes after the wait ran out raises `DecisionConflict` (`django_ai_sdk.workflows.services`; a `UserError` with code `conflict`, answered 409 by the SDK's views), as does one for a step already decided or cancelled. The timeout and a decision race on the same conditional update, so a decision that was accepted is never lost to the timeout. Approved, the step completes with `{approved, by, note}`; rejected, or without a decision within `timeout_seconds` (default a week), it fails and its `on_error` decides what follows.

## Triggers

A workflow declares what starts it besides a person or code asking:

```python
WorkflowDefinition(
    name="wiki-gardener",
    input_fields={"mode": FieldDefinition(enum=["hourly", "morning"])},
    steps=[...],
    triggers=[
        CronTrigger(name="hourly", cron="5 * * * *", timezone="Europe/Amsterdam",
                    run_as="wiki-bot", inputs={"mode": "hourly"}),
        EventTrigger(name="inbox", event="telegram.message", inputs={"text": "{text}"}),
    ],
)
```

- **Cron.** The worker keeps one DBOS schedule per enabled cron trigger, named `<workflow>:<trigger>`, and syncs them all on start; saving a stored workflow syncs that workflow's schedules only. DBOS fires each occurrence once; the run's id derives from the occurrence, so a repeat opens nothing new. Five fields only, so nothing fires more than once a minute. An operator pauses a trigger at runtime and a deploy keeps that. `AI_SDK_WORKFLOWS["SCHEDULES_ENABLED"] = False` makes every occurrence start nothing and leaves the schedules as they are, so turning it back on needs no clean-up. Firing a trigger by hand works either way, and records who fired it.
- **Event.** Every webhook delivery is reported as `<integration>.message`, with `text` and `conversation_id` in the payload. Report other events with `await start_for_event("invoice.received", payload, idempotency_key=...)` (`django_ai_sdk.workflows.triggers`). Every enabled trigger on that event starts a run whose id derives from the key, so a redelivered message starts nothing new.
- **Inputs.** `"{word}"` in a string is filled in: from `name`, `workflow`, `scheduled_at` and `last_run_at` for cron, from the payload for an event. A string that is only `"{word}"` takes the value with its type. Every input a trigger fills must be declared.
- **Who it acts as.** `run_as` (a USERNAME_FIELD value) on a workflow declared in code. A stored workflow's runs act as the person who created it: a submitted definition that sets `run_as` is refused, or authoring a workflow would be a way to act as anyone.

### Beta limitations

Workflows ship as a beta. What the engine does not do yet, in one list:

- **No retries.** A failed step fails its run (or, with `on_error: continue`, is recorded and skipped past). Nothing tries it again by itself: `resume` queues the whole run again, which continues past the steps that completed.
- **No timeouts beyond the model's.** A model call is bounded only by its client; a step doing its own HTTP or subprocess work bounds its own client.
- **Sequential steps only.** Steps run one at a time, in declared order: no fan-out and no join, even where two steps read nothing from each other.
- **Only model calls are overseen.** A step's turn is its model call: an `AgentStep`'s `agent.run()`, or `self.turn()` in a step of your own. A step that calls no model (a count, a fetch, a save) opens no turn and no policy judges it; one that calls a model outside a turn is refused, or with `UNGUARDED_MODEL_CALLS` set to `warn`, logged.
- **Oversight does not wait.** An oversight rule that blocks a step's turn fails the step; a `Queue` rule files a review item, and the step carries on with the answer it got. To have a person decide before the run goes on, put an [approval step](#approval-steps) after it.
- **One worker.** Workers do not share the load; a second one is a standby (see [Durable runs on DBOS](#durable-runs-on-dbos)).
- **Step code changes replay.** A run recovered after a deploy replays its finished units by name. Changing which units a run goes through (a new step type's internals are fine; reordering the runner's units is not) calls for a new `AI_SDK_DBOS["VERSION"]`, and the old version's unfinished runs are left to finish or be resumed.
- **Run data is plaintext and stays.** `inputs`, `outputs` and each `StepRun.output` are stored as plain JSON and kept until you delete the rows. Deleting a user nulls the run's `user` but keeps its inputs and outputs; deleting a thread does not reach them. Pass ids rather than content where you can, and prune old runs yourself.
- **Payload ids are not checked.** A document id or any other key in `inputs` is passed to the steps as given; a step that loads a row by it must check that `ctx.user` may read that row.
- **A run executes the definition captured when it was queued.** Editing a stored workflow changes later runs, not one already queued or one that is resumed. Deactivating or deleting it, or removing a registered one from code, stops its runs from being resumed. The step types and listeners it names are looked up in the worker, so they must be registered there too.
- **Declared schemas are structural.** No value constraints (`min_length`, `ge`, …) and no validators: check those in a step.

In-turn multi-agent chat without a durable record belongs to **subagents**, not workflows.

## System checks

| Id | When |
| --- | --- |
| `ai_sdk.workflows.E001` | A workflow passed to `register_workflow` was refused (no steps, a name that is not a slug, an unknown step type or listener, …). |
| `ai_sdk.workflows.E002` | An `AI_SDK_WORKFLOWS["STEPS"]` entry does not import, or is not a `Step` subclass. |
| `ai_sdk.workflows.E003` | An `AI_SDK_WORKFLOWS["LISTENERS"]` entry does not import, or is not a `RunListener` subclass. |
| `ai_sdk.workflows.E004` | `AI_SDK_WORKFLOWS["STEPS"]` has the key `agent` or `approval`, which the built-in step types reserve. |
| `ai_sdk.workflows.E005` | A 0.1 setting is still set: `AI_SDK_WORKFLOW_STEPS` (now `AI_SDK_WORKFLOWS["STEPS"]`) or `AI_SDK_WORKFLOW_ACTIONS` (now `AI_SDK_WORKFLOWS["LISTENERS"]`). |
| `ai_sdk.workflows.E012` | `AI_SDK_WORKFLOWS` is not a dict, or one of its keys has a value of the wrong type. |

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
