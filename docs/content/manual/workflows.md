---
title: Workflows
sidebar:
  order: 121
---

The workflow engine runs multi-step agent tasks. Steps pass results forward, and listeners watch the run. The [Views and Routing guide](../serving/views-and-routing.md#workflows) covers the public API. This page documents the definition schema, the models and how a run executes.

## Installing

:badge[Beta]{variant="secondary"} :badge[dbos extra]{variant="outline"}

Workflows are an opt-in app. Runs execute on DBOS, so the app needs `"django_ai_sdk.durable"` too, from the `dbos` extra. Add both after the core app, run `migrate`, and run `manage.py ai_sdk_worker`. See [Durable runs on DBOS](#durable-runs-on-dbos).

```python
INSTALLED_APPS = [..., "django_ai_sdk", "django_ai_sdk.workflows", "django_ai_sdk.durable"]
```

On start the app reads every installed app's `workflows.py` and registers the [system checks](#system-checks). It reads them after the core app has set up the agents, so a step can name `MyAgent().agent_id`. Without the app the SDK has no workflow tables, checks or endpoints. The DRF router leaves `workflows/` out, and you do not mount the Ninja `get_workflows_router()`.

:::details[Upgrading from 0.1.2a21 or earlier]
When you upgrade from 0.1.2a21 or earlier, `migrate` updates the tables in place with `django_ai_sdk_workflows.0004_dbos_runs`. Stored workflows and run history carry over, and each old run gets its workflow's slug as `workflow_key`. A stored workflow whose slug was `adhoc` is renamed to `adhoc-<id>`, because `adhoc` now names runs of no stored workflow. A run that the old runner left running has no DBOS workflow, so the worker's stale-run check fails it on its first pass. Resume it to finish it on DBOS.
:::

## The definition

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

A run's state has two halves, and nothing crosses between them. `ctx.input(name)` is what the caller supplied. `ctx.step(name)` is what an earlier step produced, keyed by **that step's own name**. A step needs no `output_key`, and an input and a step can share a name without renaming either.

### Declared inputs

`input_fields` compiles to a pydantic model, the same way `output_fields` does. The engine validates a run's inputs against it once, before the run is queued. A caller who left something out gets the error from the call that queued the run, not three steps later. Undeclared extras are dropped, because the schema is the contract.

The shape is the whole contract, with its structure, enums and defaults. A declared schema has to survive a dump, cross the queue, and be read back by an editor or an agent. Behaviour does not cross. Validators and value constraints such as `min_length` or `ge` belong in a step, not in a field.

| `type` | Becomes |
| --- | --- |
| `str` / `int` / `float` / `bool` | The Python type. |
| `list` / `dict` | An open JSON container the host shapes itself. |
| `list` + `items` | A typed list; `items` is another `FieldDefinition`, so elements are checked too. |
| `object` + `fields` | A typed record; `fields` is a `{name: FieldDefinition}` map, recursively. |

A field can also carry `enum`, `default` and `description`. `enum` lists the values a scalar may take and compiles to `Literal`. `default` implies `required: false` and gives the value used when the caller omits the field. `description` is for the model, the editor and whoever reads the JSON. Nesting is capped at 8 levels and 32 fields per object, so a stored definition cannot make the compiler walk an unbounded schema. A whole definition is capped as well, at 50 steps, 20 listeners counted across the workflow and its steps, and 256 KiB as JSON. The parser refuses a definition over a cap, before it is stored or run.

A workflow written in code can declare its inputs with a pydantic model instead, as `WorkflowDefinition(inputs=MyModel, ...)`. The model is normalized into the same fields and is not stored on the definition, so stored and code-declared definitions share one wire format. Anything pydantic expresses as a shape carries over. Validators, `default_factory` and `Annotated` constraints do not, and the engine refuses a model that has any of them instead of applying it partly.

```python
from pydantic import BaseModel
from typing import Literal

class DigestInputs(BaseModel):
    thread: str
    tone: Literal["brief", "thorough"] = "brief"
```

There is no `file` type and no `messages` type. A file is whatever the host already stores it as, such as a `str` id or a `dict` reference under a key the host picks. The first step that reads it resolves it. A conversation is an ordinary input, and the step that names it in `history` turns it into the transcript, as described below.

A definition that declares no `input_fields` accepts any inputs, so a pipeline written in code does not have to describe itself in JSON.

There is no workflow-level `output_fields`. A run's `outputs` is the map of step outputs keyed by step name, and each step already declares its own shape.

### Step semantics

An agent step runs its `agent_id` with `agent.run()`, without streaming. Its result is filed under the step's `name`.

| Field | Purpose |
| --- | --- |
| `type` | `"agent"`, or a key of [`AI_SDK_WORKFLOWS["STEPS"]`](settings.md) naming a registered `Step` class. |
| `name` | Required, unique. The key the result is filed under, the key later steps require, and the key the record is written under. |
| `agent_id` | Agent to run, resolved through `AgentService`. Required for `type: "agent"`, refused otherwise. |
| `requires` | Names of **earlier** steps this one reads. Each is passed to an agent step as a `[name]` user message. A name no earlier step declares is refused before any step runs. Inputs are not listed here, because they are always available. |
| `history` | Agent steps only. The inputs this step sends as its conversation, in order. A `str` field is one user turn. A `list` field is a conversation of messages, message dicts or strings. The author names them, and the engine never infers a transcript from the shape of a value. The engine refuses a name that `input_fields` does not declare. When the step runs, it also refuses a `system` message in it, because history holds only user and assistant turns. |
| `on_error` | `"stop"` stops the run. `"continue"` records the failure and runs the rest. |
| `output_fields` | Agent steps only. The agent runs with structured output. The engine builds a pydantic model from the `{name: FieldDefinition}` map, and a result that does not match that model fails the step. A registered step decides its own return value, so the engine refuses `output_fields` on one. |
| `listeners` | Listeners that fire for this step alone. |
| `config` | Registered steps only. Settings handed to the step as `Step.config`, checked by the class's `validate_config` when the definition is stored or registered. A step takes none unless its class says otherwise. |

A definition cannot replace an agent's system prompt. The engine refuses `system_prompt_override`, because a user who may store or run a definition does not own the agent. Change the agent, or write the step in code, where `AgentStep.instructions` and `system_prompt(ctx)` are available.

Both kinds compile to `Step` objects and run through the same runner, so every run reads back from one record. An agent step compiles to a configured `AgentStep`. Definitions have no separate agent-step class.

### Registered step types

A definition can only name a `Step` class that the deployment registers:

```python
# settings.py
AI_SDK_WORKFLOWS = {
    "STEPS": {"gather_thread": "apps.agents.steps.GatherStep"},
}
```

The definition names and wires the step with `name`, `requires`, `on_error` and `listeners`, and the class supplies the behaviour. A key whose path does not import is left out with a warning and system check E002, so the site keeps working. A definition that names an unregistered type is refused with an `ImproperlyConfigured` error that lists the registered types.

The engine checks `Operation.CHAT` for every agent step against the user the run acts as. That covers a code `AgentStep` and a JSON `type: "agent"` step. It never checks permissions for a registered Python step, because registering it is a deployment decision.

:::caution
The engine does **not** check object ids in step inputs against any ACL. A host that needs object authorization must enforce it inside the step, or not register those step types.
:::

### Three registries

| Registry | Setting / API | Holds |
| --- | --- | --- |
| Step types | `AI_SDK_WORKFLOWS["STEPS"]` | Python `Step` classes a JSON definition may compose |
| Workflow names | `register_workflow` / `Workflow.slug` | Runnable definitions, keyed by slug |
| Listeners | `AI_SDK_WORKFLOWS["LISTENERS"]` | Everything that watches a run rather than doing its work |

### Listeners

A listener is one kind of object that attaches in two places. `ListenerDefinition(type, config)` names a key of `AI_SDK_WORKFLOWS["LISTENERS"]`. The engine passes `config` to the listener when it builds it, so one registered class serves every definition that names it.

| Attached to | Sees |
| --- | --- |
| `WorkflowDefinition.listeners` | `on_run_start`, `on_step_start`, `on_step_end`, `on_run_end`, for the run and every step in it. |
| `StepDefinition.listeners` | `on_step_start`, `on_step_end`, for that step alone. |

A listener reacts to failures. A workflow-level listener sees every step that ends, whatever its status. A listener cannot fail the run, because steps do the work and listeners only watch it. When a listener raises, the engine logs the error against its name and the run continues. The package ships no listeners that a definition can name, because delivering a result is up to the host. The one built-in is `RunRecorder`. A definition does not name it. It is the run's own record, passed to `run_steps` as `recorder=`. An error it raises stops the run, because a step does not run without its row.

A key whose path does not import is left out with a warning and system check E003, so the site keeps working. A definition that names an unregistered listener is refused when it is created. `WorkflowService.list_listeners(user=...)` (`GET /workflows/listeners/`) returns `[{key, description}]` from the registry to users who may manage workflows.

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
| `Workflow` | A persisted, named workflow: `name`, `slug` (registry key, shared with code declarations), `definition` (JSON), `active`, `created_by` (who last wrote its steps or switched it back on, and so who its triggered runs act as). |
| `WorkflowRun` | One execution: status `pending` / `running` / `completed` / `failed` / `cancelled`, `trigger` (what started it: `manual`, `cron:<trigger>` or `event:<trigger>`), `attempt` (how many times a worker claimed it), `heartbeat_at` (when it last made progress: the claim or a step boundary), `workflow_key` (set by the server: the stored workflow's slug, the registered workflow's name, or `adhoc`), `workflow_definition` snapshot, `inputs` (the declared inputs the caller supplied; undeclared keys are dropped), `outputs` (`{step name: output}` for every completed step), `error`, `task_id` (the DBOS workflow id of the current attempt), `user` (who it acts as), `started_by` (the person who asked for it, when one did: the starter of a manual run, or the operator who fired a trigger by hand). |
| `StepRun` | Per-step progress: `sequence`, `step_name`, `output`, status `pending` / `running` / `completed` / `failed` / `skipped` / `waiting` (an approval step), `detail` / `error`, `turn_id` (the oversight turn its model call opened), timestamps. |

`registry.definition_of(workflow)` parses and validates the stored JSON. A stored definition takes its name from its row's slug, so `create` and `update` overwrite the `name` it was sent with.

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

The engine checks a definition when it is written, not only when it runs. `create` and `update` refuse a definition that could not execute. Examples are an unknown step type or listener, a duplicate step name, a `requires` that no earlier step produces, or an `input_fields` or `output_fields` map that pydantic cannot build. Inputs are checked on the call that queues the run. Every definition stores a `version`, currently `1`, so a future format revision can branch on it.

## Who may do what

`WorkflowDefaultPermission` is the default for the `workflow` [permission domain](permissions.md):

| Operation | Who |
| --- | --- |
| Start a registered or active stored workflow (`start`, `POST /workflows/{key}/run/`), read a stored workflow's inputs schema | Any authenticated user. |
| Run a registered workflow with `WorkflowService.run` | Any authenticated user. |
| Create, edit, delete or list stored workflows, list the listeners; run any other definition with `WorkflowService.run` | Holders of `django_ai_sdk_workflows.change_workflow`. A holder manages every stored workflow. |
| Read a run (`GET /workflows/runs/{id}/`, or under its stored workflow's path) | The user it ran as, or holders of `django_ai_sdk_workflows.view_workflowrun`. |
| Resume a run (`resume`, `POST /workflows/runs/{id}/resume/`) | The user it ran as. An ad-hoc run, neither stored nor registered, also needs the manage permission, as starting it did. |
| Cancel a run (`cancel`, `POST /workflows/runs/{id}/cancel/`) | The user it ran as. |
| Decide an approval step (`decide`, `POST /workflows/runs/{id}/steps/{step}/decision/`) | Holders of `django_ai_sdk_workflows.approve_steprun`, and never on a run that acts as them. Nobody approves their own run, including a superuser. |
| List triggers (`GET /workflows/triggers/`) | Any authenticated user. |
| Pause, resume or fire a cron trigger (`POST /workflows/triggers/{key}/{trigger}/pause/` and `.../fire/`) | The manage permission. A stored workflow's trigger is fired by hand only by its author, since the run acts as them. |

`is_staff` grants nothing here, and a superuser holds every model permission. Over HTTP only a registered or stored workflow runs, by its key. No endpoint runs a definition sent in the request. `POST /workflows/{key}/run/` answers `{run_id, status}`. Follow the run at `GET /workflows/runs/{run_id}/`, which returns any run the caller may read, whether or not a stored workflow started it. Its `steps` carry each step's `status`, `output`, `detail` and `error`. `WorkflowService.run` is for definitions that your own code builds. A run, stored workflow or resume the caller may not reach returns 404, not 403. To change these rules, set `AI_SDK_PERMISSIONS["workflow"]`.

## Execution model

`WorkflowService.start()` and `run()` create a `WorkflowRun` in `pending` and queue it with `enqueue_run` (`django_ai_sdk.workflows.runs`). `enqueue_run` hands the run to DBOS once the current transaction commits, so the worker never looks for a row that does not exist yet. If queueing fails, the run is marked `failed` with the reason. A process that dies between the commit and the queueing leaves the run `pending`. The worker queues every pending run again when it starts. This runs nothing twice, because the DBOS workflow id is fixed per attempt.

The run row keeps a copy of the definition in `workflow_definition`, taken when the run was queued, and the worker runs that copy. Editing a stored workflow changes later runs, not a queued one, and a run from before the edit no longer resumes. The worker resolves the step types and listeners the definition names by key, so they must be registered in the worker's settings as well.

In the worker, the DBOS workflow calls `execute_run(run_id)` and checkpoints each unit, as described below. `execute_run` does the following:

::::trace[execute_run(run_id)]
:::step[Claims the run]{icon="square-terminal"}
**Claims** the run with one conditional update: `pending` becomes `running`, `attempt` goes up by one and `heartbeat_at` is set. A second delivery of the same run, or a delivery of a run that already finished, claims nothing and stops there. The claim stores the DBOS workflow id as `task_id`. If the worker dies after the claim but before DBOS records it, the recovered workflow finds the run running under its own id and continues it.
:::

:::step[Validates the inputs]{icon="search"}
Validates the run's inputs against the definition's `input_fields`, when it declares any.
:::

:::step[Compiles and runs the steps]{icon="code"}
Compiles the definition to `Step` objects and listeners, and runs them through `run_steps` with a `RunRecorder`. JSON and code definitions share this path.
:::

:::step[Replays completed steps]{icon="file-text"}
**Replays** completed steps from a previous attempt, so a resumed run continues where it stopped and keeps the timestamps of the rows it reuses.
:::

:::step[Marks the run completed]
Marks the run `completed` with `{step name: output}` for every completed step. A failed step whose `on_error` is `stop` marks the run `failed`.
:::
::::

At every step the recorder refreshes `heartbeat_at` and checks that the run is still `running` under the attempt this worker claimed. When the run was cancelled, or resumed and claimed again, the recorder raises `RunSuperseded` before it writes anything, and the walk stops. A step whose row this attempt already completed does not run again, and the engine uses the row's outcome. That happens when the worker died after writing the row but before DBOS recorded the step.

Called directly, `execute_run(run_id)` does the same walk in the calling process without checkpoints, as an inline demo endpoint does. Such a run cannot wait, so an approval step in it fails at once. In a host's tests, `django_ai_sdk.durable.testing.run_inline(monkeypatch)` in an autouse fixture makes every queued run execute that way, without DBOS tables.

## Durable runs on DBOS

[DBOS](https://docs.dbos.dev) runs every workflow run in the project's own database. In production that is Postgres, in a schema named `dbos`. In development it is a file next to the SQLite database. No other service is needed. `manage.py ai_sdk_worker` is the worker, and it creates or upgrades DBOS's tables at launch. If a web process may queue a run before the first worker has started, run `manage.py dbos_migrate` in your release step. A host registers its own DBOS schedules on the `durable.signals.worker_launched` signal, which is sent once DBOS is up.

```python
INSTALLED_APPS = [..., "django_ai_sdk", "django_ai_sdk.workflows", "django_ai_sdk.durable"]
TASKS = {"default": {"BACKEND": "django_ai_sdk.durable.tasks.DBOSTaskBackend"}}
AI_SDK_DBOS = {"VERSION": "1"}  # every key optional; see Settings
```

- **The same walk, checkpointed.** A run is the DBOS workflow `ai_sdk.workflow_run`. The runner hands each unit to DBOS as a step, and DBOS records its result. The units are the claim, the run's start, each step and the run's end. When a worker dies mid-run, the next worker recovers the run. It replays the finished units from their records, and only the step in flight runs again.
- **At least once.** A step runs at least once, not exactly once. The step in flight when a worker dies runs again on recovery. So does a step whose worker died after it finished but before DBOS recorded it. A step with an effect outside the run's own rows, such as a payment, a booking or a message, makes that effect idempotent with `self.idempotency_key(ctx)`. The key is `"<run id>:<step name>"` and stays the same on every recovery and resume. Pass it to the API as its idempotency key, or store it with the effect and skip the effect when it already exists.
- **Stale runs.** A run left `running` by a DBOS workflow that no longer runs would never close. The DBOS workflow may have ended, been cancelled outside the SDK, or run out of recovery attempts. `fail_stale_runs()` (`django_ai_sdk.workflows.runs`) checks every run whose `heartbeat_at` is older than `AI_SDK_WORKFLOWS["STALE_AFTER_SECONDS"]` (15 minutes) against DBOS, and fails the ones DBOS no longer runs. `resume` continues them. A run that DBOS still runs is never touched, however old its heartbeat, for example a long step, an approval waiting for days or a run waiting for recovery. The worker calls `fail_stale_runs()` at start and every minute.
- **Ids.** The run's id is its first attempt's DBOS workflow id. A resumed run's attempt is `<run id>:<attempt>`. Queueing an id that exists runs nothing.
- **Pinned version.** `AI_SDK_DBOS["VERSION"]` is a fixed value, not a hash of the code, so a deploy recovers the runs the previous one left. Each run carries its definition, so you need a new version only for a change to step code that an unfinished run cannot replay. A worker runs only its own version's runs. Runs left unfinished under the old version, such as runs waiting for an approval, wait until a worker of that version runs. Alternatively, cancel and resume them, which queues them under the new version. The worker warns at start when such runs exist, and `runs_under_other_versions()` in `django_ai_sdk.workflows.durable` lists them.
- **JSON, not pickle.** DBOS stores inputs, step results and errors as JSON (`portable_json`), so loading them runs no code.
- **Cancel.** `cancel` marks the run `cancelled` and tells DBOS, which stops it before its next step. The step in flight finishes and its outcome is recorded, so a resume does not do it again. A run waiting for an approval is woken first, because DBOS's cancel alone leaves it waiting. `resume` continues it.
- **One worker at a time.** Without DBOS Conductor, a worker recovers every unfinished run of its executor (`"local"`) at launch, so two workers at once would both run them. `ai_sdk_worker` first takes a Postgres advisory lock on a database session of its own. A second worker waits as a standby until the first stops. This also makes a deploy safe that starts the new worker before it stops the old one. Every few seconds the worker checks that the session still holds the lock. When the lock is gone, for example after a database restart or a dropped connection, the worker exits with an error and its supervisor restarts it as a standby. The worker closes broken or expired database connections around each durable unit and task, as Django does around a request. A database restart therefore fails only the unit in flight. Concurrency comes from the queues, not from more workers.
- **No cap on the `workflows` queue.** A run waiting for an approval counts against its queue's `worker_concurrency` for as long as it waits. With a cap there, a few waiting approvals can block every other run. Limit model spend in the agents, not in the queue.
- **One queue system.** `DBOSTaskBackend` runs every other `django_tasks` task as a one-step DBOS workflow on the queue its `queue_name` names, so `db_worker` is not needed. Those tasks include oversight's reflection and notifications, and document processing. `AI_SDK_DBOS["QUEUES"]` declares the queues and their concurrency.

Web processes never launch DBOS. They queue runs, send decisions and manage schedules through a `DBOSClient` on the same tables.

## Approval steps

The built-in step type `approval` waits for a person:

```json
{"type": "approval", "name": "approve", "requires": ["check"],
 "config": {"prompt": "Book this invoice?", "timeout_seconds": 86400}}
```

The step is set to `waiting` with the prompt as its `detail`, and the run waits durably with `DBOS.recv` across restarts and deploys. A decision comes from `WorkflowService.decide` or `POST /workflows/runs/{id}/steps/{step}/decision/` (`{"approved": true, "note": "..."}`). It needs the `approve_steprun` permission. The engine refuses it on a run that acts as the approver or that the approver started, such as a trigger they fired by hand that runs as a bot. The first decision wins. A decision that arrives after the wait timed out raises `DecisionConflict`, and so does a decision for a step already decided or cancelled. `DecisionConflict` (`django_ai_sdk.workflows.services`) is a `UserError` with code `conflict`, and the SDK's views answer it with 409. The timeout and a decision race on the same conditional update, so the timeout never discards an accepted decision. When approved, the step completes with `{approved, by, note}`. When rejected, or without a decision within `timeout_seconds` (one week by default), the step fails and its `on_error` decides what follows.

:::caution[Read an approval with ctx.approval]
A step that acts on the approval reads it with `ctx.approval("approve")`, never with `ctx.step("approve")`. `ctx.approval` returns the `{approved, by, note}` decision only when the step of that name is an `ApprovalStep` that a person approved, and None otherwise. Anyone who may compose a definition could add another step named `approve` whose output is `{"approved": true}`.
:::

```python
class PayStep(Step):
    async def run(self, ctx):
        if (decision := ctx.approval("approve")) is None:
            return StepOutcome(output={"paid": False}, detail="Not approved.")
        await pay(..., idempotency_key=self.idempotency_key(ctx))
        return StepOutcome(output={"paid": True}, detail=f"Approved by {decision['by']}.")
```

## Triggers

A workflow declares triggers, which start it without a person or code asking:

```python
from django_ai_sdk.workflows import CronTrigger, EventTrigger, FieldDefinition, WorkflowDefinition

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

- **Cron.** The worker keeps one DBOS schedule per enabled cron trigger, named `<workflow>:<trigger>`, and syncs all of them on start. Saving a stored workflow syncs only that workflow's schedules. DBOS fires each occurrence once. The run's id derives from the occurrence, so a repeat opens nothing new. A cron expression has five fields only, so nothing fires more than once a minute. An operator can pause a trigger at runtime, and the pause survives a deploy. With `AI_SDK_WORKFLOWS["SCHEDULES_ENABLED"] = False`, occurrences start nothing and the schedules stay as they are, so turning it back on needs no clean-up. Firing a trigger by hand works either way, and records who fired it.
- **Event.** Every webhook delivery is reported as `<integration>.message`, with `text` and `conversation_id` in the payload. Report other events with `await start_for_event("invoice.received", payload, idempotency_key=...)` (`django_ai_sdk.workflows.triggers`). Every enabled trigger on that event starts a run whose id derives from the key, so a redelivered message starts nothing new.
- **Inputs.** The engine fills in `"{word}"` in a string. For cron it uses `name`, `workflow`, `scheduled_at` and `last_run_at`. For an event it uses the payload. A string that is only `"{word}"` takes the value with its type. Every input a trigger fills must be declared.
- **Who it acts as.** A workflow declared in code sets `run_as`, a USERNAME_FIELD value. A stored workflow's runs act as the person who last wrote its steps or switched it back on. The engine refuses a submitted definition that sets `run_as`, because otherwise authoring a workflow would let a user act as anyone. The triggers listing shows `run_as` only to those with the manage permission.

### Beta limitations

Workflows ship as a beta. The engine does not yet do the following:

- **No retries.** A failed step fails its run. With `on_error: continue`, the engine records the failure and moves on. Nothing retries the step automatically. `resume` queues the whole run again, and the run continues past the steps that completed.
- **No timeouts beyond the model's.** Only the client's timeout limits a model call. A step that does its own HTTP or subprocess work sets its own client's timeout.
- **Sequential steps only.** Steps run one at a time, in declared order. There is no fan-out and no join, even when two steps read nothing from each other.
- **Only model calls are overseen.** A step's turn is its model call, which is an `AgentStep`'s `agent.run()` or `self.turn()` in a step of your own. A step that calls no model, such as a count, a fetch or a save, opens no turn, and no policy judges it. A step that calls a model outside a turn is refused. With `UNGUARDED_MODEL_CALLS` set to `warn`, it is logged instead.
- **Oversight does not wait.** An oversight rule that blocks a step's turn fails the step. A `Queue` rule files a review item, and the step continues with the answer it got. To have a person decide before the run goes on, put an [approval step](#approval-steps) after it.
- **One worker.** Workers do not share the load. A second worker is a standby, see [Durable runs on DBOS](#durable-runs-on-dbos).
- **Step code changes replay.** A run recovered after a deploy replays its finished units by name. Changing which units a run goes through needs a new `AI_SDK_DBOS["VERSION"]`. Changing a step type's internals is fine, reordering the runner's units is not. The old version's unfinished runs are left to finish or to be resumed.
- **Run data is plaintext and stays.** `inputs`, `outputs` and each `StepRun.output` are stored as plain JSON and kept until you delete the rows. Deleting a user sets the run's `user` to null but keeps its inputs and outputs. Deleting a thread does not affect them. DBOS keeps its own copy of each run's arguments and every checkpointed step's result in its system tables, in schema `dbos`. Pass ids instead of content where you can, and prune both copies of old runs yourself. Delete the rows, and delete their DBOS workflows with `runtime.client().delete_workflows(ids)`. A run's ids are its own id and `"<run id>:<n>"` for each resume.
- **Payload ids are not checked.** The engine passes a document id or any other key in `inputs` to the steps as given. A step that loads a row by it must check that `ctx.user` may read that row.
- **A run executes the definition captured when it was queued.** Editing a stored workflow changes later runs, not one already queued. A run resumes only while its workflow is unchanged. Editing, deactivating or deleting a stored workflow, or changing or removing a registered one in code, stops its older runs from resuming. Resuming an old run therefore never skips a fix, such as an approval step added before a risky one. The worker looks up the step types and listeners the definition names, so they must be registered there too.
- **Declared schemas are structural.** They have no value constraints such as `min_length` or `ge`, and no validators. Check those in a step.

For multi-agent chat within one turn and without a durable record, use subagents, not workflows.

## System checks

| Id | When |
| --- | --- |
| `ai_sdk.workflows.E001` | A workflow passed to `register_workflow` was refused, for example for no steps, a name that is not a slug, or an unknown step type or listener. |
| `ai_sdk.workflows.E002` | An `AI_SDK_WORKFLOWS["STEPS"]` entry does not import, or is not a `Step` subclass. |
| `ai_sdk.workflows.E003` | An `AI_SDK_WORKFLOWS["LISTENERS"]` entry does not import, or is not a `RunListener` subclass. |
| `ai_sdk.workflows.E004` | `AI_SDK_WORKFLOWS["STEPS"]` has the key `agent` or `approval`, which the built-in step types reserve. |
| `ai_sdk.workflows.E005` | A 0.1 setting is still set: `AI_SDK_WORKFLOW_STEPS` (now `AI_SDK_WORKFLOWS["STEPS"]`) or `AI_SDK_WORKFLOW_ACTIONS` (now `AI_SDK_WORKFLOWS["LISTENERS"]`). |
| `ai_sdk.workflows.E012` | `AI_SDK_WORKFLOWS` is not a dict, or one of its keys has a value of the wrong type. |

At run time an entry that does not import is left out of its registry with a warning, so a definition that names it is refused. The checks report this before the first run.

## Step workflows

A queued run compiles its definition into `Step` objects and calls the `run_steps` walker. The [Step Workflows guide](../workflows/workflow-steps.md) covers writing step classes and listeners. This section is the reference for the runtime objects.

### Step

| Attribute | Default | Purpose |
| --- | --- | --- |
| `name` | `""` | Required, unique within a pipeline. The key its outcome is recorded under and the key later steps require. |
| `requires` | `()` | Steps declared **earlier** that this one reads. A name that is not one of them is refused before anything runs. |
| `on_error` | `OnError.STOP` | `STOP` stops the run. `CONTINUE` records the failure and runs the rest. |
| `listeners` | `()` | Listeners that fire for this step alone. |

`config` holds what a definition configured a registered step with. It is empty for a step written in code. The classmethod `validate_config(config)` raises `ValueError` unless the type accepts the config, and by default a step accepts none. `skip_if(ctx) -> str` returns a reason to skip, or `""` to run. A skipped step is recorded with its reason and produces nothing, so every step that reads it is skipped as well. `run(ctx) -> StepOutcome` does the work. `channel(ctx)` is the channel the step's turns are recorded on (`workflow:<key>:<step>`). `turn(ctx, agent_or_policy, input=... | messages=...)` opens a turn for a model the step calls itself, because a step's model calls must come from a turn. See [Model calls of your own](../oversight/oversight.md#model-calls-of-your-own-run_with_oversight).

`AgentStep` adds `agent` / `agent_id`, `schema`, `instructions` and `history`. `schema` is a pydantic model, so it can describe nested output. `history` lists the inputs to send as the conversation. A `str` input is one turn, and a list is a conversation. `AgentStep` implements `run` as one structured call. You can override the async `system_prompt(ctx)`, `user_message(ctx)` and `messages(ctx)`, and the overrides may run queries. A `None` result is reported as `failed`, not stored.

### WorkflowContext and StepOutcome

One context, for steps and listeners alike.

| `WorkflowContext` | Purpose |
| --- | --- |
| `inputs` / `input(name)` | What the caller supplied, coerced to the declared `input_fields`. |
| `steps` / `step(name)` | What has run so far, keyed by step name. `step()` returns the output of a **completed** step; `steps[name]` is the whole outcome, a failure's detail included. |
| `user` | The user the steps act as, forwarded to `AgentStep` calls. |
| `workflow` / `run_id` | The run's `workflow_key` and its id, for listeners that report on the run. |

| `StepOutcome` | Purpose |
| --- | --- |
| `output` | The step's value, filed under the step's own name. |
| `status` | `completed` / `failed` / `skipped`. Anything but `completed` blocks dependents. |
| `detail` | One line for a log or a timeline. |

### Running

`run_steps(steps, *, inputs=None, user=None, listeners=(), recorder=None, completed=None, workflow="", run_id="")` validates the pipeline, runs the steps **in the order they were declared**, and returns `{step_name: StepOutcome}`. A step runs once every step in its `requires` has completed. If one has not, the step is `skipped` and names the step it waited for. `completed=` replays a previous attempt's finished steps. `run_steps` tells `recorder=` about every step before the listeners, and the recorder's errors stop the run. Without a recorder nothing is recorded.

### Models

`StepRun` carries `RUNNING` and `SKIPPED` statuses and a `detail` column holding the one line a step's outcome reported. `sequence` is the step's position in the declared pipeline, and `("run", "sequence")` identifies the row. A skipped step never starts, so `started_at` cannot order a run.

A step that never ran is a `skipped` row naming what it was waiting on, not an absence.
