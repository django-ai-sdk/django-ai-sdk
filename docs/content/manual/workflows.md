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
    name="summarize-and-alert",
    steps=[
        WorkflowStep(
            name="summarize",
            agent_id=SummarizerAgent().agent_id,
            output_key="summary",
        ),
        WorkflowStep(
            name="classify",
            agent_id=ClassifierAgent().agent_id,
            input_key="summary",                 # inject step 1's output
            output_key="priority",
            output_fields={                     # structured output
                "priority": {"type": "str", "description": "high|low"},
                "score": {"type": "float"},
            },
        ),
    ],
    actions=[WorkflowAction(type="console_log", input_key="summary")],
)
```

### Step semantics

Each `WorkflowStep` runs its `agent_id` via `agent.run()` (non-streaming). Its result is stored under `output_key` in the run's outputs.

| Field | Purpose |
| --- | --- |
| `name` | Display name for the step record. |
| `agent_id` | Agent to run (resolved through `AgentService`). |
| `output_key` | Where the result is stored in the run's outputs. |
| `input_key` | Optional: a prior step's `output_key` injected as a `[Workflow context]` user message (missing keys are skipped with a warning). |
| `system_prompt_override` | Optional system prompt for this step. |
| `output_fields` | When set, the agent runs with structured output: a dynamic Pydantic model is built from the `{name: {type, description}}` map (`type` ∈ `str` / `int` / `float` / `bool`). |

### Actions

`WorkflowAction(type, input_key)` runs after all steps complete. The `type` is looked up in the action registry (`AI_SDK_WORKFLOW_ACTIONS`); the action receives the `input_key` output, or the full outputs dict when `input_key` is unset. Unknown types and missing inputs are skipped with a warning.

## Models

| Model | Purpose |
| --- | --- |
| `WorkflowSettings` | A persisted, named workflow: `name`, `slug` (registry key), `definition` (JSON), `active`, `created_by`. |
| `WorkflowRun` | One execution: status `pending` / `running` / `completed` / `failed`, `workflow_definition` snapshot, `input_messages`, `outputs`, `error`, `task_id`, `user`. |
| `WorkflowRunStep` | Per-step progress: `sequence`, `step_name`, `output_key`, `output`, status `pending` / `completed` / `failed`, `error`, timestamps. |

A `WorkflowSettings.to_workflow_definition()` round-trips the stored JSON.

## Declaring a workflow

Call `register()` in any installed app's `workflows.py`. It is autodiscovered on startup, the same way `agents.py` is — no settings entry, no `ready()` hook. Autodiscovery runs after the agent registry is built, so a declaration can name its agents as `MyAgent().agent_id`.

```python
# myapp/workflows.py
from django_ai_sdk.workflows import WorkflowAction, WorkflowDefinition, WorkflowStep, register

register(WorkflowDefinition(
    name="weekly-triage",
    steps=[
        WorkflowStep(name="collect", agent_id=EngineeringAgent().agent_id, output_key="issues"),
        WorkflowStep(name="rank", agent_id=OperationsAgent().agent_id,
                     input_key="issues", output_key="triage"),
    ],
    actions=[WorkflowAction(type="thread_message", input_key="triage")],
))
```

`name` is the registry key, and must be a slug of at most 100 characters — the same form as `WorkflowSettings.slug`, so a declaration and a row that share a name collide visibly instead of both being reachable. A registered workflow can be resolved by that name from anywhere (`aget_workflow`), rather than existing only inside whichever caller constructed it.

A workflow is data, not behaviour, so it is registered by a call rather than by subclassing.

### Validation

`register()` validates, because the executor skips a step whose input is missing and only logs it — which turns a one-character typo into a silent half-run.

- there is at least one step
- every step has an `agent_id` and an `output_key`
- no two steps share an `output_key`
- every `input_key`, on a step or an action, names an `output_key` produced by an **earlier** step

A definition that fails is left out of the registry and reported by a system check as `ai_sdk.workflows.E001`:

```console
$ ./manage.py check
ERRORS:
?: (ai_sdk.workflows.E001) Workflow 'weekly-triage' step 1 reads 'isues', which no
earlier step produces. Available at that point: ['issues'].
```

An error, so `manage.py check` and `runserver` fail on it and it is caught before release; a warning is logged at startup too. It is deliberately not an exception: one app's typo must not stop the site from booting.

`agent_id` is a registry id (`uuid5(AGENT_NAMESPACE, class_path)`) or an `AgentSettings` primary key. Rather than pasting a UUID into a declaration, take it from the class — `MyAgent().agent_id` — so a typo is an `ImportError` where the workflow is declared rather than a failure in a worker. Validation cannot check the id itself: the agent may live only in the database.

### Database workflows

Active `WorkflowSettings` rows are merged into the registry under their `slug`, exactly as the integrations registry merges `MCPServerConfig` rows. A code declaration always wins a collision and warns once — a row is for adding a workflow where there is no code, never for overriding one. A row whose definition fails the same validation is skipped and logged, so one bad row never takes the others with it.

```python
from django_ai_sdk.workflows import aget_workflow, aget_workflows, get_declared_workflows

await aget_workflow("weekly-triage")   # code or database
await aget_workflows()                 # merged
get_declared_workflows()               # code only, synchronous
```

The merged lookups are async because they read the database. `get_declared_workflows()` is the synchronous view, for the system check and management commands that must not need one.

`slug` is derived from `name` on first save and suffixed if taken, so two workflows may share a display name. Rows are read fresh on every dispatch, so an admin edit takes effect immediately. A database that cannot be read raises rather than quietly falling back to the code-declared workflows, so an outage fails a run the queue will retry instead of recording it as a workflow that does not exist.

## Running

```python
from django_ai_sdk.workflows import WorkflowService

# Ad-hoc run (inline definition, no persisted record)
run = await WorkflowService.run(workflow, messages, user=request.user)

# Persisted workflows
record = await WorkflowService.create("My workflow", workflow, user=request.user)
await WorkflowService.update(workflow_id, name=..., workflow=..., active=...)
await WorkflowService.delete(workflow_id)
await WorkflowService.get(workflow_id)
await WorkflowService.list_workflows(active_only=True)

# Run a persisted workflow (optionally resuming an existing run)
run = await WorkflowService.run_by_id(workflow_id, messages, user=request.user, run_id=None)

# Run history
runs = await WorkflowService.list_runs(workflow_id)
run = await WorkflowService.get_run(run_id)      # prefetches steps
```

The `user` passed here is stored on the `WorkflowRun` and reloaded in the worker, so every step runs as that user: their permissions are checked and their per-user integration credentials are available. A run created without a `user` is unowned and runs as anonymous, which a non-public agent rejects.

## Execution Model

`WorkflowService.run()` creates a `WorkflowRun` in `pending`, then `WorkflowExecutor.enqueue()` schedules the `execute_workflow` background task (`django_tasks`), which runs `WorkflowExecutor.run()`. Whether that happens in a worker or inline in the calling request depends on the configured `TASKS` backend - see [Background Tasks](/manual/settings/#background-tasks).

`WorkflowExecutor.run()`:

1. Marks the run `running` (or returns early for an already-completed run; the operation is idempotent).
2. **Replays** completed steps from a previous attempt into outputs, so a resumed run continues where it left off.
3. Runs each remaining step: `AgentService.get(agent_id)` → `agent.run(...)` with optional context injection and structured output.
4. Runs the registered actions.
5. Marks the run `completed` with the full outputs; a step exception marks the run (and its step) `failed` and propagates.

## Actions

An action decides where a finished run's output goes. `thread_message` ships built in: it
opens a new chat thread for `context.user`, posts the payload into it, and gives it a
generated title. A run started from a queue has nobody watching a response stream, so
without it a scheduled workflow produces output nobody sees.

```python
actions=[WorkflowAction(type="thread_message", input_key="triage")]
```

It needs someone to deliver to. A run with no user logs and returns rather than failing
the run, on the grounds that the workflow itself succeeded.

### Custom Actions

```python
from django_ai_sdk.workflows.actions import ActionContext

class ConsoleLogAction:
    description = "Log the payload"

    async def execute(self, payload, context: ActionContext) -> None:
        print(context.source, payload)
```

`context` carries the run's `user` (`None` when nobody owns the run), the `agent_id` that
produced the payload, and a human-readable `source`. Ignore it if you only need the
payload — it is always passed.

```python
# settings.py
AI_SDK_WORKFLOW_ACTIONS = {
    "console_log": "apps.agents.actions.ConsoleLogAction",
}
```

Declaring a built-in key shadows the built-in.

`WorkflowService.list_actions()` returns `[{key, description}]` from the registry.
