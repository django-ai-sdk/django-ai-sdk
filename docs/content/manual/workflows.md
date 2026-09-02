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
| `WorkflowSettings` | A persisted, named workflow: `name`, `definition` (JSON), `active`, `created_by`. |
| `WorkflowRun` | One execution: status `pending` / `running` / `completed` / `failed`, `workflow_definition` snapshot, `input_messages`, `outputs`, `error`, `task_id`, `user`. |
| `WorkflowRunStep` | Per-step progress: `sequence`, `step_name`, `output_key`, `output`, status `pending` / `completed` / `failed`, `error`, timestamps. |

A `WorkflowSettings.to_workflow_definition()` round-trips the stored JSON.

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

## Execution Model

`WorkflowService.run()` creates a `WorkflowRun` in `pending`, then `WorkflowExecutor.enqueue()` schedules the `execute_workflow` background task (`django_tasks`), which runs `WorkflowExecutor.run()`. Whether that happens in a worker or inline in the calling request depends on the configured `TASKS` backend - see [Background Tasks](/manual/settings/#background-tasks).

`WorkflowExecutor.run()`:

1. Marks the run `running` (or returns early for an already-completed run; the operation is idempotent).
2. **Replays** completed steps from a previous attempt into outputs, so a resumed run continues where it left off.
3. Runs each remaining step: `AgentService.get(agent_id)` → `agent.run(...)` with optional context injection and structured output.
4. Runs the registered actions.
5. Marks the run `completed` with the full outputs; a step exception marks the run (and its step) `failed` and propagates.

## Custom Actions

```python
from django_ai_sdk.workflows.actions import BaseAction

class ConsoleLogAction:
    description = "Log the payload"

    async def execute(self, payload) -> None:
        print(payload)
```

```python
# settings.py
AI_SDK_WORKFLOW_ACTIONS = {
    "console_log": "apps.agents.actions.ConsoleLogAction",
}
```

`WorkflowService.list_actions()` returns `[{key, description}]` from the registry.
