---
title: Automations
type: docs
weight: 122
---

The [Automations guide](/automations/) covers declaring and running them. This page documents the class, the tick and the model.

## `Automation`

| Attribute | Default | Meaning |
| --- | --- | --- |
| `name` | `""` | Registry key. Required and stable: state and run history are keyed on it. |
| `label`, `description` | `""` | Display only. |
| `cron` | `""` | 5-field expression, days of the week by name. Required. |
| `timezone` | `"UTC"` | IANA zone `cron` is read in. An unknown zone refuses the declaration. |
| `workflow` | `""` | Name of a registered [workflow](/manual/workflows/). Required. |
| `run_as` | `""` | The account a run acts as, by `USERNAME_FIELD`. Empty runs with no user. |
| `input` | `"Run the {name} automation."` | The turn the default `get_inputs()` supplies. |
| `enabled` | `True` | `False` keeps it out of the schedule. |

| Method | Default | Override to |
| --- | --- | --- |
| `get_inputs(*, user, last_run_at) -> dict` | `{"prompt": render_input(...)}` | Supply the workflow's inputs in the shape it declares. |
| `render_input(*, user, last_run_at) -> str` | fills `{user}`, `{name}`, `{last_run_at}` | Name the run's subject differently. |

A declaration without `name`, `cron` or `workflow`, or whose `cron` or `timezone` does not parse, is kept out of the registry and reported as `W006`, so one typo cannot stop the site from booting. A `workflow` no installed app declares is `W003`: it may live in the database, which registration must not touch.

## The tick

`automations_tick` checks each enabled automation, unless `AI_SDK_AUTOMATIONS["ENABLED"]` is `False`. It creates the state row on first sight, scheduled forward so a deploy never fires for a past match, and then claims a due one:

```sql
UPDATE ... SET next_run_at = ?, last_dispatched_at = ?
WHERE id = ? AND next_run_at <= ?
```

Predicate and write are one statement, so two ticks racing on the same row cannot both win. A won claim resolves the workflow and `run_as`, calls `get_inputs`, refuses keys the workflow does not declare, and opens a `WorkflowRun` (linked to its `WorkflowSettings` row when the workflow is stored) with an `AutomationRun` row naming the automation, and queues it. After an outage the automation runs once and the cursor moves to the next future occurrence.

`automations_tick NAME` and `runner.dispatch_now(name)` dispatch immediately, whether or not the automation is enabled. They leave `next_run_at` alone and respect the kill switch.

## Model

**`AutomationState`**: one row per automation, `next_run_at` and `last_dispatched_at`. Created by the first tick.

**`AutomationRun`**: which automation opened a run, one row per `WorkflowRun` (deleted with it). Status, outputs and errors are the workflow run's own. History grows without bound; prune `WorkflowRun` as suits you.
