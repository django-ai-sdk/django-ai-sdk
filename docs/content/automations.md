---
title: Automations
type: docs
weight: 8
---

**Automations** run agent work on a schedule. An automation says *when* it is due, *as whom* it acts, and *which workflow* to run; the SDK claims each occurrence once and hands the run to your task worker.

Work starts three ways: someone asks for it, an event happens, or the clock says so. Chat covers the first. Automations cover the third.

## Install

Automations are an opt-in app on top of [workflows](/manual/workflows/), with APScheduler for reading cron expressions (`pip install django-ai-sdk[automations]`):

```python
INSTALLED_APPS = [
    ...,
    "django_ai_sdk",
    "django_ai_sdk.workflows",
    "django_ai_sdk.automations",
]
```

Run `manage.py migrate`. Without the app there are no automation tables, no `automations_tick` command, and nothing imports the package.

## Declaring one

Put an `automations.py` in any installed app. It is autodiscovered on startup.

```python
from django_ai_sdk.automations import Automation, register_automation


@register_automation
class MorningDigest(Automation):
    name     = "morning-digest"
    label    = "Morning digest"
    cron     = "0 7 * * mon-fri"
    timezone = "Europe/Amsterdam"
    workflow = "notion-digest"
    run_as   = "digest-bot@example.com"
    input    = "Summarise the open Notion tasks changed since {last_run_at}."
```

That runs on weekday mornings, as `digest-bot@example.com`.

## When

`cron` is a 5-field expression read in `timezone` (UTC by default). Set `timezone` when the expression means a wall-clock time: `"0 9 * * *"` in a zone with DST is a different instant for half the year. Write days of the week by name (`mon-fri`); numbers are refused, because the cron library underneath counts them differently from standard cron.

## As whom

A scheduled run has no request and no session, so it is told whose identity to act with: `run_as`, matched on the user model's `USERNAME_FIELD`, with that account's permissions and integrations. Leave it empty to run as the deployment itself, with no user; such a run cannot deliver into a chat, because nobody owns the thread, so its result is on the `WorkflowRun` or delivered by a workflow listener. An account that does not exist skips the run with an error in the log, rather than running with less than it should.

## What it runs

An automation names a [workflow](/manual/workflows/); the workflow owns the behaviour and can also be run from chat or the API. By default the automation supplies its rendered `input` as the workflow's `prompt` input:

```python
register(WorkflowDefinition(
    name="notion-digest",
    input_fields={"prompt": FieldDefinition(type="str")},
    steps=[StepDefinition(name="digest", agent_id=OperationsAgent().agent_id, history=["prompt"])],
))
```

A workflow that takes something else gets it from `get_inputs()`. Every key has to be one the workflow declares; a mismatch is logged and no run is created.

`input` supports `{user}`, `{name}` and `{last_run_at}`: when this automation's last completed run started, which makes a recurring run incremental.

## Running the scheduler

The tick only claims and enqueues; a [django-tasks](https://pypi.org/project/django-tasks/) worker executes the runs. Run both:

```
manage.py automations_tick --loop 60
manage.py db_worker
```

`--loop` keeps ticking; without it one tick runs, for a host that prefers cron. Ticks are safe on any number of hosts: each occurrence is claimed by one conditional `UPDATE`.

To run one now, outside its schedule and even while disabled:

```
manage.py automations_tick morning-digest
```

`AI_SDK_AUTOMATIONS = {"ENABLED": False}` stops everything, scheduled or by hand. `enabled = False` on the class keeps one out of the schedule.

## An admin page

The SDK ships no HTTP routes for automations; `AutomationService` is what a page sits on, with the `automations` permission domain checked on every call:

```python
from django_ai_sdk.automations import AutomationService

rows = await AutomationService.list_for_user(request.user)          # schedule, enabled, last run
await AutomationService.set_enabled("morning-digest", enabled=False, user=request.user)
runs = await AutomationService.run_now("morning-digest", user=request.user)  # "Run now"
history = await AutomationService.list_runs("morning-digest", user=request.user)
```

Any signed-in user may see the automations. Turning one on or off and running one now act as its `run_as` account, so they take the `django_ai_sdk_automations.change_automationstate` model permission (`MANAGE_AUTOMATION`); `is_staff` alone grants nothing. A run's output is its account's content: only a holder of that permission sees every run, others their own, in `list_runs`, `get_run` and the `last_run` that `list_for_user` attaches.

## Limitations

- **Delivery is at least once.** Write idempotent workflows. A dispatch that raises hands its occurrence back, so the next tick tries again; a scheduler process killed between claiming and queueing loses that occurrence.
- **Runs can overlap.** A run slower than the gap to the next occurrence is still going when the next one starts.
- **A run is one-shot.** No retry; for a recurring automation the next occurrence is the retry.
- **Missed windows are not replayed.** An outage produces one catch-up run, not a burst.
- **Skips leave no row**, only a log line.
