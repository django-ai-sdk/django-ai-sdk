---
title: Automations
type: docs
weight: 8
---

**Automations** run agent work on a schedule. An automation says *when* it is due, *as whom* it acts, and *which workflow* to run.

Work starts three ways: someone asks for it, an event happens, or the clock says so. Chat covers the first. Automations cover the third.

## Declaring one

Put an `automations.py` in any installed app. It is autodiscovered on startup, the same way `agents.py` is — no settings entry, no `ready()` hook.

```python
from django_ai_sdk.automations import Audience, Automation, register


@register
class MorningDigest(Automation):
    name       = "morning-digest"
    label      = "Morning digest"
    cron       = "0 7 * * 1-5"
    timezone   = "Europe/Amsterdam"
    requires   = ["notion"]
    audience   = Audience.SUBSCRIBED
    workflow   = "notion-digest"
    input      = "Summarise my open Notion tasks changed since {last_run_at}."
```

That runs on weekday mornings, once for each user who connected Notion, as that user, and posts the result into their chat.

Integration apps are Django apps, so an integration can ship its own `automations.py` and it is picked up only when that app is installed.

## When

Set `cron`, a 5-field expression:

```python
cron = "0 9 * * 1-5"
```

It's read in `timezone` (UTC by default). This matters: `"0 9 * * *"` means nine in the morning, and in a zone that observes DST that is a different absolute instant for half the year. Set `timezone` when the expression encodes a wall-clock intention; leave it alone when the automation just needs to happen periodically.

Everything the SDK stores is UTC. Only the reading of the expression is localised.

## As whom

A scheduled run has no request and no session, so it has to be told whose identity to act with. There are exactly two shapes:

| Audience | Runs |
| --- | --- |
| `Audience.APP` (default) | once, with no user — housekeeping acting as the deployment itself |
| `Audience.SUBSCRIBED` | once per user who has personally turned this automation on |

**An `Audience.APP` run has no user, so it cannot use the `thread_message` action** — there is nobody to own the thread, and the action logs and returns rather than failing the run. App-level work either delivers through a tool the agent holds, through an action you register, or not at all: its result is recorded on the `AutomationRun` and readable from the admin and the API. If you want a message in someone's chat, you want `Audience.SUBSCRIBED`.

A private runtime agent (an `AgentSettings` row with `is_public = False`) also refuses an app-level run, since there is no user to match against its allowed users and groups.

An automation's author does not choose who it runs for beyond that — each user does, for themselves, via `PATCH /automations/{name}/subscription`. There is no developer-configured "run for this queryset" or "run for users connected to X"; if you need one of those, implement the `AudienceResolver` protocol (`async def resolve(self, automation) -> list`, `def describe(self) -> str`) and set it as `Automation.audience` yourself, rather than reaching for a built-in, since none is shipped.

**Known limitation:** `Audience.SUBSCRIBED` does not check that a subscriber actually holds a credential for anything the automation `requires`. For an integration authenticating with a shared deployment secret that never mattered — every user shares the one credential. For an OAuth-style, per-user integration, a user can subscribe without ever connecting it, and that run fails rather than being cleanly skipped.

The claim happens once and the fan-out happens after it, so a run for 400 subscribers is 400 `AutomationRun` rows sharing one `dispatch_id`.

## What it runs

An automation names a [workflow](/manual/workflows/). It never contains one.

A workflow owns the *behaviour* — the steps, their system prompts, the actions that deliver the result — and is declared in an app's `workflows.py` where every other caller can reach it. The automation owns the *occasion*: when it is due, whose identity it acts with, and the input that occasion starts from.

```python
# myapp/workflows.py — what the work is
register(WorkflowDefinition(
    name="notion-digest",
    steps=[WorkflowStep(agent_id=OperationsAgent().agent_id, output_key="digest",
                        system_prompt_override="Summarise open work. Three sections.")],
    actions=[WorkflowAction(type="thread_message", input_key="digest")],
))

# myapp/automations.py — when, as whom, and what is being asked now
class MorningDigest(Automation):
    cron     = "0 7 * * 1-5"
    audience = Audience.SUBSCRIBED
    workflow = "notion-digest"
    input    = "Summarise everything changed since {last_run_at}."
```

Keeping those apart is what lets the same workflow run from chat, from the API, by hand, and on a schedule, rather than existing only inside the thing that happens to schedule it.

### Naming who a run is for

`input` supports `{user}`, `{last_run_at}` and `{name}`, where `{user}` is the username. When a run needs a different identity — an email an issue tracker can match an assignee on, a tenant id, an account number — override `render_input`. It receives the resolved principal, so it is the one place that knows which identity a run belongs to:

```python
class Standup(Automation):
    audience = Audience.SUBSCRIBED
    workflow = "my-open-issues"

    def render_input(self, *, user=None, last_run_at=None):
        return f"Open issues assigned to {user.email}, changed since {last_run_at}."
```

This is how a workflow stays generic while each run is about one person. The workflow says *find the open issues for whoever the request names*; the automation says *who*.

For work scoped by a **credential** rather than by a name, use a tool factory instead — `APIIntegration.tools` accepts a callable that is passed the user and builds a tool around them, so the scope is enforced in code rather than asked for in a prompt. See [Integrations](/integrations/).

`{last_run_at}` is what makes a recurring run incremental rather than re-doing the same summary every morning; it only advances on success, so a failed run does not cause the next one to skip the window it never processed.

Scheduling arbitrary Python is out of scope by design — write a management command and give it a cron line. What Django cannot already do is run *agent* work on a schedule, for a resolved principal, gated on integration health, with the result delivered somewhere a person sees it.

## Delivering output

A scheduled workflow has nobody watching a response stream, so its result has to be put somewhere deliberately. Three ways, in the order you should reach for them:

**Give the agent a tool.** A step runs an agent with its integrations, so an agent holding a Slack or Teams MCP posts the message itself. Nothing to configure on the automation, and it composes with everything else the agent can do.

**A `thread_message` action on the workflow.** The built-in: the result arrives as a new conversation in your app for the run's user. This is what makes "message every user their digest" a declaration rather than a plumbing exercise. It needs a user, so pair it with `Audience.SUBSCRIBED`; an `Audience.APP` run has nowhere to deliver and logs instead.

**Register your own action** when delivery must be deterministic rather than left to a model's choice to call a tool:

```python
AI_SDK_WORKFLOW_ACTIONS = {"slack_post": "apps.alerts.actions.SlackPostAction"}
```

An action is a class with an `async def execute(self, payload, context)`. The context carries the run's user, the agent that produced the payload, and where it came from.

## Configuration

```python
AI_SDK_AUTOMATIONS = {
    "morning-digest": {"CRON": "0 8 * * *", "TIMEZONE": "Europe/Amsterdam"},
    "nightly-reindex": {"ENABLED": False},
}
AI_SDK_AUTOMATIONS_ENABLED = True    # global kill switch
```

Enabled-ness resolves **database row → settings → class attribute**, and the resolved value carries which layer decided so "why is this off?" has an answer. The database row is what an admin toggle writes; settings is what a deployer pins; the class is the shipped default.
