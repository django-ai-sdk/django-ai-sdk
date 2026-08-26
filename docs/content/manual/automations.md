---
title: Automations
type: docs
weight: 122
---

The [Automations guide](/automations/) covers declaring them. This page documents the class contract, the models, the checks, and the two extension points.

## `Automation`

| Attribute | Default | Meaning |
| --- | --- | --- |
| `name` | `""` | Registry key. Required, unique, and stable — state, settings overrides and run history are all keyed on it. Renaming orphans the old state row. |
| `label` | `""` | Display name; falls back to `name`. |
| `description` | `""` | Shown in the API and admin. |
| `cron` | `""` | 5-field expression. Required. |
| `timezone` | `"UTC"` | IANA zone the `cron` expression is read in. |
| `workflow` | `""` | Name of a registered [workflow](/manual/workflows/). Required. |
| `input` | `"Run the {name} automation."` | The user turn this occurrence starts from. Supports `{user}`, `{last_run_at}`, `{name}`. |
| `audience` | `Audience.APP` | Which principal(s) it runs as. |
| `requires` | `[]` | Integration names that must be `ACTIVE`. |
| `enabled` | `True` | Shipped default; a database row or settings entry overrides it. |
| `allow_overlap` | `False` | When `False`, a tick is skipped while a previous run holds the lease. |
| `timeout` | `None` | Seconds; falls back to `AI_SDK_AUTOMATION_TIMEOUT` (900). |

`cron` and `workflow` must both be non-empty. Both are enforced at registration: a half-specified declaration is kept out of the registry and reported as `W006`, rather than silently never firing. Registration itself does not raise, so one app's typo cannot stop the site from booting.

Whether the named workflow *exists* is a check rather than a registration error — a workflow may live in the database, and registration must not touch one.

Methods worth knowing: `get_schedule()` resolves settings over the class attribute; `has_perms(user, operation)` checks one operation; and `render_input(user=..., last_run_at=...)` builds the input turn.

`render_input` is the identity hook. The default substitutes `{user}` (the username), `{last_run_at}` and `{name}`, but it receives the resolved principal, so overriding it is how a run says which identity it is for — an email, a tenant, an account id. That keeps the workflow generic and the occasion specific, and it is why a shared workflow can serve every user without one per person.

## Resolution order

Both the schedule and enabled-ness resolve **database row → settings → class attribute**.

```python
AI_SDK_AUTOMATIONS = {
    "morning-digest": {
        "CRON": "0 8 * * *",
        "TIMEZONE": "Europe/Amsterdam",
        "ENABLED": False,
    },
}
```

Keys are upper-cased and the resolution never raises — a malformed entry is logged and ignored. The resolved value carries which layer decided: `"kill-switch"`, `"db"`, `"settings"` or `"code"`.

Every layer governs *scheduled* dispatch. Asking for one automation explicitly is a person rather than the clock, and bypasses all of them including the kill switch.

## Models

**`AutomationState`** — one row per automation, the scheduler's cursor and lease. Created lazily by the first tick that sees the automation, never at app load. Key fields: `enabled` (nullable, `None` = defer), `next_run_at`, `last_dispatched_at`, `last_success_at`, `locked_until`, `schedule_repr`.

`last_success_at` only advances on success, which is what makes `{last_run_at}` correct: a failed run must not cause the next one to skip the window it never processed.

**`AutomationRun`** — one row per execution per principal. `status` is `pending` / `running` / `succeeded` / `failed` / `skipped`; `trigger` is `schedule` or `manual` — the clock, or a person. `output` holds the workflow's results, and is the only place an app-level run's result is readable. `dispatch_id` groups everything one tick produced. `name` is denormalised from state, so deleting the code does not delete the audit trail. `scheduled_for` is the window the run is for, not when it started, so two runs of an hourly job stay distinguishable even if a backed-up worker ran them together.

`workflow_run` links to the `WorkflowRun` holding per-step detail.

## System checks

All warnings, never errors: boot must not fail because a background job is misconfigured.

| ID | Fires when |
| --- | --- |
| `W001` | The schedule cannot be resolved at all — `cron` is unset or invalid. |
| `W002` | `timezone` is not a known IANA zone. The expression is read as UTC. |
| `W003` | `workflow` names no workflow any installed app declares. |
| `W004` | `requires` names an integration no installed app registers. |
| `W005` | `AI_SDK_AUTOMATIONS` configures a name nothing declares. |
| `W006` | The declaration was refused at registration and is not in the registry. |

`W003` and `W004` only consult code-declared workflows and integrations: the checks are synchronous and must not need a database, so a workflow living only in `WorkflowSettings` cannot be confirmed. `W003`'s message says so rather than asserting the name is wrong.

## Extension points

Two protocols, neither of which requires a registry, a setting, or subclassing.

An audience resolving to `[None]` (as `Audience.APP` does) means one run with no principal. Anything downstream that needs a user — the `thread_message` action, a per-user integration credential, a non-public runtime agent — has none, so app-level automations pair with delivery that does not depend on one.

**`AudienceResolver`** — `resolve(automation) -> list` and `describe() -> str`. The shipped ones are `Audience.APP` and `Audience.SUBSCRIBED`; implement your own for a tenant, a subscription tier, an on-call rota.

**`BaseAction`** — `async def execute(self, payload, context)`, where `context` is an `ActionContext` carrying the run's user, agent and origin. Registered by key in `AI_SDK_WORKFLOW_ACTIONS`. Declaring a built-in key shadows it.
