---
title: Automations
type: docs
weight: 122
---

The [Automations guide](/automations/) covers declaring and running them. This page documents the class contract, the claim mechanics, the models, the checks, and the two extension points.

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
| `allow_overlap` | `False` | When `False`, a tick is skipped while a previous run holds the lease. A fan-out holds it until every run in the dispatch has finished, not until the first one has. |
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

Every layer governs *scheduled* dispatch. Naming one automation and asking for it explicitly — `run_automations --force --automation NAME`, or `POST /automations/{name}/run` — bypasses all of them, including the kill switch, because that is a person rather than the clock. A blanket `--force` does not.

## The tick

`manage.py run_automations` is one pass. For each registered automation it bootstraps the state row if absent, checks enabled-ness, checks due-ness, then claims.

The claim is one conditional `UPDATE` that advances the cursor and takes the lease in the same statement:

```sql
UPDATE ... SET next_run_at = ?, last_dispatched_at = ?, locked_until = ?
WHERE id = ? AND next_run_at <= ? AND (locked_until IS NULL OR locked_until < ?)
```

Because the predicate and the write are one statement, two ticks racing on the same row cannot both see "due" — the second one's `UPDATE` matches zero rows. No transaction, no `SELECT … FOR UPDATE`, and no dependency on `SKIP LOCKED`, which SQLite does not have.

Consequences worth internalising:

- **Concurrency is free.** Run the tick on every host. Correctness does not depend on there being exactly one scheduler.
- **Lease expiry is the crash-recovery path.** A worker that dies mid-run blocks nothing past `AI_SDK_AUTOMATION_LEASE`. There is no sweeper process.
- **A lease is only dropped by the dispatch that took it.** Releasing is conditional on the lease value, so a run finishing long after its own dispatch cannot free one a later tick is holding.
- **A new automation is scheduled forward, never backward.** Deploying one does not fire it immediately because its cron matched some moment in the past.
- **Runs are enqueued after their rows are committed**, so a worker is never handed an id whose row has not landed.

### Falling behind

After an outage `next_run_at` is in the past. The tick runs the automation **once**, stamped with the occurrence it missed rather than with `now`, and the claim advances the cursor to the next future occurrence.

Missed windows are not replayed. Whether replaying an outage's worth of digests is correct or spam depends entirely on the payload, and the safe default is the one that cannot flood anyone. If a run must cover everything since the last success, read `{last_run_at}` in the `input` — that is what makes a recurring run incremental.

## Execution

The tick never runs a workflow; it enqueues `execute_automation(run_id)`, and a django-tasks worker executes it. In the worker:

1. Load the run, resolve the automation. A run whose declaration was deleted in a deploy becomes `SKIPPED`, not an error.
2. Check `requires`. A degraded integration makes the run `SKIPPED`, not `FAILED` — a dead upstream is an outage, not a broken automation, and alerting every fifteen minutes about a self-healing condition trains everyone to ignore alerts.
3. Rehydrate the principal from `run.user_id`.
4. Resolve `workflow` through the registry. A name that resolves to nothing is `SKIPPED` with a reason, for the same reason as a degraded integration: the automation is not broken, its dependency is absent.
5. Run it inline under `asyncio.wait_for(..., timeout)`. django-tasks has no native timeout.
6. Write the terminal status. The lease is released once no *other* run sharing this one's `dispatch_id` is still `PENDING` or `RUNNING` — one claim covers the whole fan-out, so the first finisher freeing it would let the next tick dispatch over the top of its own siblings. A task cancelled before it could write a terminal status drops the lease too, leaving its row `RUNNING` as the record that it stopped.

Failures re-raise so django-tasks marks its own result `FAILED` too — two sources of truth that disagree are worse than one duplicated.

## Models

**`AutomationState`** — one row per automation, the scheduler's cursor and lease. Created lazily by the first tick that sees the automation, never at app load. Key fields: `enabled` (nullable, `None` = defer), `next_run_at`, `last_dispatched_at`, `last_success_at`, `locked_until`, `schedule_repr`.

`last_success_at` only advances on success, which is what makes `{last_run_at}` correct: a failed run must not cause the next one to skip the window it never processed. Across a fan-out, any one success advances it — the field is per-automation while the audience is per-user, so one principal failing cannot be allowed to make every other principal reprocess a window it already covered.

**`AutomationRun`** — one row per execution per principal. `status` is `pending` / `running` / `succeeded` / `failed` / `skipped`; `trigger` is `schedule` or `manual` — the clock, or a person. `output` holds the workflow's results, and is the only place an app-level run's result is readable. `dispatch_id` groups everything one tick produced. `name` is denormalised from state, so deleting the code does not delete the audit trail. `scheduled_for` is the window the run is for, not when it started, so two runs of an hourly job stay distinguishable even if a backed-up worker ran them together.

`workflow_run` links to the `WorkflowRun` holding per-step detail.

### Retention

Run history grows without bound — a daily digest for 500 users is ~182k rows a year. Nothing deletes on a timer: run history is your audit trail, and the retention horizon is your policy rather than the SDK's silent default. `AutomationRun` is a plain model, so prune it on whatever schedule suits you, filtering on `finished_at` so a run still in flight is never deleted out from under its worker.

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

There is no check for the scheduler itself never having run. The SDK cannot notice its own absence — if the cron entry was never added, no code of ours executes — so wire a dead-man's switch to the most recent `AutomationRun` instead.

## Extension points

Two protocols, neither of which requires a registry, a setting, or subclassing.

An audience resolving to `[None]` (as `Audience.APP` does) means one run with no principal. Anything downstream that needs a user — the `thread_message` action, a per-user integration credential, a non-public runtime agent — has none, so app-level automations pair with delivery that does not depend on one.

**`AudienceResolver`** — `resolve(automation) -> list` and `describe() -> str`. The shipped ones are `Audience.APP` and `Audience.SUBSCRIBED`; implement your own for a tenant, a subscription tier, an on-call rota.

**`BaseAction`** — `async def execute(self, payload, context)`, where `context` is an `ActionContext` carrying the run's user, agent and origin. Registered by key in `AI_SDK_WORKFLOW_ACTIONS`. Declaring a built-in key shadows it.
