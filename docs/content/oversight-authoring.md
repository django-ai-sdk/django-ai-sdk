---
title: Runtime authoring
type: docs
weight: 11
---

**Define a whole policy without a deploy: its lenses, its rules and what it builds on.**

Runtime authoring is an opt-in app on top of [oversight](/oversight/) and its [review API](/manual/oversight/#review-api). An operator saves a policy definition; every process turns it into an ordinary `Policy` class and runs it exactly like one declared in code.

The line between the two is fixed: **a lens or rule declared in code changes only with a deploy**. What an operator authors (a policy, its rules, its lenses) can be changed and turned on or off at runtime.

What a definition holds:

- **What it extends**: code's policies, the packs (`security`, `safety`, `quality`) and other authored policies.
- **Lenses of its own**: a judge lens answered by aliases the deployment set up (`"judge"`, `"classifier?"`), a pattern lens of keywords, a check lens of validators the developers registered, or a built-in lens (`pii`, `secrets`, `refusal`, `length`, `language`).
- **Rules of its own**: a condition over its lenses (inherited ones included) and actions: `block`, `mask`, `restrict_tools`, `queue`, `notify`, `alert`.
- **Options**: `capture`, `stream_mode`, `fail_mode`, `retention`.

## Install

```python
INSTALLED_APPS = [
    ...,
    "django_ai_sdk",
    "django_ai_sdk.oversight",
    "django_ai_sdk.oversight.authoring",
]

AI_SDK_OVERSIGHT_AUTHORING = {
    # Every authored policy keeps the protective rules of these, directly or through a parent.
    "MUST_EXTEND": ["security"],
}
```

Run `manage.py migrate`. On start the app adds a policy loader (`django_ai_sdk.oversight.resolve.add_policy_loader`), so every process, task workers included, registers the authored policies. Remove the app and policies are code's alone: core never imports it. To remove it, run `manage.py migrate django_ai_sdk_oversight_authoring zero` first, then take it out of `INSTALLED_APPS`; turns of agents still naming an authored policy are refused from then on.

With the app installed, `get_oversight_router()` also serves its routes, next to the review routes. Leave them out with its `exclude=`. The [manual's permission table](/manual/oversight/#review-api) names the model permission behind each operation.

| Route | Body | Permission |
| --- | --- | --- |
| `GET /oversight/authoring/catalog/` | | `VIEW_OVERSIGHT_AUTHORING` |
| `GET /oversight/authoring/policies/` | | `VIEW_OVERSIGHT_AUTHORING` |
| `GET /oversight/authoring/policies/{name}/` | | `VIEW_OVERSIGHT_AUTHORING` |
| `PUT /oversight/authoring/policies/{name}/` | a definition | `AUTHOR_OVERSIGHT` |
| `PATCH /oversight/authoring/policies/{name}/` | `{"active": false}` | `AUTHOR_OVERSIGHT` |
| `PATCH /oversight/authoring/policies/{name}/rules/{rule}/` | `{"mode": "off" \| "report" \| "enforce"}` | `AUTHOR_OVERSIGHT` |
| `PATCH /oversight/authoring/policies/{name}/lenses/{lens}/` | `{"enabled": false}` | `AUTHOR_OVERSIGHT` |
| `DELETE /oversight/authoring/policies/{name}/` | | `AUTHOR_OVERSIGHT` |
| `POST /oversight/authoring/policies/{name}/try/` | `{"input", "output", "tool_calls", "definition"}` | `AUTHOR_OVERSIGHT` |

The catalog lists what a definition may use: lens kinds, answer types, actions, modes, condition operators, stages and subjects, the judge aliases by name (never their config), what a lens may configure inline with the arguments each takes, the registered validators, the policies to extend and `MUST_EXTEND`. A definition that does not check is 400 with every reason. A save replaces the policy's definition; the last save wins.

## A definition

```json
{
  "label": "Refunds",
  "extends": ["security"],
  "lenses": [
    {
      "kind": "judge",
      "name": "refund",
      "question": "Does the person ask for a refund?",
      "stages": ["gate"],
      "judges": [{"class": "Patterns", "keywords": ["refund", "money back"]}, "judge?"]
    }
  ],
  "rules": [
    {
      "name": "refunds_to_people",
      "when": {"lens": "refund", "op": "is_true"},
      "then": [{"kind": "queue", "queue": "refunds"}, {"kind": "block", "message": "A colleague will mail you about your refund."}],
      "mode": "report"
    }
  ],
  "options": {"retention": {"content": 30}}
}
```

A lens is the JSON form of a lens in code (`lens_from_dict`): `{"kind": "pattern" | "check" | "judge", "answer": "flag" | "choice" | "score" | "entities", ...}`, or `{"preset": "pii", "kinds": [...]}` for a built-in one. Conditions are the JSON form of `L()`: `{"lens", "op", "value"}`, combined with `{"all": [...]}`, `{"any": [...]}` and `{"not": ...}`. A rule starts in `report` unless it says otherwise, like a rule in code. An agent runs an authored policy when its code names it (`oversight = "refunds"`) or when it is `DEFAULT_POLICY`. Which policy an agent runs is code's choice; what an authored policy holds is the operator's.

From Python, `AuthoringService` does the same, with the same checks:

```python
from django_ai_sdk.oversight.authoring import AuthoringService
from django_ai_sdk.oversight.authoring.schemas import PolicyIn

await AuthoringService.save_policy("refunds", PolicyIn(**definition), user=request.user)
```

## What a definition may not do

An authored policy is checked as code's are (every lens has what answers it, every rule acts, a Mask points at an Entities lens, no cycle), and on top of that it may not:

- **take a code policy's name**: code wins, and the save is refused;
- **change an inherited lens or rule**: after the build, every lens and rule of every parent must be exactly as that parent builds it, so neither a lens or rule of the same name, nor a preset that takes its default name (`pii`, `secrets`, …), nor a later parent can replace one. A lens or rule declared in code stays code's, mode included;
- **configure a lens beyond a safe set**: inline only keywords (a pattern lens, or a `Patterns` judge, of whole words) and the built-in lenses `pii` (by kind), `secrets`, `refusal`, `length` and `language`, with plain arguments; a check lens names validators the developers registered (`register_validator`). No regex, no code, no import path: a pattern an author writes would run on every turn and could backtrack for seconds. A model judge, a classifier or a regex comes as an alias in `AI_SDK_OVERSIGHT["JUDGES"]`, which the deployment sets;
- **leave out what `MUST_EXTEND` names**;
- **read a lens it does not have**, or **grow beyond** 50 lenses, 100 rules or 256 KiB.

## Turning things on and off

Toggles change only what the policy itself declares:

- **a rule of its own** gets a mode: `off`, `report` or `enforce`;
- **a lens of its own** is turned off (`enabled: false`) or on;
- **the whole policy** is turned off (`active: false`): its own lenses and rules stop, and **the policies it extends keep running**. An agent that names it then runs under its parents alone.

A rule or lens it inherits cannot be toggled: 400. Each toggle is checked like a save, and applies like one.

Deleting the policy is different from turning it off: its name is gone, so turns that name it are refused (unless `FAIL_MODE` is `"open"`). A policy cannot be deleted while another authored policy extends it, an agent names it in code, or it is `DEFAULT_POLICY`.

## Try before saving

`POST .../try/` runs a draft (or, without `definition`, the saved one) over one exchange through gate, the tool calls, finish and reflect, and records no turn: the same `testing.dry_run` code uses. Its judges are real, so a judge it names is called and costs what a turn costs; that is why trying needs `AUTHOR_OVERSIGHT`.

## The record

An `AuthoredPolicy` row per policy holds its definition, whether it is `active`, and who changed it last and when (`updated_by_id`, `updated_at`). There is no history of earlier definitions and no reason per change.

## When a change applies

A save applies at once in the process that made it. Every other process reads the authored policies at most once per `AI_SDK_OVERSIGHT["RUNTIME_CACHE_SECONDS"]` (default `10`), so for a few seconds two workers may run different definitions. When the database cannot be read, a process keeps what it registered last.

A saved policy that no longer loads (a deploy removed a code policy it extends, or its parent dropped a lens it reads) is taken out of the registry, not skipped: turns that name it are refused (unless `FAIL_MODE` is `"open"`), the reason is logged, and `GET .../policies/{name}/` shows it in `error`.

## Settings

```python
AI_SDK_OVERSIGHT_AUTHORING = {
    "MUST_EXTEND": [],          # policies every authored policy must extend
}
```

What a lens may configure inline is fixed: keywords, and the `pii`, `secrets`, `refusal`, `length` and `language` presets. None takes code or a regex. A regex, a model or a classifier needs an alias in `AI_SDK_OVERSIGHT["JUDGES"]`.

## Checks

| Id | Level | When |
| --- | --- | --- |
| `ai_sdk.oversight.authoring.E012` | Error | `AI_SDK_OVERSIGHT_AUTHORING` has an unknown key, or a value of the wrong kind. |
| `ai_sdk.oversight.authoring.E001` | Error | `MUST_EXTEND` names a policy no code registers. |
| `ai_sdk.oversight.authoring.W001` | Warning (deploy) | `MUST_EXTEND` is empty: an operator can author a policy without the protective rules code declares, for every agent that names it. |

## Limitations

- Saving a policy others extend is not checked against them first; a child that no longer loads is refused, and shows its `error`.
- A definition cannot declare a new judge, validator or condition operator: those take code.
- The editor is yours to build: the routes and the catalog are what it needs.
