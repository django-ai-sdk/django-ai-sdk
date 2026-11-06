---
title: Runtime authoring
type: docs
weight: 11
---

**Define a whole policy without a deploy: its lenses, its rules and what it builds on.**

Runtime authoring is an opt-in app on top of [oversight](/oversight/) and its [review API](/manual/oversight/#review-api). An operator saves a policy definition; every process turns it into an ordinary `Policy` class and runs it exactly like one declared in code. Where [runtime switches](/oversight-switches/) change a rule's mode or an agent's policy, authoring adds policies.

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
| `PUT /oversight/authoring/policies/{name}/` | a definition and a `reason` | `AUTHOR_OVERSIGHT` |
| `POST /oversight/authoring/policies/{name}/try/` | `{"input", "output", "tool_calls", "definition"}` | `AUTHOR_OVERSIGHT` |
| `POST /oversight/authoring/policies/{name}/withdraw/` | `{"reason"}` | `AUTHOR_OVERSIGHT` |

The catalog lists what a definition may use: lens kinds, answer types, actions, modes, condition operators, stages and subjects, the judge aliases by name (never their config), what a lens may configure inline with the arguments each takes, the registered validators, the policies to extend and `MUST_EXTEND`. A definition that does not check is 400 with every reason; two operators saving one policy at the same moment may get 409.

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
  "options": {"retention": {"content": 30}},
  "reason": "Refund requests go to the refunds team"
}
```

A lens is the JSON form of a lens in code (`lens_from_dict`): `{"kind": "pattern" | "check" | "judge", "answer": "flag" | "choice" | "score" | "entities", ...}`, or `{"preset": "pii", "kinds": [...]}` for a built-in one. Conditions are the JSON form of `L()`: `{"lens", "op", "value"}`, combined with `{"all": [...]}`, `{"any": [...]}` and `{"not": ...}`. A rule starts in `report` unless it says otherwise, like a rule in code. An agent runs an authored policy when its code names it (`oversight = "refunds"`), when it is `DEFAULT_POLICY`, or when a [runtime switch](/oversight-switches/) assigns it.

From Python, `AuthoringService` does the same, with the same checks:

```python
from django_ai_sdk.oversight.authoring import AuthoringService
from django_ai_sdk.oversight.authoring.schemas import PolicyIn

await AuthoringService.save_policy("refunds", PolicyIn(**definition), user=request.user)
```

## What a definition may not do

An authored policy is checked as code's are (every lens has what answers it, every rule acts, a Mask points at an Entities lens, no cycle), and on top of that it may not:

- **take a code policy's name**: code wins, and the save is refused;
- **redefine an inherited lens or rule**: a code rule's test, actions and `min_mode` floor stay code's. Change an inherited rule's mode with a [runtime switch](/oversight-switches/), within its floor;
- **configure a lens beyond a safe set**: inline only keywords (a pattern lens, or a `Patterns` judge, of whole words) and the built-in lenses `pii` (by kind), `secrets`, `refusal`, `length` and `language`, with plain arguments; a check lens names validators the developers registered (`register_validator`). No regex, no code, no import path: a pattern an author writes would run on every turn and could backtrack for seconds. A model judge, a classifier or a regex comes as an alias in `AI_SDK_OVERSIGHT["JUDGES"]`, which the deployment sets;
- **leave out what `MUST_EXTEND` names**;
- **read a lens it does not have**, or **grow beyond** 50 lenses, 100 rules or 256 KiB.

## Try before saving

`POST .../try/` runs a draft (or, without `definition`, the saved version) over one exchange through gate, the tool calls, finish and reflect, and records nothing: the same `testing.dry_run` code uses. Its judges are real, so a judge it names is called and costs what a turn costs; that is why trying needs `AUTHOR_OVERSIGHT`.

## The record

Every save is a new version: an `AuthoredPolicy` row with the next `version`, the whole definition, who saved it and why. The newest version runs; earlier ones stay. Withdrawing saves a version marked `withdrawn`. Every save and withdrawal also writes an `AccessRecord` with operation `author` (`policy`, `version`, the version `before` it, `withdrawn`, `reason`) and sends an `oversight.authoring` event to every notifier. With the [ledger](/oversight-ledger/) installed each version is chained too, so a version changed in the database afterwards fails `ledger_verify`.

A policy cannot be withdrawn while another authored policy extends it, a runtime switch assigns it to an agent, or it is `DEFAULT_POLICY`: their turns would be refused.

## When a change applies

A save applies at once in the process that made it. Every other process reads the authored policies at most once per `AI_SDK_OVERSIGHT["RUNTIME_CACHE_SECONDS"]` (default `10`, shared with switches), so for a few seconds two workers may run different versions. When the database cannot be read, a process keeps what it registered last.

A saved version that no longer loads (a deploy removed a code policy it extends, or a new version of its parent dropped a lens it reads) is withdrawn from the registry, not skipped: turns that name it are refused (unless `FAIL_MODE` is `"open"`), the reason is logged, and `GET .../policies/{name}/` shows it in `error`.

## Settings

```python
AI_SDK_OVERSIGHT_AUTHORING = {
    "MUST_EXTEND": [],          # policies every authored policy must extend
    "INLINE": ["keywords", "pii", "secrets", "refusal", "length", "language"],
}
```

`INLINE` can only narrow the safe set; naming anything else is an error.

## Checks

| Id | Level | When |
| --- | --- | --- |
| `ai_sdk.oversight.authoring.E012` | Error | `AI_SDK_OVERSIGHT_AUTHORING` has an unknown key, a value of the wrong kind, or an `INLINE` name outside the safe set. |
| `ai_sdk.oversight.authoring.E002` | Error | `MUST_EXTEND` names a policy no code registers. |
| `ai_sdk.oversight.authoring.W001` | Warning (deploy) | `MUST_EXTEND` is empty: an operator can author a policy without the protective rules code declares, and switch an agent onto it. |

## Limitations

- Saving a new version of a policy others extend is not checked against them first; a child that no longer loads is refused, and shows its `error`.
- A definition cannot declare a new judge, validator or condition operator: those take code.
- The editor is yours to build: the routes and the catalog are what it needs.
