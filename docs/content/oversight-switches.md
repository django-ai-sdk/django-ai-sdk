---
title: Runtime switches
type: docs
weight: 11
---

**Flip a rule from report to enforce, or give an agent another policy, without a deploy.**

Runtime switches are an opt-in app on top of [oversight](/oversight/) and its [review API](/manual/oversight/#review-api). Two things switch, nothing else:

- **A rule's mode** in a policy: `off`, `report` or `enforce`. A switch on a policy reaches every policy that extends it.
- **The policy an agent runs under**: another registered policy, or none (off).

Each switch is a `Switch` row. In core the seam it plugs into is the overlay (`OverlaySource`): values laid over what code declares. The switches app is the overlay that reads its `Switch` rows.

What a rule tests, its actions and thresholds, a lens and its judges stay in code: changing them takes a deploy.

## Install

```python
INSTALLED_APPS = [
    ...,
    "django_ai_sdk",
    "django_ai_sdk.oversight",
    "django_ai_sdk.oversight.switches",
]
```

Run `manage.py migrate`. On start the app installs its overlay (`django_ai_sdk.oversight.resolve.set_overlay_source`), so every process, task workers included, reads the switches. Remove the app and turns run on code alone: core never imports it.

To remove it, first run `manage.py migrate django_ai_sdk_oversight_switches zero`, then take it out of `INSTALLED_APPS`. That drops the `Switch` table with every switch in it. If you take the app out without it, the table stays, and adding the app back later brings back every switch that was active then, downgrades included. The `AccessRecord` of each switch stays either way.

With the app installed, `get_oversight_router()` also serves its routes, next to the review routes. Leave them out with its `exclude=`. The [manual's permission table](/manual/oversight/#review-api) names the model permission behind each operation.

| Route | Body | Permission |
| --- | --- | --- |
| `GET /oversight/switches/` | | `VIEW_OVERSIGHT_SWITCHES` |
| `POST /oversight/switches/rules/{policy}/{rule}/` | `{"mode": "enforce", "reason": ""}`; `"mode": null` hands the rule back to code | `SWITCH_OVERSIGHT` |
| `POST /oversight/switches/agents/{agent_id}/` | `{"policy": "support", "reason": ""}`; `""` is off, `null` hands the agent back to code | `SWITCH_OVERSIGHT` |

Each answers the rule or agent as it now stands: `code` (what code declares), `switched` (the operator's value, `null` when code decides) and `effective` (what turns run). An unknown policy, rule or agent is 404; a mode that is not `off`, `report` or `enforce` is 422; two operators switching the same rule at the same moment may get 409. Agent ids are those of the agents API: a registered agent's id, or a runtime agent's (`AgentSettings`) id.

From Python, `SwitchService` does the same, with the same checks:

```python
from django_ai_sdk.oversight.switches import SwitchService
from django_ai_sdk.oversight.switches.schemas import RuleModeIn

await SwitchService.set_rule_mode(
    "support", "block_injection", RuleModeIn(mode="enforce"), user=request.user
)
```

## A switch that weakens oversight

A switch weakens oversight when it lowers a rule's mode (`enforce` → `report` → `off`, a hand-back to code included), switches an agent off, or moves an agent from the policy it runs under to another one. Such a switch:

- **needs a reason** (`reason`): without one it is refused (400);
- **notifies**: an `oversight.switch` event goes to every notifier in `NOTIFIERS`, from the task worker, with the switch, the value before it, the reason and who made it;
- **stops at code's floor**: no switch takes a rule below its `min_mode` (`Rule(..., min_mode="enforce")`), and no switch turns off an agent whose class sets `oversight_required = True`. The floor holds in core too: an overlay entry below `min_mode` is rejected where the policy resolves, and an overlay's "off" for such an agent is ignored.

An agent's policy switch keeps those floors too. The policy it moves to must load, and must carry every rule of the agent's code policy that has a `min_mode`, with the same condition and actions, at its floor or above and keeping the floor. Off carries none, so an agent whose code policy has a floor cannot be switched off either (400). When the agent's code picks its policy per user or channel (`get_oversight()`), the switch cannot tell which one, and checks no floor; switching such an agent off still counts as weakening.

```python
class Support(Policy):
    name = "support"
    extends = ["security"]
    rules = [
        Rule("block_injection", when=L("prompt_injection").is_true, then=Block(),
             mode="enforce", min_mode="enforce"),
    ]


class HelpdeskAgent(Agent):
    oversight = Support
    oversight_required = True
```

## The record

Every switch, whether it weakens oversight or not, writes an `AccessRecord` with operation `switch`: who, when, and in `query` the switch (`kind`, `policy`, `target`), its `value`, the value `before` it, the `reason` and whether it `weakens` oversight. The switch itself is an `Override` row (`kind` `rule_mode` or `agent_policy`) with the next `version` of that rule or agent; the newest is in force and the older ones are its history. Handing a rule or agent back to code adds a version without a `value`. A refused switch writes nothing.

## When a switch applies

A switch applies to the next turn in the process that made it, except a turn whose policy was already being resolved. Every other process, each web worker and task worker, reads the switches at most once per `AI_SDK_OVERSIGHT["RUNTIME_CACHE_SECONDS"]` (default `10`, shared with authored policies): while the database answers, a switch reaches all of them within that time, so for a few seconds two workers may run a rule in different modes. `0` reads the switches on every turn: the agent assignments once, and the rule modes once per policy.

When the database cannot be read, a process keeps the switches it read last, for as long as the database stays away; one that has read none runs on code alone until a read succeeds.

A switch that no longer applies says why in `problem`: its rule left the policy, its agent is gone, or the policy it names does not load (that agent's turns are refused, or with `FAIL_MODE` `"open"` run unguarded).

A delegation from sync code (a Haystack pipeline calling `run()`) cannot read the switches itself. It uses the agent assignments this process read last, while they are within `RUNTIME_CACHE_SECONDS`; before any turn has read them, or once they are stale, it refuses a subagent that would need its own turn.

## Checks

`ai_sdk.oversight.switches.W001`: runtime agents are set up (`AI_SDK_RUNTIME_AGENT_BASES`), `DEFAULT_POLICY` is empty, and a runtime base class names no policy. Such an agent runs unguarded until someone switches it to a policy, and any user who may create an agent can make one. Set `DEFAULT_POLICY`, or give the base class an `oversight` policy.

The switches app registers this check itself, so a project without the app does not get it. There a runtime agent runs under the policy its base class or `DEFAULT_POLICY` gives it, and nothing can switch it at runtime; set one of the two in code.
