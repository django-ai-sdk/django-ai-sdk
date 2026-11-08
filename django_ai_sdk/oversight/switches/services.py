"""Switch a rule's mode or an agent's policy at runtime, and list the switches in force.

A switch that weakens oversight needs a reason and sends an "oversight.switch" event.
Code sets the floor: no switch takes a rule below its `min_mode` or turns off an agent
with `oversight_required`.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from asgiref.sync import sync_to_async

from django_ai_sdk.errors import UserError
from django_ai_sdk.logger import get_logger
from django_ai_sdk.oversight.rules import below
from django_ai_sdk.oversight.services.access import (
    OversightAccess,
    append_change,
    require_reason,
)
from django_ai_sdk.oversight.switches.overlay import in_force, rule_modes
from django_ai_sdk.oversight.switches.schemas import AgentPolicyIn, RuleModeIn, SwitchOut
from django_ai_sdk.permissions import Operation

if TYPE_CHECKING:
    from collections.abc import Callable

    from django_ai_sdk.oversight.switches.models import Switch
    from django_ai_sdk.types import UserType

logger = get_logger(__name__)

# The access record's operation for a switch.
OPERATION = "switch"
VIEW = Operation.VIEW_OVERSIGHT_SWITCHES
SWITCH = Operation.SWITCH_OVERSIGHT


class SwitchService(OversightAccess):
    """Manage the rule modes and agent policies that operators switch."""

    @classmethod
    async def list_switches(cls, *, user: UserType) -> list[SwitchOut]:
        """Return every switch in force, rules first."""
        from django_ai_sdk.oversight.switches.models import Switch

        await cls.has_perms(user, VIEW)
        rows = in_force().order_by("-kind", "policy", "target")
        # Cache so each policy is built and resolved only once.
        policies: dict[str, tuple[Any, Any]] = {}
        described = []
        async for row in rows:
            if row.kind == Switch.Kind.RULE_MODE:
                described.append(await _describe_rule(row, row.policy, row.target, policies))
            else:
                found = await sync_to_async(_find_agent)(row.target)
                agent = found[0] if found else None
                described.append(await _describe_agent(row, row.target, agent))
        return described

    @classmethod
    async def set_rule_mode(
        cls, policy: str, rule: str, change: RuleModeIn, *, user: UserType
    ) -> SwitchOut | None:
        """Switch `rule` in `policy` and its children. None if the rule does not exist."""
        from django_ai_sdk.oversight.checks.policies import policy_errors
        from django_ai_sdk.oversight.policy import build
        from django_ai_sdk.oversight.registry import get_policy_class
        from django_ai_sdk.oversight.resolve import code_ancestors, invalidate
        from django_ai_sdk.oversight.switches.models import Switch

        await cls.has_perms(user, SWITCH)
        if get_policy_class(policy) is None:
            return None
        try:
            declared = build(policy)
        except Exception as exc:
            raise UserError(f"Policy {policy!r} does not load: {exc}") from exc
        if rule not in declared.rules:
            return None
        mode = change.mode

        def weakens(_switched: str | None) -> bool:
            # Parents' switches reach this policy too, so build it with them.
            ancestors = rule_modes([policy, *code_ancestors(policy)])
            own = ancestors.pop(policy)
            current = build(policy, modes=own, ancestors=ancestors)
            before = current.rules[rule].mode
            kept = {name: value for name, value in own.items() if name != rule}
            if mode is not None:
                kept[rule] = str(mode)
            trial = build(policy, modes=kept, ancestors=ancestors)
            after = trial.rules[rule].mode
            if mode is not None and after is not mode:
                # The build refuses a mode below the rule's floor.
                reasons = [r["reason"] for r in trial.rejected if r["rule"] == rule]
                raise UserError(reasons[-1] if reasons else f"{rule!r} cannot be {mode}")
            # The system checks only see code modes, so a switch that would make
            # them fail (an enforced rule no judge can answer) is refused here.
            new = set(policy_errors(policy, trial)) - set(policy_errors(policy, current))
            if new:
                raise UserError(" ".join(sorted(new)))
            return below(after, before)

        row = await _write(
            Switch.Kind.RULE_MODE,
            policy,
            rule,
            None if mode is None else str(mode),
            change.reason,
            user,
            weakens,
        )
        invalidate(policy)
        return await _describe_rule(_switched(row), policy, rule, {})

    @classmethod
    async def set_agent_policy(
        cls, agent_id: str, change: AgentPolicyIn, *, user: UserType
    ) -> SwitchOut | None:
        """Switch the policy of `agent_id`. None if the agent does not exist."""
        from django_ai_sdk.oversight.registry import get_policy_class
        from django_ai_sdk.oversight.resolve import invalidate
        from django_ai_sdk.oversight.switches.models import Switch

        await cls.has_perms(user, SWITCH)
        found = await sync_to_async(_find_agent)(agent_id)
        if found is None:
            return None
        # Stored as turns look it up, whatever spelling of the id was asked for.
        agent, agent_id = found
        value = change.policy
        if value and get_policy_class(value) is None:
            raise UserError(f"No policy {value!r}.")
        required = bool(getattr(agent, "oversight_required", False))
        if value == "" and required:
            raise UserError("This agent requires oversight: it cannot be switched off.")
        code = await _code_policy(agent)
        if value is not None:
            _check_target(code, value)

        def weakens(current: str | None) -> bool:
            if value == "":
                # Off weakens whatever ran before, even when code's choice is unknown here.
                return current != ""
            before = _runs(current, code, required)
            return bool(before) and _runs(value, code, required) != before

        row = await _write(
            Switch.Kind.AGENT_POLICY, "", agent_id, value, change.reason, user, weakens
        )
        # Rereads the agent assignments at once in this process.
        invalidate()
        return await _describe_agent(_switched(row), agent_id, agent)


async def _write(
    kind: str,
    policy: str,
    target: str,
    value: str | None,
    reason: str,
    user: UserType,
    weakens: Callable[[str | None], bool],
) -> Switch:
    """Append the switch and its access record, retrying on a concurrent write."""
    from django_ai_sdk.oversight.switches.models import Switch

    reason = reason.strip()

    def describe(current: Switch | None) -> tuple[dict[str, Any], dict[str, Any] | None]:
        before = current.value if current is not None else None
        downgrade = weakens(before)
        if downgrade:
            require_reason(reason, "Say why: this switch weakens oversight.")
        change = {
            "kind": kind,
            "policy": policy,
            "target": target,
            "value": value,
            "before": before,
            "reason": reason,
        }
        event = {"type": "oversight.switch", **change} if downgrade else None
        return {**change, "weakens": downgrade}, event

    return await append_change(
        Switch,
        "override",
        {"kind": kind, "policy": policy, "target": target},
        {"value": value, "reason": reason},
        user=user,
        operation=OPERATION,
        describe=describe,
        conflict="Switched by someone else at the same time; try again.",
    )


def _check_target(code: str, value: str) -> None:
    """Refuse a policy that does not load, or that drops a floor of the agent's code policy.

    "" is off: no rule runs, so every floor is dropped.
    """
    from django_ai_sdk.oversight.policy import build
    from django_ai_sdk.oversight.rules import floor_problems

    rules: dict[str, Any] = {}
    if value:
        try:
            rules = build(value).rules
        except Exception as exc:
            raise UserError(f"Policy {value!r} does not load: {exc}") from exc
    try:
        floored = build(code).rules if code else {}
    except Exception:
        # Code's own policy does not load, so its turns are refused anyway (FAIL_MODE).
        floored = {}
    if problems := floor_problems(floored, rules):
        raise UserError(
            f"{value or 'Off'} drops what {code!r} sets in code: " + "; ".join(problems)
        )


def _switched(row: Switch) -> Switch | None:
    """The row, or None when it hands back to code."""
    return row if row.value is not None else None


def _runs(switched: str | None, code: str, required: bool) -> str:
    """Return the policy an agent runs under.

    The switched value wins, unless there is none or it is "off" for an agent that
    requires oversight.
    """
    if switched is None or (switched == "" and required):
        return code
    return switched


async def _describe_rule(
    row: Switch | None, policy: str, rule: str, policies: dict[str, tuple[Any, Any]]
) -> SwitchOut:
    """Describe a rule: its code mode, its switch and its effective mode."""
    from django_ai_sdk.oversight.switches.models import Switch

    code, effective = await _rule_modes(policy, rule, policies)
    problem = ""
    if row is not None and policies.get(policy, (None,))[0] is None:
        problem = f"Policy {policy!r} does not load."
    elif row is not None and not code:
        problem = f"Policy {policy!r} has no rule {rule!r} any more."
    return _switch_out(Switch.Kind.RULE_MODE, policy, rule, row, code, effective, problem)


async def _describe_agent(row: Switch | None, agent_id: str, agent: Any) -> SwitchOut:
    """Describe an agent: its code policy, its switch and the policy it runs under."""
    from django_ai_sdk.oversight.resolve import aget_policy, broken_reason
    from django_ai_sdk.oversight.switches.models import Switch

    code = await _code_policy(agent) if agent is not None else ""
    required = bool(getattr(agent, "oversight_required", False))
    effective = _runs(row.value if row is not None else None, code, required)
    problem = ""
    if row is not None and agent is None:
        problem = "No such agent any more."
    elif effective and await aget_policy(effective) is None:
        # Its turns are refused (FAIL_MODE "closed") or run unguarded ("open").
        problem = f"Policy {effective!r} does not load. {broken_reason(effective)}".strip()
    return _switch_out(Switch.Kind.AGENT_POLICY, "", agent_id, row, code, effective, problem)


def _switch_out(
    kind: str,
    policy: str,
    target: str,
    row: Switch | None,
    code: str,
    effective: str,
    problem: str = "",
) -> SwitchOut:
    return SwitchOut(
        kind=kind,
        policy=policy,
        target=target,
        code=code,
        switched=row.value if row is not None else None,
        effective=effective,
        reason=row.reason if row is not None else "",
        changed_by=row.changed_by_id if row is not None else "",
        changed_at=row.changed_at if row is not None else None,
        problem=problem,
    )


async def _rule_modes(
    policy: str, rule: str, policies: dict[str, tuple[Any, Any]]
) -> tuple[str, str]:
    """Return a rule's (code mode, effective mode). Each is "" when the rule is missing.

    `policies` caches each policy's built and resolved forms across calls.
    """
    from django_ai_sdk.oversight.policy import build
    from django_ai_sdk.oversight.resolve import aget_policy

    if policy not in policies:
        try:
            built = build(policy)
        except Exception:
            built = None
        policies[policy] = (built, await aget_policy(policy))
    built, resolved = policies[policy]
    declared = built.rules.get(rule) if built is not None else None
    running = resolved.rules.get(rule) if resolved is not None else None
    return (
        str(declared.mode) if declared else "",
        str(running.mode) if running else "",
    )


async def _code_policy(agent: Any) -> str:
    """Return the policy the agent's code names outside a turn.

    Falls back to the class attribute when the agent's hook raises.
    """
    from django_ai_sdk.oversight.agents import acode_policy, code_policy

    try:
        return (await acode_policy(agent))[0]
    except Exception:
        logger.opt(exception=True).warning("Oversight could not ask an agent for its policy")
        return code_policy(agent)[0]


def _find_agent(agent_id: str) -> tuple[Any, str] | None:
    """Find a registered agent, else an active runtime agent, by id, with the key turns use.

    None if neither exists.

    Not `AgentService.get`: when a runtime agent's class fails to build, this still
    answers with that class's oversight floor.
    """
    from django.core.exceptions import ValidationError

    from django_ai_sdk.agents.config import get_runtime_agent_class
    from django_ai_sdk.agents.models import AgentSettings
    from django_ai_sdk.agents.registry import registry
    from django_ai_sdk.oversight.agents import agent_key

    try:
        agent = registry.get(agent_id)
    except RuntimeError:
        agent = None
    if agent is not None:
        return agent, agent_key(agent) or agent_id
    try:
        config = AgentSettings.objects.filter(pk=agent_id, active=True).first()
    except (ValidationError, ValueError):
        return None
    if config is None:
        return None
    cls = get_runtime_agent_class(config.agent)
    key = str(config.pk)
    try:
        return cls(config), key
    except Exception:
        logger.opt(exception=True).warning("Could not build runtime agent {}", config.slug)
        # Still answer with the class's oversight settings, so the floor holds.
        return SimpleNamespace(
            oversight=getattr(cls, "oversight", None),
            oversight_required=getattr(cls, "oversight_required", False),
        ), key


__all__ = ["SwitchService"]
