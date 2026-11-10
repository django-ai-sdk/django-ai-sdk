"""Switching a rule's mode and an agent's policy: who may, what turns then run, the record."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from django_ai_sdk.errors import UserError
from django_ai_sdk.oversight.guard import TurnGuard
from django_ai_sdk.oversight.models import AccessRecord
from django_ai_sdk.oversight.resolve import aget_policy
from django_ai_sdk.oversight.switches.models import Switch
from django_ai_sdk.oversight.switches.schemas import AgentPolicyIn, RuleModeIn
from django_ai_sdk.oversight.switches.services import SwitchService
from django_ai_sdk.oversight.types import Mode
from django_ai_sdk.permissions import PermissionDenied
from tests.oversight.helpers import make_agent
from tests.oversight.review.helpers import reader
from tests.oversight.switches.helpers import SWITCH, VIEW, events


async def operator():
    return await reader(SWITCH, VIEW)


async def mode_of(policy: str, rule: str) -> Mode:
    resolved = await aget_policy(policy)
    assert resolved is not None
    return resolved.rules[rule].mode


@pytest.mark.django_db(transaction=True)
class TestWhoMaySwitch:
    async def test_staff_alone_may_neither_see_nor_switch(self):
        staff = await reader(staff=True)
        with pytest.raises(PermissionDenied):
            await SwitchService.list_switches(user=staff)
        with pytest.raises(PermissionDenied):
            await SwitchService.set_rule_mode(
                "sw-parent", "block_x", RuleModeIn(mode="enforce"), user=staff
            )

    async def test_seeing_the_switches_is_not_switching(self):
        viewer = await reader(VIEW)
        assert await SwitchService.list_switches(user=viewer) == []
        with pytest.raises(PermissionDenied):
            await SwitchService.set_agent_policy("a-1", AgentPolicyIn(policy=""), user=viewer)
        assert not await Switch.objects.aexists()
        assert not await AccessRecord.objects.filter(operation="switch").aexists()


@pytest.mark.usefixtures("no_reflect_queue", "sent")
@pytest.mark.django_db(transaction=True)
class TestRuleModes:
    async def test_enforcing_a_rule_blocks_the_next_turn(self):
        agent = make_agent([], None, policy="sw-parent")
        assert not (await (await start(agent, "a xylophone")).gate()).blocked

        out = await SwitchService.set_rule_mode(
            "sw-parent", "block_x", RuleModeIn(mode="enforce"), user=await operator()
        )
        assert out is not None
        assert (out.code, out.switched, out.effective) == ("report", "enforce", "enforce")
        assert (await (await start(agent, "a xylophone")).gate()).blocked

    async def test_switching_a_parent_reaches_the_policies_extending_it(self):
        await SwitchService.set_rule_mode(
            "sw-parent", "block_x", RuleModeIn(mode="enforce"), user=await operator()
        )
        child = await aget_policy("sw-child")
        assert child is not None
        assert child.rules["block_x"].mode is Mode.ENFORCE
        assert child.sources["rule:block_x.mode"] == "overlay:sw-parent"

    async def test_back_to_code_adds_a_history_row(self):
        user = await operator()
        await SwitchService.set_rule_mode("sw-parent", "block_x", RuleModeIn(mode="enforce"), user=user)
        out = await SwitchService.set_rule_mode(
            "sw-parent", "block_x", RuleModeIn(mode=None, reason="false positives"), user=user
        )
        assert out is not None
        assert (out.switched, out.effective) == (None, "report")
        assert await mode_of("sw-parent", "block_x") is Mode.REPORT
        versions = [(row.version, row.value) async for row in Switch.objects.order_by("version")]
        assert versions == [(1, "enforce"), (2, None)]
        assert await SwitchService.list_switches(user=user) == []

    async def test_an_unknown_policy_or_rule_is_not_found(self):
        user = await operator()
        change = RuleModeIn(mode="enforce")
        assert await SwitchService.set_rule_mode("nope", "block_x", change, user=user) is None
        assert await SwitchService.set_rule_mode("sw-parent", "nope", change, user=user) is None

    def test_only_a_mode_is_a_mode(self):
        with pytest.raises(ValidationError):
            RuleModeIn(mode="enforced")  # ty: ignore[invalid-argument-type]
        with pytest.raises(ValidationError):
            RuleModeIn(mode="hold")  # ty: ignore[invalid-argument-type]

    async def test_lowering_a_mode_needs_a_reason_and_notifies(self, sent):
        user = await operator()
        with pytest.raises(UserError, match="Say why"):
            await SwitchService.set_rule_mode("sw-child", "note_x", RuleModeIn(mode="report"), user=user)
        assert await mode_of("sw-child", "note_x") is Mode.ENFORCE
        assert not sent.called

        await SwitchService.set_rule_mode(
            "sw-child", "note_x", RuleModeIn(mode="off", reason="noisy in the test week"), user=user
        )
        assert await mode_of("sw-child", "note_x") is Mode.OFF
        [event] = events(sent)
        assert event["type"] == "oversight.switch"
        assert (event["policy"], event["target"], event["value"]) == ("sw-child", "note_x", "off")
        assert event["reason"] == "noisy in the test week"
        assert event["by"].endswith(f"({user.pk})")

    async def test_raising_a_mode_needs_no_reason_and_sends_nothing(self, sent):
        await SwitchService.set_rule_mode(
            "sw-parent", "block_x", RuleModeIn(mode="enforce"), user=await operator()
        )
        assert not sent.called

    async def test_handing_back_to_code_that_lowers_the_mode_needs_a_reason(self, sent):
        user = await operator()
        await SwitchService.set_rule_mode("sw-parent", "block_x", RuleModeIn(mode="enforce"), user=user)
        with pytest.raises(UserError, match="Say why"):
            await SwitchService.set_rule_mode("sw-parent", "block_x", RuleModeIn(mode=None), user=user)
        assert await mode_of("sw-parent", "block_x") is Mode.ENFORCE

    async def test_no_switch_goes_below_the_floor_code_set(self):
        user = await operator()
        with pytest.raises(UserError, match="min_mode"):
            await SwitchService.set_rule_mode(
                "sw-floored", "always_block_x", RuleModeIn(mode="report", reason="r"), user=user
            )
        with pytest.raises(UserError, match="min_mode"):
            await SwitchService.set_rule_mode(
                "sw-floored", "watch_x", RuleModeIn(mode="off", reason="r"), user=user
            )
        await SwitchService.set_rule_mode(
            "sw-floored", "watch_x", RuleModeIn(mode="report", reason="r"), user=user
        )
        assert await mode_of("sw-floored", "always_block_x") is Mode.ENFORCE
        assert await mode_of("sw-floored", "watch_x") is Mode.REPORT
        # A row written behind the service's back is rejected where the policy resolves.
        await Switch.objects.filter(target="watch_x").aupdate(value="off")
        from django_ai_sdk.oversight.resolve import invalidate

        invalidate()
        resolved = await aget_policy("sw-floored")
        assert resolved is not None
        assert resolved.rules["watch_x"].mode is Mode.ENFORCE
        assert [r["rule"] for r in resolved.rejected] == ["watch_x"]

    async def test_no_switch_enforces_a_rule_no_judge_can_answer(self):
        # E007 would catch it in code; the system check never sees a switch.
        with pytest.raises(UserError, match="no judge here can answer"):
            await SwitchService.set_rule_mode(
                "sw-blind", "block_blind", RuleModeIn(mode="enforce"), user=await operator()
            )
        assert await mode_of("sw-blind", "block_blind") is Mode.REPORT


@pytest.mark.usefixtures("no_reflect_queue", "sent")
@pytest.mark.django_db(transaction=True)
class TestAccessRecords:
    async def test_every_switch_writes_an_access_record(self, registered):
        user = await operator()
        agent_id = registered(make_agent([], None, policy="sw-parent"))
        await SwitchService.set_rule_mode("sw-parent", "block_x", RuleModeIn(mode="enforce"), user=user)
        await SwitchService.set_rule_mode(
            "sw-parent", "block_x", RuleModeIn(mode=None, reason="back"), user=user
        )
        await SwitchService.set_agent_policy(agent_id, AgentPolicyIn(policy="sw-child", reason="stricter"), user=user)
        await SwitchService.set_agent_policy(
            agent_id, AgentPolicyIn(policy="", reason="retired"), user=user
        )
        records = [r async for r in AccessRecord.objects.filter(operation="switch").order_by("created_at")]
        assert [(r.query["target"], r.query["value"], r.query["weakens"]) for r in records] == [
            ("block_x", "enforce", False),
            ("block_x", None, True),
            (agent_id, "sw-child", True),
            (agent_id, "", True),
        ]
        assert {r.reader_id for r in records} == {str(user.pk)}
        assert records[1].query["before"] == "enforce"
        assert records[3].query["reason"] == "retired"
        assert all(r.turn_ids == [] and not r.content for r in records)

    async def test_a_refused_switch_writes_nothing(self):
        with pytest.raises(UserError):
            await SwitchService.set_rule_mode(
                "sw-floored", "always_block_x", RuleModeIn(mode="off", reason="r"), user=await operator()
            )
        assert not await AccessRecord.objects.filter(operation="switch").aexists()
        assert not await Switch.objects.aexists()


@pytest.mark.usefixtures("no_reflect_queue", "sent")
@pytest.mark.django_db(transaction=True)
class TestAgentPolicies:
    async def test_the_switched_policy_wins_over_code(self, registered):
        user = await operator()
        agent = make_agent([], None, policy="sw-parent")
        agent_id = registered(agent)

        out = await SwitchService.set_agent_policy(
            agent_id, AgentPolicyIn(policy="", reason="decommissioned"), user=user
        )
        assert out is not None
        assert (out.code, out.switched, out.effective) == ("sw-parent", "", "")
        assert not (await TurnGuard.start(agent, channel="api", input="hi")).active

        out = await SwitchService.set_agent_policy(
            agent_id, AgentPolicyIn(policy="sw-child"), user=user
        )
        assert out is not None and out.effective == "sw-child"
        guard = await TurnGuard.start(agent, channel="api", input="hi")
        assert guard.policy is not None and guard.policy.name == "sw-child"

        out = await SwitchService.set_agent_policy(
            agent_id, AgentPolicyIn(policy=None, reason="back"), user=user
        )
        assert out is not None
        assert (out.switched, out.effective) == (None, "sw-parent")
        with pytest.raises(UserError):
            await SwitchService.set_agent_policy(agent_id, AgentPolicyIn(policy="nope"), user=user)
        assert await SwitchService.set_agent_policy("missing", AgentPolicyIn(policy=""), user=user) is None

    async def test_an_agent_that_requires_oversight_is_never_switched_off(self, registered):
        agent = make_agent([], None, policy="sw-parent")
        type(agent).oversight_required = True
        agent_id = registered(agent)
        user = await operator()
        with pytest.raises(UserError, match="requires oversight"):
            await SwitchService.set_agent_policy(
                agent_id, AgentPolicyIn(policy="", reason="r"), user=user
            )
        out = await SwitchService.set_agent_policy(
            agent_id, AgentPolicyIn(policy="sw-child", reason="stricter"), user=user
        )
        assert out is not None and out.effective == "sw-child"

    async def test_taking_an_agents_policy_away_needs_a_reason_and_notifies(self, registered, sent):
        agent_id = registered(make_agent([], None, policy="sw-parent"))
        user = await operator()
        with pytest.raises(UserError, match="Say why"):
            await SwitchService.set_agent_policy(agent_id, AgentPolicyIn(policy=""), user=user)
        assert not sent.called
        await SwitchService.set_agent_policy(agent_id, AgentPolicyIn(policy="", reason="r"), user=user)
        [event] = events(sent)
        assert (event["kind"], event["target"], event["value"], event["before"]) == (
            "agent_policy",
            agent_id,
            "",
            None,
        )

    async def test_guarding_an_unguarded_agent_needs_no_reason(self, registered, sent):
        agent_id = registered(make_agent([], None, policy=""))
        out = await SwitchService.set_agent_policy(
            agent_id, AgentPolicyIn(policy="sw-parent"), user=await operator()
        )
        assert out is not None and (out.code, out.effective) == ("", "sw-parent")
        assert not sent.called

    async def test_a_runtime_agent_is_switched_by_its_id(self):
        from django_ai_sdk.agents.models import AgentSettings
        from django_ai_sdk.agents.runtime import RuntimeAgent

        row = await AgentSettings.objects.acreate(name="coach", slug="coach", model="m")
        # Any spelling of the id is stored as turns look it up.
        shouted = row.id.hex.upper()
        out = await SwitchService.set_agent_policy(
            shouted, AgentPolicyIn(policy="sw-parent"), user=await operator()
        )
        assert out is not None and out.effective == "sw-parent"
        assert out.target == str(row.id)
        guard = await TurnGuard.start(RuntimeAgent(row), channel="chat", input="x")
        assert guard.policy is not None and guard.policy.name == "sw-parent"

    async def test_a_policy_swap_keeps_the_floors_of_the_agents_code_policy(self, registered):
        """Moving a floored agent to a policy without its rule would drop the floor."""
        agent_id = registered(make_agent([], None, policy="sw-floored"))
        user = await operator()
        for target in ("sw-parent", ""):
            with pytest.raises(UserError, match="always_block_x"):
                await SwitchService.set_agent_policy(
                    agent_id, AgentPolicyIn(policy=target, reason="r"), user=user
                )
        assert not await Switch.objects.aexists()

    async def test_a_policy_that_does_not_load_is_no_target(self, registered):
        from django_ai_sdk.oversight import Policy
        from django_ai_sdk.oversight.registry import register_policy

        class Broken(Policy):
            name = "sw-broken"
            extends = ["no-such-parent"]  # noqa: RUF012

        register_policy(Broken)
        agent_id = registered(make_agent([], None, policy="sw-parent"))
        with pytest.raises(UserError, match="does not load"):
            await SwitchService.set_agent_policy(
                agent_id, AgentPolicyIn(policy="sw-broken", reason="r"), user=await operator()
            )

    async def test_off_weakens_even_when_code_cannot_say_what_ran(self, registered, sent):
        """An agent that picks its policy per user reads as unguarded outside a turn."""
        agent = make_agent([], None, policy="sw-parent")
        agent.get_oversight = lambda **kw: (_ for _ in ()).throw(RuntimeError("needs a user"))
        agent_id = registered(agent)
        with pytest.raises(UserError, match="Say why"):
            await SwitchService.set_agent_policy(
                agent_id, AgentPolicyIn(policy=""), user=await operator()
            )

    async def test_the_list_says_when_a_switch_no_longer_applies(self, registered):
        user = await operator()
        await Switch.objects.acreate(
            kind=Switch.Kind.RULE_MODE,
            policy="sw-parent",
            target="gone_rule",
            value="enforce",
            version=1,
        )
        [listed] = await SwitchService.list_switches(user=user)
        assert "no rule 'gone_rule'" in listed.problem

    async def test_the_list_shows_what_is_switched(self, registered):
        user = await operator()
        agent_id = registered(make_agent([], None, policy="sw-parent"))
        await SwitchService.set_agent_policy(agent_id, AgentPolicyIn(policy="sw-child", reason="r"), user=user)
        await SwitchService.set_rule_mode("sw-parent", "block_x", RuleModeIn(mode="enforce"), user=user)
        listed = await SwitchService.list_switches(user=user)
        assert [(s.kind, s.target, s.switched, s.effective) for s in listed] == [
            ("rule_mode", "block_x", "enforce", "enforce"),
            ("agent_policy", agent_id, "sw-child", "sw-child"),
        ]


async def start(agent, text: str) -> TurnGuard:
    return await TurnGuard.start(agent, channel="api", input=text)


@pytest.mark.usefixtures("no_reflect_queue", "sent")
@pytest.mark.django_db(transaction=True)
async def test_listing_builds_each_policy_once_however_many_of_its_rules_are_switched():
    from unittest.mock import patch

    from django_ai_sdk.oversight import policy as policies

    user = await operator()
    for rule in ("always_block_x", "watch_x"):
        await SwitchService.set_rule_mode(
            "sw-floored", rule, RuleModeIn(mode="enforce", reason="r"), user=user
        )
    await SwitchService.set_rule_mode("sw-parent", "block_x", RuleModeIn(mode="enforce"), user=user)
    with patch.object(policies, "build", wraps=policies.build) as built:
        listed = await SwitchService.list_switches(user=user)
    assert [(s.policy, s.target, s.effective) for s in listed] == [
        ("sw-floored", "always_block_x", "enforce"),
        ("sw-floored", "watch_x", "enforce"),
        ("sw-parent", "block_x", "enforce"),
    ]
    assert sorted(call.args[0] for call in built.call_args_list) == ["sw-floored", "sw-parent"]
