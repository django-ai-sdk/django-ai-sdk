"""Authoring policies at runtime: who may, what a definition may hold, what turns then run."""

from __future__ import annotations

import pytest
from django_ai_sdk.errors import UserError
from django_ai_sdk.oversight.authoring.models import AuthoredPolicy
from django_ai_sdk.oversight.authoring.schemas import PolicyIn, TryIn
from django_ai_sdk.oversight.authoring.services import AuthoringService
from django_ai_sdk.oversight.guard import TurnGuard
from django_ai_sdk.oversight.registry import get_policy_class
from django_ai_sdk.oversight.resolve import aget_policy
from django_ai_sdk.permissions import PermissionDenied

from tests.oversight.authoring.helpers import AUTHOR, VIEW, refunds
from tests.oversight.helpers import make_agent
from tests.oversight.review.helpers import reader


async def author():
    return await reader(AUTHOR, VIEW)


async def save(name: str = "refunds", **changes):
    return await AuthoringService.save_policy(
        name, PolicyIn(**refunds(**changes)), user=await author()
    )


def judged(*judges) -> dict:
    """An authored judge lens named like the refunds lens, answered by `judges`."""
    return {"kind": "judge", "name": "refund", "stages": ["gate"], "judges": list(judges)}


async def gate(policy: str, text: str):
    guard = await TurnGuard.start(make_agent([], None, policy=policy), channel="api", input=text)
    return await guard.gate()


@pytest.mark.django_db(transaction=True)
class TestWhoMayAuthor:
    async def test_staff_alone_may_neither_see_nor_author(self):
        staff = await reader(staff=True)
        with pytest.raises(PermissionDenied):
            await AuthoringService.list_policies(user=staff)
        with pytest.raises(PermissionDenied):
            await AuthoringService.save_policy("refunds", PolicyIn(**refunds()), user=staff)

    async def test_seeing_is_not_authoring_or_trying(self):
        viewer = await reader(VIEW)
        assert await AuthoringService.list_policies(user=viewer) == []
        with pytest.raises(PermissionDenied):
            await AuthoringService.save_policy("refunds", PolicyIn(**refunds()), user=viewer)
        with pytest.raises(PermissionDenied):
            await AuthoringService.try_policy("refunds", TryIn(input="hi"), user=viewer)
        assert not await AuthoredPolicy.objects.aexists()

@pytest.mark.usefixtures("no_reflect_queue")
@pytest.mark.django_db(transaction=True)
class TestSaving:
    async def test_a_saved_policy_runs_like_code(self):
        out = await save()
        assert (out.active, out.loaded, out.error) == (True, True, "")
        refused = await gate("refunds", "I want a refund")
        assert refused.blocked and refused.message == "Refunds go through support."
        # What it extends still runs.
        assert (await gate("refunds", "a xylophone")).blocked
        assert not (await gate("refunds", "hello")).blocked
        resolved = await aget_policy("refunds")
        assert resolved is not None and resolved.origin == "authored"

    async def test_a_judge_lens_of_keywords_and_a_registered_check_run_too(self):
        from django_ai_sdk.oversight import Violation, register_validator

        @register_validator("test_no_iban_word")
        def no_iban_word(text):
            if "iban" in text.lower():
                raise Violation("No account numbers here.")

        lenses = [
            judged({"class": "Patterns", "keywords": ["refund"]}),
            {
                "kind": "check",
                "name": "iban",
                "stages": ["gate"],
                "validators": ["test_no_iban_word"],
            },
        ]
        rules = [
            *refunds()["rules"],
            {
                "name": "no_iban",
                "when": {"lens": "iban", "op": "is_true"},
                "then": [{"kind": "block", "message": "No."}],
                "mode": "enforce",
            },
        ]
        await save(lenses=lenses, rules=rules)
        assert (await gate("refunds", "I want a refund")).blocked
        assert (await gate("refunds", "my IBAN please")).message == "No."
        assert not (await gate("refunds", "hello")).blocked

    async def test_a_save_replaces_the_definition(self):
        await save()
        rules = refunds()["rules"]
        rules[0]["mode"] = "report"
        await save(rules=rules)
        assert not (await gate("refunds", "I want a refund")).blocked
        assert await AuthoredPolicy.objects.acount() == 1

    async def test_deleting_refuses_the_turns_that_name_it(self):
        await save()
        assert await AuthoringService.delete_policy("refunds", user=await author())
        assert get_policy_class("refunds") is None
        assert not await AuthoredPolicy.objects.aexists()
        assert (await gate("refunds", "hello")).blocked
        assert not await AuthoringService.delete_policy("refunds", user=await author())

    async def test_a_policy_still_extended_cannot_be_deleted(self):
        await save()
        await save("refunds-eu", extends=["refunds"], lenses=[], rules=[])
        with pytest.raises(UserError, match="refunds-eu"):
            await AuthoringService.delete_policy("refunds", user=await author())


@pytest.mark.usefixtures("no_reflect_queue")
@pytest.mark.django_db(transaction=True)
class TestToggling:
    async def test_an_own_rule_changes_mode(self):
        await save()
        out = await AuthoringService.set_rule_mode(
            "refunds", "no_refunds", "report", user=await author()
        )
        assert out is not None and out.rules[0]["mode"] == "report"
        assert not (await gate("refunds", "I want a refund")).blocked

    async def test_an_own_lens_turns_off(self):
        await save()
        await AuthoringService.set_lens_enabled("refunds", "refund", False, user=await author())
        assert not (await gate("refunds", "I want a refund")).blocked

    async def test_a_policy_turned_off_still_runs_what_it_extends(self):
        await save()
        out = await AuthoringService.set_active("refunds", False, user=await author())
        assert out is not None and not out.active and out.loaded
        assert not (await gate("refunds", "I want a refund")).blocked
        assert (await gate("refunds", "a xylophone")).blocked
        await AuthoringService.set_active("refunds", True, user=await author())
        assert (await gate("refunds", "I want a refund")).blocked

    async def test_an_inherited_rule_or_lens_cannot_be_toggled(self):
        await save()
        user = await author()
        with pytest.raises(UserError, match="changes only in code"):
            await AuthoringService.set_rule_mode("refunds", "block_x", "off", user=user)
        with pytest.raises(UserError, match="changes only in code"):
            await AuthoringService.set_lens_enabled("refunds", "mentions_x", False, user=user)
        assert (await gate("refunds", "a xylophone")).blocked

    async def test_a_code_policy_cannot_be_toggled(self):
        assert await AuthoringService.set_active("au-base", False, user=await author()) is None
        assert (await gate("au-base", "a xylophone")).blocked

    async def test_toggling_takes_authoring(self):
        await save()
        with pytest.raises(PermissionDenied):
            await AuthoringService.set_active("refunds", False, user=await reader(VIEW))


@pytest.mark.usefixtures("no_reflect_queue")
@pytest.mark.django_db(transaction=True)
class TestWhatADefinitionMayNotDo:
    async def refused(self, name: str = "refunds", **changes) -> str:
        with pytest.raises(UserError) as caught:
            await save(name, **changes)
        assert not await AuthoredPolicy.objects.filter(name=name).aexists()
        return str(caught.value)

    async def test_take_a_code_policys_name(self):
        assert "declared in code" in await self.refused("au-base")

    async def test_redefine_an_inherited_rule(self):
        rules = [{**refunds()["rules"][0], "name": "block_x", "mode": "report"}]
        assert "'block_x' of 'au-base' is replaced" in await self.refused(rules=rules)

    async def test_redefine_an_inherited_lens(self):
        lenses = [{**refunds()["lenses"][0], "name": "mentions_x"}]
        assert "'mentions_x' of 'au-base' is replaced" in await self.refused(lenses=lenses)

    async def test_configure_a_regex_inline(self):
        lenses = [{**refunds()["lenses"][0], "patterns": ["(a+)+$"]}]
        assert "does not take ['patterns']" in await self.refused(lenses=lenses)

    async def test_give_a_patterns_judge_a_regex(self):
        regex = {"class": "Patterns", "patterns": ["(a+)+$"]}
        assert "only keywords" in await self.refused(lenses=[judged(regex)])

    async def test_import_any_class(self):
        assert "an alias, or Patterns" in await self.refused(
            lenses=[judged({"class": "os.system"})]
        )

    async def test_name_an_alias_nobody_set_up(self):
        assert "no judge alias" in await self.refused(lenses=[judged("no-such-judge")])

    async def test_name_a_validator_nobody_registered(self):
        lens = {"kind": "check", "name": "refund", "validators": ["no_such_validator"]}
        assert "No validator" in await self.refused(lenses=[lens])

    async def test_read_a_lens_it_does_not_have(self):
        rules = [{**refunds()["rules"][0], "when": {"lens": "nope", "op": "is_true"}}]
        assert await self.refused(rules=rules)

    async def test_leave_out_what_must_be_extended(self, settings):
        settings.AI_SDK_OVERSIGHT_AUTHORING = {"MUST_EXTEND": ["au-base"]}
        assert "MUST_EXTEND" in await self.refused(extends=[])

    async def test_open_up_a_parent_that_fails_closed(self):
        # A failed lens would let au-base's enforced block through.
        assert "fail_mode 'open'" in await self.refused(options={"fail_mode": "open"})

    async def test_stream_an_answer_it_withholds_at_finish(self):
        # W011, which the system check never sees for an authored policy.
        lenses = [{**refunds()["lenses"][0], "stages": ["finish"], "subject": "output"}]
        refused = await self.refused(lenses=lenses, options={"stream_mode": "live"})
        assert "streams it live" in refused

    async def test_grow_without_bound(self):
        lenses = [{**refunds()["lenses"][0], "name": f"l{i}"} for i in range(51)]
        assert "at most 50 lenses" in await self.refused(lenses=lenses, rules=[])


@pytest.mark.usefixtures("no_reflect_queue")
@pytest.mark.django_db(transaction=True)
class TestTrying:
    async def test_a_draft_runs_without_being_saved(self):
        draft = PolicyIn(**refunds())
        out = await AuthoringService.try_policy(
            "refunds", TryIn(definition=draft, input="I want a refund"), user=await author()
        )
        assert out.outcome == "blocked" and out.ended_at == "gate"
        assert "no_refunds" in out.stages[0].rules
        assert not await AuthoredPolicy.objects.aexists()
        assert get_policy_class("refunds") is None


@pytest.mark.usefixtures("no_reflect_queue")
@pytest.mark.django_db(transaction=True)
class TestOtherProcesses:
    async def test_another_process_follows_and_unregisters(self, fresh_loader):
        from django_ai_sdk.oversight.authoring.loader import AuthoredPolicyLoader

        await save()
        other = AuthoredPolicyLoader()
        # Another process: its registry has not seen the save yet.
        from django_ai_sdk.oversight.registry import unregister_policy

        unregister_policy("refunds")
        await other.load()
        assert get_policy_class("refunds") is not None
        await AuthoringService.delete_policy("refunds", user=await author())
        await other.load()
        assert get_policy_class("refunds") is None

    async def test_a_policy_that_no_longer_loads_is_refused_not_skipped(self):
        await save()
        # A deploy removed the code policy it extends.
        from django_ai_sdk.oversight.authoring.loader import loader
        from django_ai_sdk.oversight.registry import unregister_policy

        unregister_policy("au-base")
        # Nothing in the table changed, so make the loader read it again.
        loader.stamp = None
        await loader.load()
        assert get_policy_class("refunds") is None
        assert "au-base" in loader.broken["refunds"]
        assert (await gate("refunds", "hello")).blocked


OFF_X = {
    "lenses": [{"kind": "pattern", "name": "never", "stages": ["gate"], "keywords": ["zzzq"]}],
    "rules": [
        {
            "name": "block_x",
            "when": {"lens": "never", "op": "is_true"},
            "then": [{"kind": "alert"}],
            "mode": "off",
        }
    ],
}


@pytest.mark.usefixtures("no_reflect_queue")
@pytest.mark.django_db(transaction=True)
class TestCodeKeepsItsRules:
    async def test_a_second_parent_cannot_replace_a_code_rule(self):
        """The later parent would win in the build, so `block_x` would be p1's: off."""
        await save("p1", extends=[], **OFF_X)
        with pytest.raises(UserError, match="'block_x' of 'au-base' is replaced"):
            await save("child", extends=["au-base", "p1"], lenses=[], rules=[])
        assert (await gate("au-base", "a xylophone")).blocked

    async def test_a_preset_cannot_replace_a_lens_by_its_default_name(self, registry_snapshot):
        from django_ai_sdk.oversight import Block, L, Policy, Rule
        from django_ai_sdk.oversight.packs.security import PIILens
        from django_ai_sdk.oversight.registry import register_policy

        class Guarded(Policy):
            name = "au-pii"
            lenses = [PIILens(stages=["gate"])]
            rules = [Rule("block_pii", when=L("pii").is_true, then=Block(), mode="enforce")]

        register_policy(Guarded)
        with pytest.raises(UserError, match="'pii' of 'au-pii' is replaced"):
            await save(
                "pii-off",
                extends=["au-pii"],
                lenses=[{"preset": "pii", "enabled": False}],
                rules=[],
            )


@pytest.mark.usefixtures("no_reflect_queue")
@pytest.mark.django_db(transaction=True)
class TestStillUsed:
    async def test_a_policy_an_agent_names_in_code_cannot_be_deleted(self, monkeypatch):
        from django_ai_sdk.agents.registry import registry

        await save()
        agent = make_agent([], None, policy="refunds")
        monkeypatch.setitem(registry._instances, "refund-bot", agent)
        with pytest.raises(UserError, match="name it in code"):
            await AuthoringService.delete_policy("refunds", user=await author())


@pytest.mark.usefixtures("no_reflect_queue")
@pytest.mark.django_db(transaction=True)
async def test_a_code_policy_that_does_not_load_cannot_be_stood_in_for(registry_snapshot):
    from django_ai_sdk.oversight import PatternLens, Policy
    from django_ai_sdk.oversight.registry import get_invalid_policies, register_policy

    class Broken(Policy):
        name = "care"
        lenses = [PatternLens(name="twice", keywords=["a"]), PatternLens(name="twice", keywords=["b"])]  # noqa: RUF012

    register_policy(Broken)
    assert "care" in get_invalid_policies()
    with pytest.raises(UserError, match="declared in code that does not load"):
        await save("care", extends=["au-base"], lenses=[], rules=[])
    assert "care" in get_invalid_policies()
