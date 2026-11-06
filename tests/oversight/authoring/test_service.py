"""Authoring policies at runtime: who may, what a definition may hold, what turns then run."""

from __future__ import annotations

import pytest
from django_ai_sdk.errors import UserError
from django_ai_sdk.oversight.authoring.models import AuthoredPolicy
from django_ai_sdk.oversight.authoring.schemas import PolicyIn, TryIn, WithdrawIn
from django_ai_sdk.oversight.authoring.services import AuthoringService
from django_ai_sdk.oversight.guard import TurnGuard
from django_ai_sdk.oversight.models import AccessRecord
from django_ai_sdk.oversight.registry import get_policy_class
from django_ai_sdk.oversight.resolve import aget_policy
from django_ai_sdk.permissions import PermissionDenied

from tests.oversight.authoring.helpers import AUTHOR, VIEW, events, refunds
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

    async def test_every_save_says_why(self):
        with pytest.raises(UserError, match="Say why"):
            await save(reason=" ")
        assert not await AuthoredPolicy.objects.aexists()


@pytest.mark.usefixtures("no_reflect_queue", "sent")
@pytest.mark.django_db(transaction=True)
class TestSaving:
    async def test_a_saved_policy_runs_like_code(self):
        out = await save()
        assert (out.version, out.loaded, out.error) == (1, True, "")
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

    async def test_a_new_version_replaces_the_old_and_keeps_it(self):
        await save()
        rules = refunds()["rules"]
        rules[0]["mode"] = "report"
        out = await save(rules=rules, reason="too strict")
        assert out.version == 2
        assert not (await gate("refunds", "I want a refund")).blocked
        detail = await AuthoringService.get_policy("refunds", user=await author())
        assert detail is not None and [v.version for v in detail.versions] == [2, 1]

    async def test_every_save_leaves_an_access_record_and_notifies(self, sent):
        await save()
        record = await AccessRecord.objects.aget(operation="author")
        assert record.query["policy"] == "refunds" and record.query["version"] == 1
        assert record.query["reason"] == "refunds are handled by people"
        (event,) = events(sent)
        assert event["type"] == "oversight.authoring" and event["policy"] == "refunds"
        # The event carries the same change as the record.
        assert {k: event[k] for k in record.query} == record.query

    async def test_withdrawing_refuses_the_turns_that_name_it(self):
        await save()
        out = await AuthoringService.withdraw_policy(
            "refunds", WithdrawIn(reason="replaced"), user=await author()
        )
        assert out is not None and out.withdrawn and out.version == 2
        assert get_policy_class("refunds") is None
        refused = await gate("refunds", "hello")
        assert refused.blocked

    async def test_a_policy_still_extended_cannot_be_withdrawn(self):
        await save()
        await save("refunds-eu", extends=["refunds"], lenses=[], rules=[])
        with pytest.raises(UserError, match="refunds-eu"):
            await AuthoringService.withdraw_policy(
                "refunds", WithdrawIn(reason="x"), user=await author()
            )


@pytest.mark.usefixtures("no_reflect_queue", "sent")
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
        assert "inherited" in await self.refused(rules=rules)

    async def test_redefine_an_inherited_lens(self):
        lenses = [{**refunds()["lenses"][0], "name": "mentions_x"}]
        assert "inherited" in await self.refused(lenses=lenses)

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

    async def test_use_a_preset_the_deployment_left_out(self, settings):
        settings.AI_SDK_OVERSIGHT_AUTHORING = {"INLINE": ["keywords"]}
        lens = {"kind": "pattern", "preset": "pii", "name": "refund"}
        assert "cannot be used here" in await self.refused(lenses=[lens])

    async def test_read_a_lens_it_does_not_have(self):
        rules = [{**refunds()["rules"][0], "when": {"lens": "nope", "op": "is_true"}}]
        assert await self.refused(rules=rules)

    async def test_leave_out_what_must_be_extended(self, settings):
        settings.AI_SDK_OVERSIGHT_AUTHORING = {"MUST_EXTEND": ["au-base"]}
        assert "MUST_EXTEND" in await self.refused(extends=[])

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


@pytest.mark.usefixtures("no_reflect_queue", "sent")
@pytest.mark.django_db(transaction=True)
class TestOtherProcesses:
    async def test_another_process_follows_and_withdraws(self, fresh_loader):
        from django_ai_sdk.oversight.authoring.loader import AuthoredPolicyLoader

        await save()
        other = AuthoredPolicyLoader()
        # Another process: its registry has not seen the save yet.
        from django_ai_sdk.oversight.registry import unregister_policy

        unregister_policy("refunds")
        await other.load()
        assert get_policy_class("refunds") is not None
        await AuthoringService.withdraw_policy(
            "refunds", WithdrawIn(reason="x"), user=await author()
        )
        await other.load()
        assert get_policy_class("refunds") is None

    async def test_a_version_that_no_longer_loads_is_refused_not_skipped(self):
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


@pytest.mark.usefixtures("no_reflect_queue", "sent")
@pytest.mark.django_db(transaction=True)
class TestWithTheLedger:
    async def test_every_version_is_chained_and_a_changed_one_is_caught(self):
        from django.apps import apps

        if not apps.is_installed("django_ai_sdk.oversight.ledger"):
            pytest.skip("the ledger app is not installed")
        from asgiref.sync import sync_to_async
        from django_ai_sdk.oversight import custody
        from django_ai_sdk.oversight.ledger import chain
        from django_ai_sdk.oversight.ledger.custody import LedgerCustody

        before = custody.get()
        custody.install(LedgerCustody())
        try:
            await save()
            await save(reason="again")
            assert (await sync_to_async(chain.verify)()).ok
            await AuthoredPolicy.objects.filter(version=1).aupdate(reason="rewritten")
            problems = (await sync_to_async(chain.verify)()).problems
            assert [p["kind"] for p in problems] == ["authored"]
        finally:
            custody.install(before)
