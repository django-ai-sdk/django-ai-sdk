"""The ledger: content kept in the vault, every record chained, and a chain that says what
was changed or removed behind its back."""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Any
from unittest.mock import patch

import pytest
from asgiref.sync import sync_to_async
from django.apps import apps
from django.db import connection
from django.test.utils import CaptureQueriesContext
from django.utils import timezone

from django_ai_sdk.oversight import custody
from django_ai_sdk.oversight.content import load_content
from django_ai_sdk.oversight.guard import TurnGuard
from django_ai_sdk.oversight.ledger import chain
from django_ai_sdk.oversight.ledger.custody import LedgerCustody
from django_ai_sdk.oversight.ledger.models import LedgerEntry, LedgerHead
from django_ai_sdk.oversight.ledger.schemas import VerifyOut
from django_ai_sdk.oversight.models import LensAnswer, RuleMatch, Turn, TurnContent
from django_ai_sdk.vault.models import DataKey
from tests.oversight.helpers import a_thread, make_agent
from tests.oversight.review.helpers import CONTENT, METRICS, a_turn, reader

QUESTION = "my salary is low, mail jan@example.com about it"
ANSWER = "Your salary is fine."


async def turn_in(thread_id: str | None = None, text: str = QUESTION) -> str:
    """A closed test-guard turn kept with full capture, in `thread_id` when given."""
    return await a_turn(text, ANSWER, thread_id=thread_id or "")


async def verified() -> VerifyOut:
    return await sync_to_async(chain.verify)()


async def kinds() -> list[str]:
    return [e.kind async for e in LedgerEntry.objects.order_by("seq")]


def raw(table: str) -> str:
    with connection.cursor() as cursor:
        cursor.execute(f"SELECT * FROM {table}")  # noqa: S608 - fixed table names
        return repr(cursor.fetchall())


@pytest.mark.django_db
def test_the_migration_made_the_head():
    head = LedgerHead.objects.get(id=1)
    assert (head.seq, head.entry_hash) == (0, "0" * 64)


def test_the_app_installs_its_custody():
    custody.install(None)
    apps.get_app_config("django_ai_sdk_oversight_ledger").ready()
    assert isinstance(custody.get(), LedgerCustody)


def test_the_app_needs_the_vault():
    from django.core.exceptions import ImproperlyConfigured

    config = apps.get_app_config("django_ai_sdk_oversight_ledger")
    with (
        patch.object(apps, "is_installed", lambda name: name != "django_ai_sdk.vault"),
        pytest.raises(ImproperlyConfigured, match="django_ai_sdk.vault"),
    ):
        config.ready()


@pytest.mark.django_db(transaction=True)
class TestContent:
    async def test_content_is_encrypted_under_the_turns_scope(self):
        turn_id = await turn_in()
        turn = await Turn.objects.select_related("content").aget(id=turn_id)
        assert turn.content is not None
        stored = turn.content.stored
        assert set(stored) == {"vault", "digest", "aside"}  # a judge's reason was kept
        assert stored["vault"]["scope"] == f"turn:{turn_id}"
        dump = await sync_to_async(raw)("django_ai_sdk_oversight_content")
        assert "salary" not in dump and "jan@example.com" not in dump
        content = await sync_to_async(load_content)(turn)
        assert content["input"] == QUESTION
        assert content["output"] == ANSWER

    async def test_a_thread_keeps_its_turns_under_one_key(self):
        thread = await a_thread()
        await turn_in(str(thread.id))
        await turn_in(str(thread.id), "and my leave?")
        assert await DataKey.objects.filter(scope=f"thread:{thread.id}").acount() == 1
        assert await DataKey.objects.acount() == 1

    async def test_what_judges_add_later_keeps_the_digest_and_chains_the_quotes(self):
        from django_ai_sdk.oversight.content import add_to_content

        turn_id = await turn_in()
        turn = await Turn.objects.select_related("content").aget(id=turn_id)
        before = turn.content.stored["digest"]
        assert (await kinds()).count("aside") == 1
        assert await sync_to_async(add_to_content)(turn, notes={"r1": "looked fine"})
        turn = await Turn.objects.select_related("content").aget(id=turn_id)
        assert turn.content.stored["digest"] == before
        assert (await sync_to_async(load_content)(turn))["notes"] == {"r1": "looked fine"}
        assert (await kinds()).count("aside") == 2
        chained = await sync_to_async(raw)("django_ai_sdk_oversight_ledger")
        assert "looked fine" not in chained and "found" not in chained
        assert (await verified()).ok

    async def test_the_feed_opens_every_turns_content_in_a_fixed_number_of_queries(self):
        from asgiref.sync import async_to_sync

        from django_ai_sdk.oversight.services import OversightService

        user = await reader(METRICS, CONTENT)
        await turn_in()
        await OversightService.list_turns(user=user)  # the permission cache warms up

        def counted() -> int:
            with CaptureQueriesContext(connection) as captured:
                page = async_to_sync(OversightService.list_turns)(user=user)
            assert all(row.preview for row in page.items)
            return len(captured.captured_queries)

        await turn_in()
        few = await sync_to_async(counted)()
        for _ in range(4):
            await turn_in()
        assert await sync_to_async(counted)() == few

    async def test_a_threads_content_unwraps_its_key_once_per_read_and_per_verify(self):
        from django_ai_sdk.oversight.models import TurnContent
        from django_ai_sdk.vault import scopes

        thread = await a_thread()
        for _ in range(4):
            await turn_in(str(thread.id))

        def unwraps(run) -> int:
            with patch.object(scopes, "provider", wraps=scopes.provider) as provider:
                run()
            return provider.call_count

        def read() -> None:
            rows = list(TurnContent.objects.all())
            loaded = custody.get().load_many([(r.scope, r.stored) for r in rows])
            assert all(loaded)

        assert await sync_to_async(unwraps)(read) == 1
        assert await sync_to_async(unwraps)(lambda: chain.verify()) == 1


@pytest.mark.django_db(transaction=True)
class TestChain:
    async def test_a_turn_chains_its_content_its_rows_and_itself_twice(self):
        turn_id = await turn_in()
        found = await kinds()
        # Written open when the gate ends, then closed with its content.
        assert found[0] == "turn"
        assert found.count("turn") == 2
        assert found.count("content") == 1
        assert found.index("content") < len(found) - found[::-1].index("turn") - 1
        answers = await LensAnswer.objects.filter(turn_id=turn_id).acount()
        matches = await RuleMatch.objects.filter(turn_id=turn_id).acount()
        assert found.count("answer") == answers > 0
        assert found.count("match") == matches > 0
        result = await verified()
        assert result.ok, result.problems
        assert result.checked == len(found)
        head = await LedgerHead.objects.aget(id=1)
        assert head.seq == len(found)
        assert result.head == head.entry_hash

    async def test_entries_hold_no_content(self):
        await turn_in()
        dump = await sync_to_async(raw)("django_ai_sdk_oversight_ledger")
        assert "salary" not in dump and "jan@example.com" not in dump

    async def test_a_changed_outcome_is_caught(self):
        turn_id = await turn_in()
        await Turn.objects.filter(id=turn_id).aupdate(outcome="allowed", report_outcome="allowed")
        result = await verified()
        assert not result.ok
        assert {p["problem"] for p in result.problems} == {"record altered"}
        assert result.problems[0]["kind"] == "turn"

    async def test_what_changes_by_design_after_close_is_not_hashed(self):
        turn_id = await turn_in()
        await Turn.objects.filter(id=turn_id).aupdate(
            labels={"x": 1}, severity="high", error="reflect broke", status="interrupted"
        )
        await Turn.objects.filter(id=turn_id).aupdate(actor_id="")
        assert (await verified()).ok

    async def test_a_changed_answer_is_caught(self):
        turn_id = await turn_in()
        answer = await LensAnswer.objects.filter(turn_id=turn_id, lens="topic").afirst()
        await LensAnswer.objects.filter(pk=answer.pk).aupdate(value="other")
        result = await verified()
        assert [(p["problem"], p["kind"]) for p in result.problems] == [
            ("record altered", "answer")
        ]

    async def test_a_deleted_match_is_caught(self):
        turn_id = await turn_in()
        await RuleMatch.objects.filter(turn_id=turn_id).adelete()
        result = await verified()
        assert result.problems and {p["problem"] for p in result.problems} == {"record missing"}

    async def test_a_changed_entry_breaks_the_links_after_it(self):
        await turn_in()
        await LedgerEntry.objects.filter(seq=2).aupdate(payload_hash="0" * 64)
        problems = {p["problem"] for p in (await verified()).problems}
        assert "entry altered" in problems

    async def test_dropping_the_newest_entries_is_caught(self):
        await turn_in()
        last = await LedgerEntry.objects.order_by("-seq").afirst()
        await LedgerEntry.objects.filter(seq=last.seq).adelete()
        assert "chain ends early" in {p["problem"] for p in (await verified()).problems}

    async def test_content_swapped_for_plain_text_is_caught(self):
        turn_id = await turn_in()
        turn = await Turn.objects.select_related("content").aget(id=turn_id)
        await TurnContent.objects.filter(pk=turn.content_id).aupdate(
            stored={"input": "something else"}
        )
        result = await verified()
        assert [(p["problem"], p["kind"]) for p in result.problems] == [
            ("record altered", "content"),
            ("record altered", "aside"),
        ]

    async def test_content_kept_under_a_forged_digest_is_caught(self):
        turn_id = await turn_in()
        other_id = await turn_in("Something else entirely")
        turn = await Turn.objects.select_related("content").aget(id=turn_id)
        other = await Turn.objects.select_related("content").aget(id=other_id)
        # The other turn's ciphertext under this turn's digest: it does not open here.
        forged = {**other.content.stored, "digest": turn.content.stored["digest"]}
        await TurnContent.objects.filter(pk=turn.content_id).aupdate(stored=forged)
        assert not (await verified()).ok

    async def requoted(self, change: Any, **stored: Any) -> str:
        """Re-encrypt a turn's content with its quotes changed, keeping its digests."""
        from django_ai_sdk.vault import encrypt_json

        turn = await Turn.objects.select_related("content").aget(id=await turn_in())
        content = await sync_to_async(load_content)(turn)
        change(content)
        envelope = await sync_to_async(encrypt_json)(turn.content.scope, content)
        await TurnContent.objects.filter(pk=turn.content_id).aupdate(
            stored={**turn.content.stored, "vault": envelope, **stored}
        )
        return [(p["problem"], p["kind"]) for p in (await verified()).problems]

    async def test_a_judges_reason_swapped_under_the_same_digests_is_caught(self):
        def swap(content: dict[str, Any]) -> None:
            for text in content["texts"].values():
                text["rationale"] = "nothing found"

        assert await self.requoted(swap) == [("quotes altered", "content")]

    async def test_judges_reasons_dropped_from_the_content_are_caught(self):
        assert await self.requoted(lambda content: content.pop("texts")) == [
            ("quotes altered", "content")
        ]

    async def test_quotes_kept_under_a_recomputed_aside_digest_are_caught(self):
        from django_ai_sdk.oversight.ledger.custody import aside_digest

        turn = await Turn.objects.select_related("content").aget(id=await turn_in())
        content = await sync_to_async(load_content)(turn)
        content["notes"] = {"r1": "approved"}
        forged = await sync_to_async(LedgerCustody().store)(turn.content.scope, content)
        assert forged["aside"] == await sync_to_async(aside_digest)(turn.content.scope, content)
        await TurnContent.objects.filter(pk=turn.content_id).aupdate(stored=forged)
        assert [(p["problem"], p["kind"]) for p in (await verified()).problems] == [
            ("record altered", "aside")
        ]

    async def test_content_without_quotes_needs_no_aside(self):
        stored = await sync_to_async(LedgerCustody().store)("turn:quiet", {"input": "hello"})
        assert set(stored) == {"vault", "digest"}

    async def test_appending_needs_a_transaction(self):
        turn_id = await turn_in()
        turn = await Turn.objects.aget(id=turn_id)
        with pytest.raises(RuntimeError, match="atomic"):
            await sync_to_async(chain.append_many)([("turn", turn)])


@pytest.mark.usefixtures("no_reflect_queue")
@pytest.mark.django_db(transaction=True)
class TestForgetting:
    async def test_a_deleted_thread_shreds_its_key_and_chains_the_forget(self):
        thread = await a_thread()
        thread_id = str(thread.id)
        turn_id = await turn_in(thread_id)
        await thread.adelete()
        key = await DataKey.objects.aget(scope=f"thread:{thread_id}")
        assert key.shredded_at is not None and key.wrapped is None
        assert not await TurnContent.objects.filter(scope=f"thread:{thread_id}").aexists()
        assert await Turn.objects.filter(id=turn_id).aexists()
        assert (await kinds())[-2:] == ["forget", "erase"]
        result = await verified()
        assert result.ok, result.problems
        assert result.forgotten == 1

    async def test_the_next_turn_in_a_forgotten_thread_keeps_no_content(self):
        thread = await a_thread()
        thread_id = str(thread.id)
        await turn_in(thread_id)
        await sync_to_async(custody.get().forget)(f"thread:{thread_id}")
        later = await turn_in(thread_id)
        turn = await Turn.objects.select_related("content").aget(id=later)
        assert turn.content is None
        dump = await sync_to_async(raw)("django_ai_sdk_oversight_content")
        assert "salary" not in dump
        assert (await verified()).ok

    async def test_a_deleted_users_threadless_turns_are_shredded(self):
        from tests.factories.db import UserFactory

        user = await UserFactory.acreate()
        guard = await TurnGuard.start(
            make_agent([], None), channel="api", input=QUESTION, user=user
        )
        await guard.gate()
        await guard.finish(ANSWER)
        turn_id = await guard.close()
        await user.adelete()
        assert (await DataKey.objects.aget(scope=f"turn:{turn_id}")).shredded_at is not None
        assert (await Turn.objects.aget(id=turn_id)).actor_id == ""
        result = await verified()
        assert result.ok, result.problems

    async def test_forgetting_twice_chains_once(self):
        await turn_in("t-1")
        forget = sync_to_async(custody.get().forget)
        await forget("thread:t-1")
        await forget("thread:t-1")
        assert (await kinds()).count("forget") == 1

    async def test_an_unshredded_key_is_caught(self):
        thread = await a_thread()
        scope = f"thread:{thread.id}"
        await turn_in(str(thread.id))
        await thread.adelete()
        assert await DataKey.objects.filter(scope=scope).aupdate(wrapped=b"back") == 1
        result = await verified()
        assert [(p["problem"], p["kind"]) for p in result.problems] == [
            ("record altered", "forget")
        ]


# Another master key: what the test key kept does not open under it alone.
OTHER_KEY = "b3RoZXItdmF1bHQtbWFzdGVyLWtleS1mb3ItdGVzdHM="


@pytest.mark.usefixtures("no_reflect_queue")
@pytest.mark.django_db(transaction=True)
class TestWhatVerifyCannotOpen:
    """SEC-7: content verify cannot open is reported, never passed as fine."""

    async def test_content_under_a_master_key_not_here_is_unreadable(self, settings):
        await turn_in()
        settings.AI_SDK_VAULT = {"MASTER_KEYS": [OTHER_KEY]}
        result = await verified()
        assert result.ok, result.problems
        assert (result.state, result.unreadable) == ("unreadable", 1)
        assert result.unread[0]["key_id"]
        assert result.model_dump()["state"] == "unreadable"

    async def test_with_the_key_back_it_verifies_again(self, settings):
        from django.conf import settings as live

        test_key = live.AI_SDK_VAULT["MASTER_KEYS"][0]
        await turn_in()
        settings.AI_SDK_VAULT = {"MASTER_KEYS": [OTHER_KEY, test_key]}
        result = await verified()
        assert (result.state, result.unreadable) == ("ok", 0)

    async def test_altered_ciphertext_under_a_key_that_is_here_is_caught(self):
        turn_id = await turn_in()
        turn = await Turn.objects.select_related("content").aget(id=turn_id)
        stored = turn.content.stored
        ct = stored["vault"]["ct"]
        flipped = ct[:-6] + ("A" if ct[-6] != "A" else "B") + ct[-5:]
        await TurnContent.objects.filter(pk=turn.content_id).aupdate(
            stored={**stored, "vault": {**stored["vault"], "ct": flipped}}
        )
        result = await verified()
        assert [(p["problem"], p["kind"]) for p in result.problems] == [
            ("content altered", "content")
        ]

    async def test_a_deleted_key_row_is_caught(self):
        await turn_in()
        await DataKey.objects.all().adelete()
        problems = [p["problem"] for p in (await verified()).problems]
        assert problems == ["key missing"]

    async def test_a_key_shredded_behind_the_chains_back_is_caught(self):
        from django_ai_sdk import vault

        turn_id = await turn_in()
        await sync_to_async(vault.shred)(f"turn:{turn_id}")
        result = await verified()
        assert not result.ok
        assert {p["problem"] for p in result.problems} == {"shred not chained"}
        # Both the content row it no longer opens and the key itself are named.
        assert {p["kind"] for p in result.problems} == {"content", "forget"}

    async def test_a_shredded_key_with_no_content_left_is_still_caught(self):
        from django_ai_sdk import vault

        turn_id = await turn_in()
        await sync_to_async(vault.shred)(f"turn:{turn_id}")
        await TurnContent.objects.all().adelete()
        problems = {(p["problem"], p["kind"]) for p in (await verified()).problems}
        assert ("shred not chained", "forget") in problems
        assert ("record missing", "content") in problems

    async def test_a_scope_another_app_shreds_is_not_oversights(self):
        from django_ai_sdk import vault

        await turn_in()
        await sync_to_async(vault.encrypt_text)("invoice:1", "x")
        await sync_to_async(vault.shred)("invoice:1")
        assert (await verified()).ok

    async def test_content_deleted_outside_forgetting_and_retention_is_caught(self):
        await turn_in()
        await TurnContent.objects.all().adelete()
        result = await verified()
        assert [(p["problem"], p["kind"]) for p in result.problems] == [
            ("record missing", "content"),
            ("record missing", "aside"),
        ]

    async def test_an_erase_covers_only_content(self):
        turn_id = await turn_in()
        content_id = (await Turn.objects.aget(id=turn_id)).content_id

        def erase_and_drop_the_turn() -> None:
            from django.db import transaction

            with transaction.atomic():
                chain.erase([content_id])
                # Claimed as erased, but a match is deleted with it: still missing.
                RuleMatch.objects.filter(turn_id=turn_id).delete()
                TurnContent.objects.filter(pk=content_id).delete()

        await sync_to_async(erase_and_drop_the_turn)()
        result = await verified()
        assert {(p["problem"], p["kind"]) for p in result.problems} == {("record missing", "match")}
        assert result.forgotten == 1

    async def test_content_expired_ahead_of_its_turn_is_chained_as_pruned(self):
        from django.db import transaction

        from django_ai_sdk.oversight import retention

        turn_id = await turn_in()
        content_id = (await Turn.objects.aget(id=turn_id)).content_id

        def expire() -> int:
            with transaction.atomic():
                return retention._delete_content([content_id])

        assert await sync_to_async(expire)() == 1
        assert (await kinds())[-1] == "prune"
        result = await verified()
        assert result.ok, result.problems
        assert (result.pruned, result.forgotten) == (1, 0)


@pytest.mark.django_db(transaction=True)
class TestRetention:
    async def test_pruned_turns_are_chained_as_pruned(self):
        from django_ai_sdk.oversight.retention import prune

        await turn_in()
        await turn_in()
        later = timezone.now() + timedelta(days=10_000)
        counts = await sync_to_async(prune)(now=later, batch_size=1)
        assert counts["turns"] == 2
        assert not await Turn.objects.aexists()
        found = await kinds()
        assert "prune" in found
        # Retention's own mark is chained as an access record.
        assert found[-1] == "access"
        result = await verified()
        assert result.ok, result.problems
        assert result.pruned > 0

    async def test_access_records_past_the_metadata_retention_are_chained_as_pruned(self):
        from django_ai_sdk.oversight.models import AccessRecord
        from django_ai_sdk.oversight.retention import prune
        from django_ai_sdk.oversight.services import OversightService

        await turn_in()
        await OversightService.list_turns(user=await reader(METRICS))
        # A first run leaves its own mark, chained as an access record too.
        await sync_to_async(prune)()
        old = [r.id async for r in AccessRecord.objects.all()]
        assert len(old) == 2
        await AccessRecord.objects.aupdate(created_at=timezone.now() - timedelta(days=800))
        counts = await sync_to_async(prune)()
        assert counts["access"] == 2
        assert not await AccessRecord.objects.filter(id__in=old).aexists()
        result = await verified()
        assert result.ok, result.problems
        assert result.pruned == 2
        # The turn and its content stay: only the access records' entries are covered.
        assert await Turn.objects.aexists()

    async def test_an_access_record_deleted_outside_retention_is_caught(self):
        from django_ai_sdk.oversight.models import AccessRecord
        from django_ai_sdk.oversight.services import OversightService

        await turn_in()
        await OversightService.list_turns(user=await reader(METRICS))
        await AccessRecord.objects.all().adelete()
        result = await verified()
        assert [(p["problem"], p["kind"]) for p in result.problems] == [
            ("record missing", "access")
        ]

    async def test_a_prune_over_access_records_does_not_cover_a_turn(self):
        from django.db import transaction

        from django_ai_sdk.oversight.models import AccessRecord
        from django_ai_sdk.oversight.services import OversightService

        turn_id = await turn_in()
        await OversightService.list_turns(user=await reader(METRICS))

        def prune_access_and_drop_a_match() -> None:
            ids = list(AccessRecord.objects.values_list("pk", flat=True))
            with transaction.atomic():
                chain.prune(access_ids=ids)
                AccessRecord.objects.filter(pk__in=ids).delete()
                RuleMatch.objects.filter(turn_id=turn_id).delete()

        await sync_to_async(prune_access_and_drop_a_match)()
        result = await verified()
        assert {(p["problem"], p["kind"]) for p in result.problems} == {("record missing", "match")}

    async def test_a_turn_deleted_outside_retention_is_caught(self):
        turn_id = await turn_in()
        await Turn.objects.filter(id=turn_id).adelete()
        problems = {p["problem"] for p in (await verified()).problems}
        # Its content stays, now pointed at by no turn.
        assert problems == {"record missing", "record altered"}


@pytest.mark.django_db(transaction=True)
class TestReadsAndSwitches:
    async def test_a_read_of_the_record_is_chained(self):
        from django_ai_sdk.oversight.services import OversightService

        await turn_in()
        before = await LedgerEntry.objects.acount()
        await OversightService.list_turns(user=await reader(METRICS))
        assert (await kinds())[before:] == ["access"]
        assert (await verified()).ok

    async def test_a_changed_reader_label_is_not_hashed(self):
        from django_ai_sdk.oversight.models import AccessRecord
        from django_ai_sdk.oversight.services import OversightService

        await turn_in()
        await OversightService.list_turns(user=await reader(METRICS))
        await AccessRecord.objects.aupdate(reader_label="42")
        assert (await verified()).ok

    @pytest.mark.skipif(
        not apps.is_installed("django_ai_sdk.oversight.switches"),
        reason="the switches app is not installed",
    )
    async def test_a_switch_and_its_access_record_are_chained(self, registry_snapshot):
        from django_ai_sdk.oversight.registry import register_policy
        from django_ai_sdk.oversight.switches.models import Switch
        from django_ai_sdk.oversight.switches.schemas import RuleModeIn
        from django_ai_sdk.oversight.switches.services import SwitchService
        from tests.oversight.switches.helpers import SWITCH, SWITCH_POLICIES, VIEW

        for policy in SWITCH_POLICIES:
            register_policy(policy)
        await SwitchService.set_rule_mode(
            "sw-parent", "block_x", RuleModeIn(mode="enforce"), user=await reader(SWITCH, VIEW)
        )
        assert await kinds() == ["switch", "access"]
        assert (await verified()).ok
        await Switch.objects.aupdate(value="off")
        assert not (await verified()).ok


def test_payloads_read_back_as_they_were_hashed():
    """A payload is hashed as JSON reads it back: a tuple and a list hash the same."""
    from types import SimpleNamespace

    row = SimpleNamespace(
        id="m1", turn_id="t1", rule="r", stage="gate", mode="enforce",
        actions=("block",), acted=True, condition={"b": 1, "a": 2},
    )  # fmt: skip
    listed = SimpleNamespace(**{**vars(row), "actions": ["block"], "condition": {"a": 2, "b": 1}})
    assert chain.payload_hash("match", row) == chain.payload_hash("match", listed)
    assert json.loads(json.dumps(chain.KINDS["match"].payload(listed)))["actions"] == ["block"]
