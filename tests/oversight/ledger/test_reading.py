"""Reading the record with the ledger installed: no table holds what was said, a forgotten
thread stays forgotten for every turn after it, content another master key kept reads as
unreadable rather than gone, and verify says when the head or a checkpoint disagrees."""

from __future__ import annotations

import json

import pytest
from asgiref.sync import sync_to_async
from django.db import connection

from django_ai_sdk.oversight import custody
from django_ai_sdk.oversight.content import load_content
from django_ai_sdk.oversight.ledger import chain
from django_ai_sdk.oversight.ledger.checkpoints import write_checkpoint
from django_ai_sdk.oversight.ledger.models import Checkpoint, LedgerEntry, LedgerHead
from django_ai_sdk.oversight.ledger.services import verify_all
from django_ai_sdk.oversight.models import ReviewItem, Turn
from django_ai_sdk.oversight.schemas import ResolveIn
from django_ai_sdk.oversight.services import OversightService
from tests.oversight.review.helpers import CONTENT, METRICS, REVIEW, a_turn, reader

QUESTION = "What about my salary?"
ANSWER = "Ask jan@example.com."
NOTE = "Jan Jansen asked twice, call 06-12345678"
SAID = ("salary", "jan@example.com", "Jan Jansen", "06-12345678", "asked twice")

# Another master key: what the test key kept does not open under it alone.
OTHER_KEY = "b3RoZXItdmF1bHQtbWFzdGVyLWtleS1mb3ItdGVzdHM="


def every_row() -> str:
    """Every row of every oversight, ledger and vault table, as one string."""
    tables = [
        t
        for t in connection.introspection.table_names()
        if t.startswith("django_ai_sdk_oversight") or t == "django_ai_sdk_data_keys"
    ]
    assert "django_ai_sdk_oversight_ledger" in tables
    dump = []
    with connection.cursor() as cursor:
        for table in tables:
            cursor.execute(f"SELECT * FROM {connection.ops.quote_name(table)}")  # noqa: S608
            dump.extend(cursor.fetchall())
    return json.dumps(dump, default=str)


async def leaks() -> list[str]:
    dump = await sync_to_async(every_row)()
    return [said for said in SAID if said in dump]


async def resolved_with_a_note(turn_id: str) -> ReviewItem:
    user = await reader(METRICS, CONTENT, REVIEW)
    item = await ReviewItem.objects.aget(turn_id=turn_id)
    await OversightService.resolve_review(
        str(item.id), ResolveIn(lens="topic", value="other", note=NOTE), user=user
    )
    return await ReviewItem.objects.aget(id=item.id)


async def detail_of(turn_id: str):
    return await OversightService.get_turn(turn_id, user=await reader(METRICS, CONTENT))


@pytest.mark.django_db(transaction=True)
class TestAtRest:
    async def test_no_row_holds_the_question_the_answer_or_a_reviewers_note(self):
        turn_id = await a_turn(QUESTION, ANSWER, thread_id="t-rest")
        item = await resolved_with_a_note(turn_id)
        assert "note" not in item.resolution
        detail = await detail_of(turn_id)
        assert detail.content_state == "full"
        assert detail.content["input"] == QUESTION
        assert detail.reviews[0].resolution["note"] == NOTE
        assert await leaks() == []
        assert (await sync_to_async(chain.verify)()).ok


@pytest.mark.django_db(transaction=True)
class TestAForgottenThread:
    @pytest.mark.parametrize("capture", ["full", "masked", "none"])
    async def test_the_next_turn_in_it_is_not_kept_in_plain_text(self, capture):
        before = await a_turn(QUESTION, ANSWER, thread_id="t-gone")
        await sync_to_async(custody.get().forget)("thread:t-gone")
        after = await a_turn(QUESTION, ANSWER, thread_id="t-gone", capture=capture)
        turn = await Turn.objects.select_related("content").aget(id=after)
        # Nothing new is kept for a forgotten scope: not encrypted, not in the clear.
        assert turn.content is None
        assert (await detail_of(after)).content_state == "none"
        assert (await detail_of(before)).content_state == "none"
        assert await leaks() == []
        result = await sync_to_async(chain.verify)()
        assert result.ok, result.problems
        assert result.forgotten == 1

    async def test_it_takes_its_reviewers_notes_and_keeps_the_verdict(self):
        turn_id = await a_turn(QUESTION, ANSWER, thread_id="t-notes")
        item = await resolved_with_a_note(turn_id)
        await sync_to_async(custody.get().forget)("thread:t-notes")
        detail = await detail_of(turn_id)
        assert detail.content is None
        [review] = detail.reviews
        assert "note" not in review.resolution
        assert review.resolution["value"] == "other"
        await item.arefresh_from_db()
        assert item.status == ReviewItem.Status.DONE
        assert await leaks() == []
        assert (await sync_to_async(chain.verify)()).ok

    async def test_another_thread_still_opens(self):
        kept = await a_turn(QUESTION, ANSWER, thread_id="t-kept")
        await a_turn(QUESTION, ANSWER, thread_id="t-gone")
        await sync_to_async(custody.get().forget)("thread:t-gone")
        turn = await Turn.objects.select_related("content").aget(id=kept)
        assert (await sync_to_async(load_content)(turn))["input"] == QUESTION


@pytest.mark.django_db(transaction=True)
class TestUnderAnotherMasterKey:
    """Content kept under a master key not configured here is there, unreadable: not gone."""

    async def test_a_reader_sees_it_as_unreadable_and_verify_says_so(self, settings):
        turn_id = await a_turn(QUESTION, ANSWER, thread_id="t-other")
        settings.AI_SDK_VAULT = {"MASTER_KEYS": [OTHER_KEY]}
        detail = await detail_of(turn_id)
        assert (detail.content, detail.content_state) == (None, "unreadable")
        page = await OversightService.list_turns(user=await reader(METRICS, CONTENT))
        assert page.items[0].preview is None
        out = await sync_to_async(verify_all)()
        assert (out.ok, out.state, out.unreadable) == (True, "unreadable", 1)

    async def test_a_turn_in_its_thread_still_keeps_content_once_the_key_is_back(self, settings):
        from django.conf import settings as live

        test_key = live.AI_SDK_VAULT["MASTER_KEYS"][0]
        first = await a_turn(QUESTION, ANSWER, thread_id="t-back")
        settings.AI_SDK_VAULT = {"MASTER_KEYS": [OTHER_KEY, test_key]}
        later = await a_turn("and my leave?", ANSWER, thread_id="t-back")
        for turn_id in (first, later):
            assert (await detail_of(turn_id)).content_state == "full"


@pytest.mark.django_db(transaction=True)
class TestTheHeadAndCheckpoints:
    def chained(self, n: int) -> None:
        from django.db import transaction

        from django_ai_sdk.oversight.models import AccessRecord

        for i in range(n):
            with transaction.atomic():
                record = AccessRecord.objects.create(operation="list", query={"n": i})
                chain.append_many([("access", record)])

    def test_a_head_that_disagrees_with_its_entry_is_caught(self):
        self.chained(2)
        LedgerHead.objects.filter(id=1).update(entry_hash="0" * 64)
        assert [p["problem"] for p in chain.verify().problems] == ["head mismatch"]

    def test_the_newest_entries_rolled_back_with_the_head_take_a_checkpoint_to_catch(self):
        self.chained(5)
        write_checkpoint()
        tip = LedgerEntry.objects.get(seq=3).entry_hash
        LedgerEntry.objects.filter(seq__gt=3).delete()
        LedgerHead.objects.filter(id=1).update(seq=3, entry_hash=tip)
        # The rows are access records nobody deleted, so walking the chain finds nothing:
        # only a checkpoint kept outside the database's reach can.
        assert chain.verify().ok
        out = verify_all()
        assert (out.ok, out.checkpoint_ok, out.state) == (False, False, "failed")

    def test_a_checkpoint_that_no_longer_matches_fails_verify(self):
        self.chained(2)
        write_checkpoint()
        assert verify_all().checkpoint_ok is True
        Checkpoint.objects.update(root_hash="0" * 64)
        out = verify_all()
        assert (out.ok, out.checkpoint_ok) == (False, False)
