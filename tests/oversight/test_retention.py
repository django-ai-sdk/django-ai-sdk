"""Retention: `oversight_prune` deletes what expired in batches, and W007 says when expired
rows are left."""

from __future__ import annotations

from datetime import timedelta
from io import StringIO

import pytest
from asgiref.sync import sync_to_async
from django.core.management import call_command
from django.db import connection
from django.test import override_settings
from django.test.utils import CaptureQueriesContext
from django.utils import timezone
from django_ai_sdk.oversight.checks import check_oversight_retention
from django_ai_sdk.oversight.models import (
    LensAnswer,
    ReviewItem,
    RuleMatch,
    Turn,
    TurnContent,
)
from django_ai_sdk.oversight.retention import prune

from tests.oversight.helpers import closed_turn, start

pytestmark = pytest.mark.django_db(transaction=True)

LATER = timedelta(days=1)


async def a_turn(text: str = "What about my salary?", output: str = "Ask jan@example.com.") -> str:
    """A closed test-guard turn: "salary" queues it for hr."""
    with override_settings(AI_SDK_OVERSIGHT={"CAPTURE": "full"}):
        guard = await start("test-guard", text)
        await guard.gate()
        await guard.finish(output)
        return str((await closed_turn(guard)).id)


async def expire(turn_id: str, *, content: bool = False, turn: bool = False) -> None:
    """Make `turn_id`'s content, or the turn itself, expired by now."""
    past = timezone.now() - LATER
    row = await Turn.objects.aget(id=turn_id)
    if content:
        await TurnContent.objects.filter(id=row.content_id).aupdate(expires_at=past)
    if turn:
        await Turn.objects.filter(id=turn_id).aupdate(expires_at=past)


async def test_expired_content_goes_and_its_turn_stays():
    gone, kept = await a_turn(), await a_turn("hello", "hi")
    await expire(gone, content=True)
    counts = await sync_to_async(prune)()
    assert counts == {"content": 1, "open_reviews": 0, "turns": 0}
    turn = await Turn.objects.aget(id=gone)
    assert turn.content_id is None
    assert await RuleMatch.objects.filter(turn=turn).aexists()
    assert (await Turn.objects.aget(id=kept)).content_id is not None


async def test_an_expired_turn_takes_its_rows_and_content():
    gone, kept = await a_turn(), await a_turn()
    content_id = (await Turn.objects.aget(id=gone)).content_id
    await expire(gone, turn=True)
    counts = await sync_to_async(prune)()
    assert counts["turns"] == 1
    # Its hr review was still open: retention wins, and the record says so.
    assert counts["open_reviews"] == 1
    assert not await Turn.objects.filter(id=gone).aexists()
    for model in (LensAnswer, RuleMatch, ReviewItem):
        assert not await model.objects.filter(turn_id=gone).aexists(), model
        assert await model.objects.filter(turn_id=kept).aexists(), model
    assert not await TurnContent.objects.filter(id=content_id).aexists()


async def test_it_deletes_in_batches_each_its_own_transaction():
    made = [await a_turn(f"salary {n}", "ok") for n in range(5)]
    for turn_id in made:
        await expire(turn_id, turn=True)
    counts, deletes = await sync_to_async(prune_counting_turn_deletes)(2)
    assert counts["turns"] == 5
    # Three batches of at most two turns: one delete of turn rows each.
    assert deletes == 3
    assert not await Turn.objects.aexists()


def prune_counting_turn_deletes(batch_size: int) -> tuple[dict[str, int], int]:
    """`prune(batch_size=...)`, and how many DELETEs of turn rows it sent."""
    with CaptureQueriesContext(connection) as queries:
        counts = prune(batch_size=batch_size)
    sql = [q["sql"] for q in queries.captured_queries]
    return counts, sum(1 for q in sql if q.startswith("DELETE") and "oversight_turns" in q)


async def test_expired_content_takes_reviewers_notes_with_it():
    turn_id = await a_turn()
    item = await ReviewItem.objects.aget(turn_id=turn_id)
    await ReviewItem.objects.filter(id=item.id).aupdate(note="about Jan")
    await expire(turn_id, content=True)
    await sync_to_async(prune)()
    await item.arefresh_from_db()
    assert item.note == ""


async def test_the_command_says_what_it_deleted():
    turn_id = await a_turn()
    await expire(turn_id, turn=True)
    out = StringIO()
    await sync_to_async(call_command)("oversight_prune", "--batch-size", "1", stdout=out)
    assert out.getvalue().strip() == (
        "Deleted 0 content rows and 1 turns.\n1 reviews were still open on deleted turns."
    )


class TestNothingPrunes:
    """W007: a deploy check against a database; fresh deployments and recent runs are fine."""

    @staticmethod
    def ids() -> list[str]:
        return [m.id for m in check_oversight_retention(databases=["default"])]

    async def test_rows_expired_two_days_ago_warn_until_pruned(self):
        turn_id = await a_turn()
        assert await sync_to_async(self.ids)() == []
        await Turn.objects.filter(id=turn_id).aupdate(expires_at=timezone.now() - timedelta(days=3))
        assert await sync_to_async(self.ids)() == ["ai_sdk.oversight.W007"]
        await sync_to_async(prune)()
        assert await sync_to_async(self.ids)() == []

    async def test_rows_expired_recently_are_fine(self):
        turn_id = await a_turn()
        await Turn.objects.filter(id=turn_id).aupdate(
            expires_at=timezone.now() - timedelta(hours=1)
        )
        assert await sync_to_async(self.ids)() == []

    async def test_a_turn_open_for_more_than_a_day_warns_until_swept(self):
        # W017: a run that crashed outside run_agent leaves its turn open until the sweep.
        turn_id = await a_turn()
        await Turn.objects.filter(id=turn_id).aupdate(
            status="open", created_at=timezone.now() - timedelta(days=2)
        )
        assert await sync_to_async(self.ids)() == ["ai_sdk.oversight.W017"]
        await Turn.objects.filter(id=turn_id).aupdate(status="interrupted")
        assert await sync_to_async(self.ids)() == []

    def test_it_reads_no_database_unless_given_one(self):
        assert check_oversight_retention() == []

    def test_it_is_a_deploy_check_on_the_database(self):
        from django.core.checks import Tags
        from django.core.checks.registry import registry

        assert check_oversight_retention in registry.get_checks(include_deployment_checks=True)
        assert check_oversight_retention not in registry.get_checks()
        assert Tags.database in check_oversight_retention.tags
