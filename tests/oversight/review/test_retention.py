"""Retention: `oversight_prune` deletes what expired in batches, marks that it ran, and W007
says when nothing has."""

from __future__ import annotations

from datetime import timedelta
from io import StringIO
from typing import Any

import pytest
from asgiref.sync import sync_to_async
from django.core.management import call_command
from django.test import override_settings
from django.utils import timezone

from django_ai_sdk.oversight import custody
from django_ai_sdk.oversight.checks import check_oversight_retention
from django_ai_sdk.oversight.models import (
    AccessRecord,
    LensAnswer,
    ReviewItem,
    RuleMatch,
    Turn,
    TurnContent,
)
from django_ai_sdk.oversight.retention import prune
from tests.oversight.review.helpers import a_turn

pytestmark = pytest.mark.django_db(transaction=True)

LATER = timedelta(days=1)


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
    assert counts == {"content": 1, "turns": 0, "access": 0}
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
    assert not await Turn.objects.filter(id=gone).aexists()
    for model in (LensAnswer, RuleMatch, ReviewItem):
        assert not await model.objects.filter(turn_id=gone).aexists(), model
        assert await model.objects.filter(turn_id=kept).aexists(), model
    assert not await TurnContent.objects.filter(id=content_id).aexists()


async def test_it_deletes_in_batches_each_its_own_transaction():
    made = [await a_turn(f"salary {n}", "ok") for n in range(5)]
    for turn_id in made:
        await expire(turn_id, turn=True)
    handed: list[list[Any]] = []

    class Counting(custody.PlainCustody):
        def pruning(self, turn_ids: Any) -> None:
            handed.append(list(turn_ids))

    custody.install(Counting())
    try:
        counts = await sync_to_async(prune)(batch_size=2)
    finally:
        custody.install(None)
    assert counts["turns"] == 5
    # Custody hears of every turn before it goes, a batch at a time.
    assert [len(batch) for batch in handed] == [2, 2, 1]
    assert sorted(str(t) for batch in handed for t in batch) == sorted(made)
    assert not await Turn.objects.aexists()


async def test_a_failed_batch_keeps_the_batches_before_it():
    made = [await a_turn(f"salary {n}", "ok") for n in range(3)]
    for turn_id in made:
        await expire(turn_id, turn=True)
    calls: list[int] = []

    class FailsSecond(custody.PlainCustody):
        def pruning(self, turn_ids: Any) -> None:
            calls.append(len(list(turn_ids)))
            if len(calls) == 2:
                raise RuntimeError("chain down")

    custody.install(FailsSecond())
    try:
        with pytest.raises(RuntimeError):
            await sync_to_async(prune)(batch_size=2)
    finally:
        custody.install(None)
    # The first batch committed on its own; the second rolled back whole.
    assert await Turn.objects.acount() == 1


async def test_a_custody_that_refuses_keeps_the_batch():
    turn_id = await a_turn()
    await expire(turn_id, turn=True)

    class Refusing(custody.PlainCustody):
        def pruning(self, turn_ids: Any) -> None:
            raise RuntimeError("chain down")

    custody.install(Refusing())
    try:
        with pytest.raises(RuntimeError):
            await sync_to_async(prune)()
    finally:
        custody.install(None)
    assert await Turn.objects.filter(id=turn_id).aexists()


async def test_old_access_records_go_with_the_metadata_retention():
    old = await AccessRecord.objects.acreate(operation="list")
    new = await AccessRecord.objects.acreate(operation="list")
    await AccessRecord.objects.filter(id=old.id).aupdate(
        created_at=timezone.now() - timedelta(days=31)
    )
    with override_settings(AI_SDK_OVERSIGHT={"RETENTION": {"metadata": 30}}):
        counts = await sync_to_async(prune)()
    assert counts["access"] == 1
    assert not await AccessRecord.objects.filter(id=old.id).aexists()
    assert await AccessRecord.objects.filter(id=new.id).aexists()


async def test_a_bad_metadata_retention_stops_the_prune():
    record = await AccessRecord.objects.acreate(operation="list")
    with (
        override_settings(AI_SDK_OVERSIGHT={"RETENTION": {"metadata": 0}}),
        pytest.raises(ValueError, match="at least 1"),
    ):
        await sync_to_async(prune)()
    assert await AccessRecord.objects.filter(id=record.id).aexists()


async def test_a_run_leaves_its_mark():
    assert await sync_to_async(AccessRecord.last_pruned)() is None
    await sync_to_async(prune)()
    mark = await AccessRecord.objects.aget(operation="prune")
    assert mark.reader_label == "oversight_prune"
    assert mark.turn_ids == []
    assert mark.query["turns"] == 0
    assert await sync_to_async(AccessRecord.last_pruned)() == mark.created_at


def test_the_admin_leaves_out_who_asked_and_lists_every_access():
    from django.test import modify_settings

    with modify_settings(INSTALLED_APPS={"append": "django.contrib.admin"}):
        from django_ai_sdk.oversight.admin import AccessRecordAdmin, TurnAdmin

    # Metrics access is enough for the admin, so it leaves out who asked and where.
    for field in ("content", "actor_id", "thread_id", "user_message_id", "message_id"):
        assert field in TurnAdmin.exclude
    # Reads and changes alike; the operation filter narrows them.
    assert AccessRecordAdmin.list_filter == ("operation", "content")


async def test_the_command_says_what_it_deleted():
    turn_id = await a_turn()
    await expire(turn_id, turn=True)
    out = StringIO()
    await sync_to_async(call_command)("oversight_prune", "--batch-size", "1", stdout=out)
    assert out.getvalue().strip() == "Deleted 0 content rows, 1 turns and 0 access records."


class TestNothingPrunes:
    """W007: a deploy check against a database; fresh deployments and recent runs are fine."""

    @staticmethod
    def ids() -> list[str]:
        return [m.id for m in check_oversight_retention(databases=["default"])]

    async def test_turns_older_than_two_days_and_no_run_warn(self):
        turn_id = await a_turn()
        assert await sync_to_async(self.ids)() == []
        await Turn.objects.filter(id=turn_id).aupdate(created_at=timezone.now() - timedelta(days=3))
        assert await sync_to_async(self.ids)() == ["ai_sdk.oversight.W007"]
        await sync_to_async(prune)()
        assert await sync_to_async(self.ids)() == []

    async def test_a_run_long_ago_warns(self):
        turn_id = await a_turn()
        await Turn.objects.filter(id=turn_id).aupdate(created_at=timezone.now() - timedelta(days=9))
        await sync_to_async(prune)()
        await AccessRecord.objects.filter(operation="prune").aupdate(
            created_at=timezone.now() - timedelta(days=3)
        )
        [warning] = await sync_to_async(check_oversight_retention)(databases=["default"])
        assert "last ran" in warning.msg

    def test_it_reads_no_database_unless_given_one(self):
        assert check_oversight_retention() == []

    def test_it_is_a_deploy_check_on_the_database(self):
        from django.core.checks import Tags
        from django.core.checks.registry import registry

        assert check_oversight_retention in registry.get_checks(include_deployment_checks=True)
        assert check_oversight_retention not in registry.get_checks()
        assert Tags.database in check_oversight_retention.tags
