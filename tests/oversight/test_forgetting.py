"""Forgetting: deleting a thread deletes its oversight content; deleting a user, who asked too."""

from __future__ import annotations

import threading
import time
from datetime import timedelta
from io import StringIO
from unittest.mock import patch

import pytest
from asgiref.sync import sync_to_async
from django.db import transaction
from django.utils import timezone
from django_ai_sdk.oversight import content
from django_ai_sdk.oversight.guard import TurnGuard
from django_ai_sdk.oversight.models import LensAnswer, ReviewItem, RuleMatch, Turn, TurnContent

from tests.factories.db import UserFactory
from tests.oversight.helpers import (
    a_thread,
    closed_turn,
    make_agent,
    race,
    start,
)

QUESTION = "my salary is low, mail jan@example.com about it"
ANSWER = "Your salary is fine."
OVERSIGHT_TABLES = [
    "django_ai_sdk_oversight_turns",
    "django_ai_sdk_oversight_content",
    "django_ai_sdk_oversight_answers",
    "django_ai_sdk_oversight_matches",
    "django_ai_sdk_oversight_reviews",
]


async def a_turn_in(thread, text: str = QUESTION) -> str:
    """A closed turn in `thread`, recorded with full capture so its content quotes it."""
    from django.test import override_settings

    with override_settings(AI_SDK_OVERSIGHT={"CAPTURE": "full"}):
        guard = await TurnGuard.start(
            make_agent([], None), channel="chat", input=text, thread_id=str(thread.id)
        )
        await guard.gate()
        await guard.finish(ANSWER)
        return str(await guard.close())


async def a_threadless_turn(user=None, text: str = QUESTION) -> str:
    """A closed turn `user` asked without a thread, with full capture."""
    from django.test import override_settings

    with override_settings(AI_SDK_OVERSIGHT={"CAPTURE": "full"}):
        guard = await TurnGuard.start(make_agent([], None), channel="api", input=text, user=user)
        await guard.gate()
        await guard.finish(ANSWER)
        return str(await guard.close())


async def has_content(turn_id: str) -> bool:
    return await TurnContent.objects.filter(turn__id=turn_id).aexists()


def dump() -> str:
    """Every row of every oversight table, read with raw SQL."""
    from django.db import connection

    rows = []
    with connection.cursor() as cursor:
        for table in OVERSIGHT_TABLES:
            cursor.execute(f"SELECT * FROM {table}")  # noqa: S608 - fixed table names
            rows.extend(repr(row) for row in cursor.fetchall())
    return "\n".join(rows)


def forget_turns(turns) -> None:
    with transaction.atomic():
        content.forget_turns(turns)


@pytest.mark.usefixtures("no_reflect_queue")
@pytest.mark.django_db(transaction=True)
class TestForgetTurns:
    async def test_it_deletes_the_content_and_keeps_the_turn(self):
        thread = await a_thread()
        turn_id = await a_turn_in(thread)
        assert await has_content(turn_id)
        await sync_to_async(forget_turns)(Turn.objects.filter(thread_id=str(thread.id)))
        assert not await TurnContent.objects.aexists()
        turn = await Turn.objects.aget(id=turn_id)
        assert turn.content_id is None
        assert not turn.content_forgotten  # it was closed already
        assert await RuleMatch.objects.filter(turn=turn).aexists()

    async def test_it_blanks_reviewers_notes(self):
        turn_id = await a_threadless_turn()
        item = await ReviewItem.objects.acreate(turn_id=turn_id, queue="q", note="Jan said so")
        await sync_to_async(forget_turns)(Turn.objects.filter(id=turn_id))
        await item.arefresh_from_db()
        assert item.note == ""

    async def test_a_turn_forgotten_while_open_closes_without_content(self):
        guard = await start("test-guard", QUESTION)
        await guard.gate()
        await sync_to_async(forget_turns)(Turn.objects.filter(id=guard.id))
        await guard.finish(ANSWER)
        turn = await closed_turn(guard)
        assert (turn.content_id, turn.content_forgotten) == (None, True)
        assert await LensAnswer.objects.filter(turn=turn).aexists()
        assert not await TurnContent.objects.aexists()

    async def test_another_turn_keeps_its_content(self):
        gone, kept = await a_threadless_turn(), await a_threadless_turn()
        await sync_to_async(forget_turns)(Turn.objects.filter(id=gone))
        assert not await has_content(gone)
        assert await has_content(kept)


@pytest.mark.usefixtures("no_reflect_queue")
@pytest.mark.django_db(transaction=True)
class TestDeletedThreads:
    async def test_a_deleted_thread_takes_its_oversight_content(self):
        from django_ai_sdk.storage.services import ThreadService

        user = await UserFactory.acreate()
        thread = await a_thread(user)
        turn_id = await a_turn_in(thread)
        assert await has_content(turn_id)
        assert await ThreadService.delete_thread(str(thread.id), user=user)
        assert not await TurnContent.objects.aexists()
        assert await Turn.objects.filter(thread_id=str(thread.id)).aexists()  # matches stay

    async def test_the_command_forgets_a_thread(self):
        from django.core.management import call_command

        thread = await a_thread()
        turn_id = await a_turn_in(thread)
        out = StringIO()
        await sync_to_async(call_command)("oversight_forget_thread", str(thread.id), stdout=out)
        assert f"Forgot thread {thread.id}." in out.getvalue()
        assert not await has_content(turn_id)

    async def test_a_turn_open_while_its_thread_is_deleted_stores_no_content(self):
        from django.test import override_settings

        thread = await a_thread()
        with override_settings(AI_SDK_OVERSIGHT={"CAPTURE": "full"}):
            guard = await TurnGuard.start(
                make_agent([], None), channel="chat", input=QUESTION, thread_id=str(thread.id)
            )
            await guard.gate()
            await thread.adelete()
            await guard.finish(ANSWER)
            turn_id = await guard.close()
        assert not await has_content(turn_id)

    async def test_a_thread_oversight_never_saw_forgets_nothing(self):
        thread = await a_thread()
        with patch("django_ai_sdk.oversight.content.forget_thread") as forget:
            await thread.adelete()
        forget.assert_not_called()


@pytest.mark.usefixtures("no_reflect_queue")
@pytest.mark.django_db(transaction=True)
class TestNothingQuotesAForgottenThread:
    async def test_the_raw_tables_hold_no_text_of_the_turn(self):
        thread = await a_thread()
        thread_id = str(thread.id)
        await a_turn_in(thread)
        before = await sync_to_async(dump)()
        assert "salary" in before and "jan@example.com" in before  # full capture kept it
        await thread.adelete()
        after = await sync_to_async(dump)()
        for text in ("salary", "jan@example.com", "fine", "found none"):
            assert text not in after
        assert await Turn.objects.filter(thread_id=thread_id).aexists()

    async def test_a_bulk_delete_forgets_every_thread(self):
        from django_ai_sdk.conversation.models import Thread

        first, second = await a_thread(), await a_thread()
        await a_turn_in(first)
        await a_turn_in(second)
        await Thread.objects.filter(id__in=[first.id, second.id]).adelete()
        assert await TurnContent.objects.acount() == 0
        assert await Turn.objects.acount() == 2

    async def test_deleting_a_user_forgets_their_threads(self):
        user = await UserFactory.acreate()
        thread = await a_thread(user)
        turn_id = await a_turn_in(thread)
        await user.adelete()
        assert not await has_content(turn_id)

    async def test_a_failed_forget_does_not_stop_the_delete(self):
        from django_ai_sdk.conversation.models import Thread

        thread = await a_thread()
        thread_id = thread.id
        await a_turn_in(thread)
        with (
            patch(
                "django_ai_sdk.oversight.content.forget_thread",
                side_effect=RuntimeError("db down"),
            ),
            patch("django_ai_sdk.oversight.signals.logger") as logger,
        ):
            await thread.adelete()
        assert not await Thread.objects.filter(id=thread_id).aexists()
        logger.exception.assert_called_once()

    async def test_a_turn_without_a_thread_is_not_forgotten_by_another_thread(self):
        turn_id = await a_threadless_turn()
        thread = await a_thread()
        await a_turn_in(thread)
        await thread.adelete()
        assert await has_content(turn_id)


@pytest.mark.usefixtures("no_reflect_queue")
@pytest.mark.django_db(transaction=True)
class TestDeletedUsers:
    """Deleting a user clears who asked and forgets the content of their threadless turns."""

    async def test_a_deleted_user_leaves_no_actor_and_no_threadless_content(self):
        user = await UserFactory.acreate()
        thread = await a_thread(user)
        in_thread = await a_turn_in(thread)
        await Turn.objects.filter(id=in_thread).aupdate(actor_id=str(user.pk))
        threadless = await a_threadless_turn(user)
        answers = await LensAnswer.objects.acount()
        matches = await RuleMatch.objects.acount()
        assert await Turn.objects.filter(actor_id=str(user.pk)).acount() == 2
        pk = str(user.pk)
        await user.adelete()
        assert not await Turn.objects.filter(actor_id=pk).aexists()
        assert await Turn.objects.acount() == 2
        assert not await has_content(threadless)
        assert not await has_content(in_thread)
        # What the record says about the turns stays.
        assert await LensAnswer.objects.acount() == answers
        assert await RuleMatch.objects.acount() == matches
        after = await sync_to_async(dump)()
        for text in ("salary", "jan@example.com"):
            assert text not in after

    async def test_another_users_turns_are_untouched(self):
        leaving, staying = await UserFactory.acreate(), await UserFactory.acreate()
        await a_threadless_turn(leaving)
        kept = await a_threadless_turn(staying)
        await leaving.adelete()
        turn = await Turn.objects.aget(id=kept)
        assert turn.actor_id == str(staying.pk)
        assert await has_content(kept)

    async def test_a_turn_open_while_its_user_is_deleted_does_not_bring_them_back(self):
        from django.test import override_settings

        user = await UserFactory.acreate()
        with override_settings(AI_SDK_OVERSIGHT={"CAPTURE": "full"}):
            guard = await TurnGuard.start(
                make_agent([], None), channel="api", input=QUESTION, user=user
            )
            await guard.gate()
            await user.adelete()
            await guard.finish(ANSWER)
            turn_id = str(await guard.close())
        assert (await Turn.objects.aget(id=turn_id)).actor_id == ""
        assert not await has_content(turn_id)

    async def test_a_bulk_delete_forgets_every_user(self):
        from django.contrib.auth import get_user_model

        first, second = await UserFactory.acreate(), await UserFactory.acreate()
        await a_threadless_turn(first)
        await a_threadless_turn(second)
        await get_user_model().objects.filter(pk__in=[first.pk, second.pk]).adelete()
        assert not await Turn.objects.exclude(actor_id="").aexists()

    async def test_a_failed_forget_does_not_stop_the_user_delete(self):
        from django.contrib.auth import get_user_model

        user = await UserFactory.acreate()
        pk = user.pk
        turn_id = await a_threadless_turn(user)
        with (
            patch("django_ai_sdk.oversight.signals.batched", side_effect=RuntimeError("db down")),
            patch("django_ai_sdk.oversight.signals.logger") as logger,
        ):
            await user.adelete()
        assert not await get_user_model().objects.filter(pk=pk).aexists()
        logger.exception.assert_called_once()
        assert "oversight_forget_actor" in logger.exception.call_args.args[0]
        # All or nothing: the actor stays named, so a retry (forget_actor) finds the turn.
        assert (await Turn.objects.aget(id=turn_id)).actor_id == str(pk)

    async def test_the_command_forgets_a_user_whose_delete_could_not(self):
        from django.core.management import call_command

        user = await UserFactory.acreate()
        pk = str(user.pk)
        turn_id = await a_threadless_turn(user)
        with (
            patch("django_ai_sdk.oversight.signals.batched", side_effect=RuntimeError("db down")),
            patch("django_ai_sdk.oversight.signals.logger"),
        ):
            await user.adelete()
        assert (await Turn.objects.aget(id=turn_id)).actor_id == pk

        out = StringIO()
        await sync_to_async(call_command)("oversight_forget_actor", pk, stdout=out)

        assert f"Forgot user {pk} on 1 turns." in out.getvalue()
        assert (await Turn.objects.aget(id=turn_id)).actor_id == ""
        assert not await has_content(turn_id)

    async def test_threadless_content_is_forgotten_a_batch_at_a_time(self):
        user = await UserFactory.acreate()
        turns = [await a_threadless_turn(user) for _ in range(5)]
        with patch("django_ai_sdk.oversight.signals.FORGET_BATCH", 2):
            forgotten, deletes = await sync_to_async(forget_counting_deletes)(str(user.pk))
        assert forgotten == 5
        assert not await TurnContent.objects.filter(turn__id__in=turns).aexists()
        # Three batches of at most two turns: one delete of content rows each, not five.
        assert deletes == 3

    async def test_a_deleted_user_blanks_notes_on_their_threadless_turns(self):
        user = await UserFactory.acreate()
        turn_id = await a_threadless_turn(user)
        item = await ReviewItem.objects.acreate(turn_id=turn_id, queue="q", note="about Jan")
        await user.adelete()
        await item.arefresh_from_db()
        assert item.note == ""


def forget_counting_deletes(actor_id: str) -> tuple[int, int]:
    """`forget_actor(actor_id)`, and how many DELETEs of content rows it sent."""
    from django.db import connection
    from django.test.utils import CaptureQueriesContext
    from django_ai_sdk.oversight.signals import forget_actor

    with CaptureQueriesContext(connection) as queries, transaction.atomic():
        forgotten = forget_actor(actor_id)
    deletes = [
        q["sql"]
        for q in queries.captured_queries
        if q["sql"].startswith("DELETE") and "django_ai_sdk_oversight_content" in q["sql"]
    ]
    return forgotten, len(deletes)


@pytest.mark.usefixtures("no_reflect_queue")
@pytest.mark.django_db(transaction=True)
class TestExpiredContent:
    async def test_it_reads_back_as_none(self):
        turn_id = await a_threadless_turn()
        turn = await Turn.objects.select_related("content").aget(id=turn_id)
        assert content.load_content(turn) is not None
        turn.content.expires_at = timezone.now() - timedelta(seconds=1)
        assert content.load_content(turn) is None

    async def test_nothing_is_added_to_it(self):
        turn_id = await a_threadless_turn()
        turn = await Turn.objects.aget(id=turn_id)
        await TurnContent.objects.filter(id=turn.content_id).aupdate(
            expires_at=timezone.now() - timedelta(seconds=1)
        )
        added = await sync_to_async(content.add_to_content)(turn, texts={"x": {"rationale": "y"}})
        assert added is False
        row = await TurnContent.objects.aget(id=turn.content_id)
        assert "x" not in (row.stored.get("texts") or {})


@pytest.mark.django_db(transaction=True)
@pytest.mark.usefixtures("postgres_only", "no_reflect_queue")
class TestForgettingRacesAClose:
    """A forget and a close on separate connections; SQLite cannot show it (`DATABASE_URL`)."""

    def test_a_close_during_a_forget_stores_no_content(self):
        from asgiref.sync import async_to_sync

        async def opened() -> TurnGuard:
            guard = await start("test-guard", QUESTION, thread_id="t-race")
            await guard.gate()
            await guard.finish(ANSWER)
            return guard

        guard = async_to_sync(opened)()
        forgot = threading.Event()

        def forget() -> None:
            with transaction.atomic():
                content.forget_thread("t-race")
                forgot.set()
                # Hold the forget open while the close runs.
                time.sleep(0.5)

        def close() -> None:
            forgot.wait(5)
            async_to_sync(closed_turn)(guard)

        race(forget, close)
        turn = Turn.objects.get(id=guard.id)
        assert (turn.status, turn.content_id, turn.content_forgotten) == ("closed", None, True)
        assert not TurnContent.objects.exists()
