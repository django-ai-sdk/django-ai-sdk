"""Forgetting: deleting a thread deletes its oversight content; deleting a user, who asked too."""

from __future__ import annotations

import contextlib
from io import StringIO
from unittest.mock import patch

import pytest

from django_ai_sdk.oversight.guard import TurnGuard
from django_ai_sdk.oversight.models import Turn
from tests.factories.db import UserFactory
from tests.oversight.helpers import (
    a_thread,
    make_agent,
)


@pytest.mark.usefixtures("no_reflect_queue")
@pytest.mark.django_db(transaction=True)
class TestDeletedThreads:
    async def test_a_deleted_thread_takes_its_oversight_content(self):
        from django_ai_sdk.oversight.models import TurnContent
        from django_ai_sdk.storage.services import ThreadService

        user = await UserFactory.acreate()
        thread = await a_thread(user)
        guard = await TurnGuard.start(
            make_agent([], None), channel="chat", input="my salary?", thread_id=str(thread.id)
        )
        await guard.gate()
        await guard.finish("It is fine.")
        await guard.close()
        scope = f"thread:{thread.id}"
        assert await TurnContent.objects.filter(scope=scope, stored__isnull=False).aexists()
        assert await ThreadService.delete_thread(str(thread.id), user=user)
        assert not await TurnContent.objects.filter(scope=scope).aexists()
        assert await Turn.objects.filter(thread_id=str(thread.id)).aexists()  # matches stay

    async def test_the_command_forgets_a_thread_under_plain_custody(self):
        from asgiref.sync import sync_to_async
        from django.core.management import call_command

        from django_ai_sdk.oversight.models import TurnContent

        thread = await a_thread()
        await a_turn_in(thread)
        out = StringIO()
        await sync_to_async(call_command)("oversight_forget_thread", str(thread.id), stdout=out)
        assert f"Forgot thread:{thread.id}." in out.getvalue()
        assert not await TurnContent.objects.filter(scope=f"thread:{thread.id}").aexists()

    async def test_a_thread_oversight_never_saw_asks_custody_nothing(self):
        thread = await a_thread()
        with patch("django_ai_sdk.oversight.custody.PlainCustody.forget") as forget:
            await thread.adelete()
        forget.assert_not_called()


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


def dump() -> str:
    """Every row of every oversight table, read with raw SQL."""
    from django.db import connection

    rows = []
    with connection.cursor() as cursor:
        for table in OVERSIGHT_TABLES:
            cursor.execute(f"SELECT * FROM {table}")  # noqa: S608 - fixed table names
            rows.extend(repr(row) for row in cursor.fetchall())
    return "\n".join(rows)


@pytest.mark.usefixtures("no_reflect_queue")
@pytest.mark.django_db(transaction=True)
class TestNothingQuotesAForgottenThread:
    async def test_the_raw_tables_hold_no_text_of_the_turn(self):
        from asgiref.sync import sync_to_async

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
        from django_ai_sdk.oversight.models import TurnContent

        first, second = await a_thread(), await a_thread()
        await a_turn_in(first)
        await a_turn_in(second)
        await Thread.objects.filter(id__in=[first.id, second.id]).adelete()
        assert await TurnContent.objects.acount() == 0
        assert await Turn.objects.acount() == 2

    async def test_deleting_a_user_forgets_their_threads(self):
        from django_ai_sdk.oversight.models import TurnContent

        user = await UserFactory.acreate()
        thread = await a_thread(user)
        scope = f"thread:{thread.id}"
        await a_turn_in(thread)
        await user.adelete()
        assert not await TurnContent.objects.filter(scope=scope).aexists()

    async def test_a_custody_that_cannot_forget_does_not_stop_the_delete(self):
        from django_ai_sdk.conversation.models import Thread

        thread = await a_thread()
        thread_id = thread.id
        await a_turn_in(thread)
        with (
            patch(
                "django_ai_sdk.oversight.custody.PlainCustody.forget",
                side_effect=RuntimeError("vault down"),
            ),
            patch("django_ai_sdk.oversight.signals.logger") as logger,
        ):
            await thread.adelete()
        assert not await Thread.objects.filter(id=thread_id).aexists()
        logger.exception.assert_called_once()

    async def test_a_turn_without_a_thread_is_not_forgotten_by_another_thread(self):
        from django_ai_sdk.oversight.models import TurnContent

        guard = await TurnGuard.start(make_agent([], None), channel="api", input=QUESTION)
        await guard.gate()
        await guard.finish(ANSWER)
        turn_id = await guard.close()
        thread = await a_thread()
        await a_turn_in(thread)
        await thread.adelete()
        assert await TurnContent.objects.filter(scope=f"turn:{turn_id}").aexists()


async def a_threadless_turn(user, text: str = QUESTION) -> str:
    """A closed turn `user` asked without a thread, with full capture."""
    from django.test import override_settings

    with override_settings(AI_SDK_OVERSIGHT={"CAPTURE": "full"}):
        guard = await TurnGuard.start(make_agent([], None), channel="api", input=text, user=user)
        await guard.gate()
        await guard.finish(ANSWER)
        return str(await guard.close())


@pytest.mark.usefixtures("no_reflect_queue")
@pytest.mark.django_db(transaction=True)
class TestDeletedUsers:
    """D10: deleting a user clears who asked and forgets the content of their threadless turns."""

    async def test_a_deleted_user_leaves_no_actor_and_no_threadless_content(self):
        from asgiref.sync import sync_to_async

        from django_ai_sdk.oversight.models import LensAnswer, RuleMatch, TurnContent

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
        assert not await TurnContent.objects.filter(scope=f"turn:{threadless}").aexists()
        assert not await TurnContent.objects.filter(scope=f"thread:{thread.id}").aexists()
        # What the record says about the turns stays.
        assert await LensAnswer.objects.acount() == answers
        assert await RuleMatch.objects.acount() == matches
        after = await sync_to_async(dump)()
        for text in ("salary", "jan@example.com"):
            assert text not in after

    async def test_another_users_turns_are_untouched(self):
        from django_ai_sdk.oversight.models import TurnContent

        leaving, staying = await UserFactory.acreate(), await UserFactory.acreate()
        await a_threadless_turn(leaving)
        kept = await a_threadless_turn(staying)
        await leaving.adelete()
        turn = await Turn.objects.aget(id=kept)
        assert turn.actor_id == str(staying.pk)
        assert await TurnContent.objects.filter(scope=f"turn:{kept}").aexists()

    async def test_a_bulk_delete_forgets_every_user(self):
        from django.contrib.auth import get_user_model

        first, second = await UserFactory.acreate(), await UserFactory.acreate()
        await a_threadless_turn(first)
        await a_threadless_turn(second)
        await get_user_model().objects.filter(pk__in=[first.pk, second.pk]).adelete()
        assert not await Turn.objects.exclude(actor_id="").aexists()

    async def test_a_custody_that_cannot_forget_does_not_stop_the_user_delete(self):
        from django.contrib.auth import get_user_model

        user = await UserFactory.acreate()
        pk = user.pk
        turn_id = await a_threadless_turn(user)
        with (
            installed(Unreachable()),
            patch("django_ai_sdk.oversight.signals.logger") as logger,
        ):
            await user.adelete()
        assert not await get_user_model().objects.filter(pk=pk).aexists()
        logger.exception.assert_called_once()
        assert "oversight_forget_actor" in logger.exception.call_args.args[0]
        # All or nothing: the actor stays named, so a retry (forget_actor) finds the turn.
        assert (await Turn.objects.aget(id=turn_id)).actor_id == str(pk)

    async def test_the_command_forgets_a_user_whose_delete_could_not(self):
        from asgiref.sync import sync_to_async
        from django.core.management import call_command

        from django_ai_sdk.oversight.models import TurnContent

        user = await UserFactory.acreate()
        pk = str(user.pk)
        turn_id = await a_threadless_turn(user)
        with installed(Unreachable()), patch("django_ai_sdk.oversight.signals.logger"):
            await user.adelete()
        assert (await Turn.objects.aget(id=turn_id)).actor_id == pk

        out = StringIO()
        await sync_to_async(call_command)("oversight_forget_actor", pk, stdout=out)

        assert f"Forgot user {pk} on 1 turns." in out.getvalue()
        assert (await Turn.objects.aget(id=turn_id)).actor_id == ""
        assert not await TurnContent.objects.filter(scope=f"turn:{turn_id}").aexists()

    async def test_plain_custody_forgets_threadless_content_a_batch_at_a_time(self):
        from asgiref.sync import sync_to_async

        from django_ai_sdk.oversight.models import TurnContent

        user = await UserFactory.acreate()
        turns = [await a_threadless_turn(user) for _ in range(5)]
        with patch("django_ai_sdk.oversight.signals.FORGET_BATCH", 2):
            forgotten, deletes = await sync_to_async(forget_counting_deletes)(str(user.pk))
        assert forgotten == 5
        scopes = [f"turn:{t}" for t in turns]
        assert not await TurnContent.objects.filter(scope__in=scopes).aexists()
        # Three batches of at most two turns: one delete of content rows each, not five.
        assert deletes == 3

    async def test_another_custody_is_handed_each_threadless_scope(self):
        user = await UserFactory.acreate()
        turns = [await a_threadless_turn(user) for _ in range(3)]
        with installed(Remembering()) as keeper:
            await user.adelete()
        assert sorted(keeper.forgot) == sorted(f"turn:{t}" for t in turns)

    async def test_a_deleted_reader_keeps_their_id_on_access_records_but_not_their_name(self):
        from django_ai_sdk.oversight.models import AccessRecord

        reader = await UserFactory.acreate()
        pk = str(reader.pk)
        await AccessRecord.objects.acreate(
            reader_id=pk, reader_label=f"{reader.get_username()} ({pk})", operation="view"
        )
        await reader.adelete()
        record = await AccessRecord.objects.aget(reader_id=pk)
        assert record.reader_label == pk


def forget_counting_deletes(actor_id: str) -> tuple[int, int]:
    """`forget_actor(actor_id)`, and how many DELETEs of content rows it sent."""
    from django.db import connection
    from django.test.utils import CaptureQueriesContext

    from django_ai_sdk.oversight.signals import forget_actor

    with CaptureQueriesContext(connection) as queries:
        forgotten = forget_actor(actor_id)
    deletes = [
        q["sql"]
        for q in queries.captured_queries
        if q["sql"].startswith("DELETE") and "django_ai_sdk_oversight_content" in q["sql"]
    ]
    return forgotten, len(deletes)


class Remembering:
    """A custody other than the plain one: it remembers the scopes it was told to forget."""

    def __init__(self) -> None:
        self.forgot: list[str] = []

    def actor(self, user_pk: str) -> str:
        return user_pk

    def store(self, scope, content):
        return content

    def load_many(self, items):
        return [stored for _scope, stored in items]

    def forget(self, scope: str) -> None:
        self.forgot.append(scope)

    def chained(self, items) -> None:
        return None

    def pruning(self, turn_ids, *, access_ids=()) -> None:
        return None

    def expiring(self, content_ids) -> None:
        return None


class Unreachable(Remembering):
    """A custody whose store is down: it cannot forget."""

    def forget(self, scope: str) -> None:
        raise RuntimeError("vault down")


@contextlib.contextmanager
def installed(keeper):
    from django_ai_sdk.oversight import custody

    custody.install(keeper)
    try:
        yield keeper
    finally:
        custody.install(None)
