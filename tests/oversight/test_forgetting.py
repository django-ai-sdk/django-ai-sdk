"""Forgetting: deleting a thread deletes its oversight content, and nothing else."""

from __future__ import annotations

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

    async def test_the_command_forgets_a_thread(self):
        from asgiref.sync import sync_to_async
        from django.core.management import call_command

        from django_ai_sdk.oversight.models import TurnContent

        thread = await a_thread()
        await a_turn_in(thread)
        out = StringIO()
        await sync_to_async(call_command)("oversight_forget_thread", str(thread.id), stdout=out)
        assert f"Forgot thread:{thread.id}." in out.getvalue()
        assert not await TurnContent.objects.filter(scope=f"thread:{thread.id}").aexists()

    async def test_a_thread_oversight_never_saw_forgets_nothing(self):
        thread = await a_thread()
        with patch("django_ai_sdk.oversight.custody.forget") as forget:
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

    async def test_a_failed_forget_does_not_stop_the_delete(self):
        from django_ai_sdk.conversation.models import Thread

        thread = await a_thread()
        thread_id = thread.id
        await a_turn_in(thread)
        with (
            patch(
                "django_ai_sdk.oversight.custody.forget",
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
