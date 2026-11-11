"""Custody: content is kept as it is, and forgetting a scope deletes it."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from asgiref.sync import sync_to_async
from django.test import override_settings

from django_ai_sdk.oversight import custody
from django_ai_sdk.oversight.checks import check_oversight_deploy
from django_ai_sdk.oversight.models import LensAnswer, ReviewItem, RuleMatch, Turn, TurnContent
from django_ai_sdk.oversight.guard import TurnGuard
from tests.oversight.helpers import (
    make_agent,
)

pytestmark = pytest.mark.usefixtures("no_reflect_queue")


async def close_a_turn(text: str = "What about my salary?", thread_id: str | None = None) -> str:
    guard = await TurnGuard.start(
        make_agent([], None), channel="api", input=text, thread_id=thread_id
    )
    await guard.gate()
    await guard.finish("It is fine.")
    return await guard.close()


def capture_ids() -> list[str]:
    # The test settings run tasks inline (W014): only what capture says counts here.
    return [m.id for m in check_oversight_deploy() if m.id != "ai_sdk.oversight.W014"]


@pytest.mark.django_db(transaction=True)
class TestCustody:
    async def test_content_is_kept_as_it_is(self):
        turn_id = await close_a_turn()
        turn = await Turn.objects.select_related("content").aget(id=turn_id)
        assert turn.content is not None
        assert turn.content.stored["input"] == "What about my salary?"
        assert turn.content.scope == f"turn:{turn_id}"

    async def test_forgetting_a_scope_deletes_its_content_and_keeps_the_turn(self):
        turn_id = await close_a_turn(thread_id="t-1")
        assert await TurnContent.objects.filter(scope="thread:t-1").aexists()
        await sync_to_async(custody.forget)("thread:t-1")
        assert not await TurnContent.objects.filter(scope="thread:t-1").aexists()
        turn = await Turn.objects.aget(id=turn_id)
        assert turn.content_id is None
        assert await RuleMatch.objects.filter(turn=turn).aexists()

    async def test_forgetting_blanks_reviewers_notes(self):
        turn_id = await close_a_turn(thread_id="t-2")
        item = await ReviewItem.objects.acreate(turn_id=turn_id, queue="q", note="Jan said so")
        await sync_to_async(custody.forget)("thread:t-2")
        await item.arefresh_from_db()
        assert item.note == ""

    async def test_a_forgotten_scope_keeps_the_turn_without_content(self):
        await TurnContent.objects.acreate(scope="thread:t-3", capture=custody.FORGOTTEN)
        turn_id = await close_a_turn(thread_id="t-3")
        turn = await Turn.objects.aget(id=turn_id)
        assert turn.content_id is None
        assert await LensAnswer.objects.filter(turn=turn).aexists()

    async def test_a_deleted_thread_is_forgotten(self):
        from django_ai_sdk.conversation.models import Thread

        thread = await Thread.objects.acreate()
        thread_id = str(thread.id)
        await close_a_turn(thread_id=thread_id)
        with patch("django_ai_sdk.oversight.custody.forget") as forget:
            await thread.adelete()
        forget.assert_called_once_with(f"thread:{thread_id}")


class TestDeployCheck:
    def test_full_capture_warns(self):
        with override_settings(AI_SDK_OVERSIGHT={"CAPTURE": "full"}):
            assert capture_ids() == ["ai_sdk.oversight.W001"]

    def test_masked_capture_says_nothing(self):
        assert capture_ids() == []
