"""Forgetting, as the review API shows it: a deleted thread's or user's turns read back without
what was forgotten, and pruned turns are gone."""

from __future__ import annotations

from datetime import timedelta

import pytest
from asgiref.sync import sync_to_async
from django.test import AsyncClient
from django.utils import timezone

from django_ai_sdk.oversight.models import Turn
from django_ai_sdk.oversight.retention import prune
from django_ai_sdk.oversight.schemas import ResolveIn, TurnFilter
from django_ai_sdk.oversight.services import OversightService
from tests.oversight.helpers import a_thread
from tests.oversight.review.helpers import CONTENT, METRICS, REVIEW, a_turn, reader

pytestmark = [
    pytest.mark.django_db(transaction=True),
    pytest.mark.urls("tests.oversight.review.urls"),
]


async def test_a_deleted_thread_reads_back_without_its_content():
    owner = await reader()
    thread = await a_thread(owner)
    thread_id = str(thread.id)
    turn_id = await a_turn(thread_id=thread_id)
    user = await reader(METRICS, CONTENT, REVIEW)
    # A reviewer's note is content too.
    [item] = (await OversightService.list_reviews(user=user)).items
    await OversightService.resolve_review(
        item.id, ResolveIn(agree=True, note="Asked about pay."), user=user
    )
    before = await OversightService.get_turn(turn_id, user=user)
    assert before.content["input"] == "What about my salary?"
    assert before.reviews[0].note == "Asked about pay."

    await thread.adelete()

    after = await OversightService.get_turn(turn_id, user=user)
    assert after.content is None
    assert after.content_state == "none"
    assert after.preview is None
    assert all(a.rationale == "" for a in after.answers)
    # The record stays: labels, matches and the reviewer's verdict, without the note.
    assert after.labels["topic"] == "hr"
    assert {m.rule for m in after.matches} >= {"hr_to_people"}
    [review] = after.reviews
    assert (review.resolved_by, review.agree, review.note) == (str(user.pk), True, "")
    [row] = (await OversightService.list_turns(TurnFilter(thread_id=thread_id), user=user)).items
    assert row.preview is None


async def test_a_deleted_user_is_nobody_on_their_turns():
    asker = await reader()
    turn_id = await a_turn(user=asker)
    user = await reader(METRICS, CONTENT)
    assert (await OversightService.get_turn(turn_id, user=user)).actor_id == str(asker.pk)

    await asker.adelete()

    detail = await OversightService.get_turn(turn_id, user=user)
    assert detail.actor_id == ""
    assert detail.content is None
    assert detail.content_state == "none"
    [row] = (await OversightService.list_turns(user=user)).items
    assert (row.actor_id, row.preview) == ("", None)


async def test_a_pruned_turn_is_not_found():
    turn_id = await a_turn()
    user = await reader(METRICS, CONTENT, REVIEW)
    client = AsyncClient()
    await sync_to_async(client.force_login)(user)
    assert (await client.get(f"/api/oversight/turns/{turn_id}/")).status_code == 200
    await Turn.objects.filter(id=turn_id).aupdate(expires_at=timezone.now() - timedelta(days=1))

    await sync_to_async(prune)()

    assert (await client.get(f"/api/oversight/turns/{turn_id}/")).status_code == 404
    assert (await client.get("/api/oversight/turns/")).json()["items"] == []
    assert (await client.get("/api/oversight/reviews/")).json()["items"] == []
    assert (await client.get("/api/oversight/queues/")).json() == {}


async def test_expired_content_reads_back_as_none():
    turn_id = await a_turn()
    user = await reader(METRICS, CONTENT)
    turn = await Turn.objects.aget(id=turn_id)
    from django_ai_sdk.oversight.models import TurnContent

    await TurnContent.objects.filter(id=turn.content_id).aupdate(
        expires_at=timezone.now() - timedelta(days=1)
    )
    # Expired and not yet pruned: withheld as expired.
    assert (await OversightService.get_turn(turn_id, user=user)).content_state == "expired"
    await sync_to_async(prune)()
    detail = await OversightService.get_turn(turn_id, user=user)
    assert (detail.content, detail.content_state) == (None, "none")
