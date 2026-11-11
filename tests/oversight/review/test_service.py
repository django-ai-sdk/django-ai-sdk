"""OversightService: the turns feed and one turn, review queues and counts."""

from __future__ import annotations

from datetime import timedelta

import pytest
from asgiref.sync import async_to_sync, sync_to_async
from django.db import connection
from django.test.utils import CaptureQueriesContext
from django.utils import timezone
from django_ai_sdk.errors import UserError
from django_ai_sdk.oversight.models import LensAnswer, ReviewItem, RuleMatch, Turn
from django_ai_sdk.oversight.permissions import OversightDefaultPermission
from django_ai_sdk.oversight.schemas import ResolveIn, TurnFilter
from django_ai_sdk.oversight.services import OversightService
from django_ai_sdk.permissions import (
    ConflictError,
    Operation,
    PermissionDenied,
)
from pydantic import ValidationError

from tests.oversight.review.helpers import CONTENT, METRICS, REVIEW, a_turn, reader


@pytest.mark.django_db(transaction=True)
class TestTurns:
    async def test_a_row_shows_the_record_and_content_only_to_who_may_see_it(self):
        asker = await reader()
        turn_id = await a_turn(user=asker)
        [row] = (await OversightService.list_turns(user=await reader(METRICS))).items
        assert row.id == turn_id
        assert row.labels["topic"] == "hr"
        assert {m.rule for m in row.matches} >= {"hr_to_people", "would_block_hr"}
        assert row.reviews_open == 1
        assert row.preview is None
        assert row.actor_id == ""
        [row] = (await OversightService.list_turns(user=await reader(METRICS, CONTENT))).items
        assert row.preview == "What about my salary?"
        assert row.actor_id == str(asker.pk)

    async def test_one_turn_in_full(self):
        turn_id = await a_turn()
        detail = await OversightService.get_turn(turn_id, user=await reader(METRICS, CONTENT))
        assert detail.content_state == "full"
        assert detail.content["input"] == "What about my salary?"
        assert "salt" not in detail.content
        assert {a.lens for a in detail.answers} >= {"topic", "pii_out"}
        [review] = detail.reviews
        assert review.queue == "hr"

    async def test_without_content_the_answers_keep_no_words(self):
        turn_id = await a_turn()
        detail = await OversightService.get_turn(turn_id, user=await reader(METRICS))
        assert detail.content is None
        assert detail.content_state == "withheld"
        assert all(a.rationale == "" and a.spans == [] for a in detail.answers)
        assert detail.thread_id == ""

    async def test_a_turn_that_kept_no_content_says_so(self):
        turn_id = await a_turn(capture="none")
        detail = await OversightService.get_turn(turn_id, user=await reader(METRICS, CONTENT))
        assert detail.content is None
        assert detail.content_state == "none"

    async def test_an_unknown_or_malformed_id_is_none(self):
        user = await reader(METRICS)
        assert await OversightService.get_turn("nope", user=user) is None
        assert (
            await OversightService.get_turn("00000000-0000-0000-0000-000000000000", user=user)
            is None
        )

    async def test_filters(self):
        await a_turn()
        await a_turn("hello there", "Hi!", channel="webhook:telegram")
        user = await reader(METRICS)

        async def count(**fields) -> int:
            return len((await OversightService.list_turns(TurnFilter(**fields), user=user)).items)

        assert await count() == 2
        assert await count(label="topic") == 2
        assert await count(label="topic", value="hr") == 1
        assert await count(report_outcome="blocked") == 1
        assert await count(rule="hr_to_people") == 1
        assert await count(review="open") == 1
        assert await count(review="unreviewed") == 2
        assert await count(channel="webhook") == 1
        assert await count(status="closed") == 2
        assert await count(until=timezone.now() - timedelta(days=1)) == 0
        assert await count(since=timezone.now() - timedelta(days=1)) == 2
        with pytest.raises(UserError):
            await count(label="topic__x")
        with pytest.raises(ValidationError):
            await count(review="maybe")
        with pytest.raises(ValidationError):
            await count(outcome="blokced")

    async def test_a_turn_with_two_open_reviews_lists_once(self):
        turn_id = await a_turn()
        turn = await Turn.objects.aget(id=turn_id)
        await ReviewItem.objects.acreate(turn=turn, queue="legal", rule="hr_to_people")
        user = await reader(METRICS)
        page = await OversightService.list_turns(TurnFilter(review="open"), user=user)
        assert [row.id for row in page.items] == [turn_id]

    async def test_after_close_a_rule_matches_a_turn_once(self):
        from django.db import IntegrityError

        turn = await Turn.objects.aget(id=await a_turn())
        await RuleMatch.objects.acreate(turn=turn, rule="late", stage="reflect", mode="enforce")
        with pytest.raises(IntegrityError):
            await RuleMatch.objects.acreate(turn=turn, rule="late", stage="reflect", mode="report")

    async def test_pages_follow_a_cursor_without_a_count(self):
        made = [await a_turn(f"question {n}", "ok") for n in range(5)]
        # Two turns at the same instant still page apart, by id.
        same = timezone.now() - timedelta(hours=1)
        await Turn.objects.filter(id__in=made[:2]).aupdate(created_at=same)
        user = await reader(METRICS)
        seen: list[str] = []
        cursor = ""
        pages = 0
        while True:
            page = await OversightService.list_turns(user=user, limit=2, cursor=cursor)
            seen += [row.id for row in page.items]
            pages += 1
            if not page.next_cursor:
                break
            cursor = page.next_cursor
        assert pages == 3
        assert sorted(seen) == sorted(made)
        assert len(set(seen)) == 5

    async def test_a_cursor_it_did_not_hand_out_is_refused(self):
        user = await reader(METRICS)
        for cursor in ("garbage", "W10", "WyJhIl0"):
            with pytest.raises(UserError):
                await OversightService.list_turns(user=user, cursor=cursor)

    async def test_a_tampered_cursor_value_is_refused_not_a_server_error(self):
        import base64
        import json

        tampered = base64.urlsafe_b64encode(
            json.dumps(["a", timezone.now().isoformat(), "not-a-uuid"]).encode()
        ).decode()
        with pytest.raises(UserError):
            await OversightService.list_reviews(user=await reader(REVIEW), cursor=tampered)

    async def test_a_label_named_like_a_lookup_is_a_key(self):
        await a_turn()
        rows = await OversightService.list_turns(
            TurnFilter(label="contains", value="x"), user=await reader(METRICS)
        )
        assert rows.items == []

    async def test_expired_content_shows_in_no_list(self):
        turn_id = await a_turn()
        turn = await Turn.objects.select_related("content").aget(id=turn_id)
        turn.content.expires_at = timezone.now() - timedelta(days=1)
        await turn.content.asave()
        user = await reader(METRICS, CONTENT, REVIEW)

        [row] = (await OversightService.list_turns(user=user)).items
        [item] = (await OversightService.list_reviews(user=user)).items

        assert row.preview is None
        assert item.turn.preview is None
        assert (await OversightService.get_turn(turn_id, user=user)).content_state == "expired"

    async def test_the_feed_reads_a_fixed_number_of_queries(self):
        user = await reader(METRICS, CONTENT)
        await a_turn()
        await OversightService.list_turns(user=user)  # the permission cache warms up

        def counted() -> int:
            # In the thread the ORM calls run in, so the context sees their connection.
            with CaptureQueriesContext(connection) as captured:
                async_to_sync(OversightService.list_turns)(user=user)
            return len(captured.captured_queries)

        async def queries() -> int:
            return await sync_to_async(counted)()

        await a_turn()
        few = await queries()
        for _ in range(4):
            await a_turn()
        assert await queries() == few


def queries_of(call) -> int:
    """How many queries `call()` (a service call's coroutine) makes."""

    async def run() -> None:
        await call()

    with CaptureQueriesContext(connection) as captured:
        async_to_sync(run)()
    return len(captured.captured_queries)


@pytest.mark.django_db(transaction=True)
class TestQueryCounts:
    """However many answers, matches, reviews or items: the same number of queries."""

    async def test_one_turn_reads_a_fixed_number_of_queries(self):
        user = await reader(METRICS, CONTENT)
        turn_id = await a_turn()
        await OversightService.get_turn(turn_id, user=user)  # the permission cache warms up
        few = await sync_to_async(queries_of)(lambda: OversightService.get_turn(turn_id, user=user))
        turn = await Turn.objects.aget(id=turn_id)
        for n in range(4):
            await LensAnswer.objects.acreate(turn=turn, lens="topic", stage="reflect", value="hr")
            await RuleMatch.objects.acreate(turn=turn, rule=f"r{n}", stage="reflect", mode="report")
            await ReviewItem.objects.acreate(turn=turn, queue=f"q{n}")
        many = await sync_to_async(queries_of)(
            lambda: OversightService.get_turn(turn_id, user=user)
        )
        assert many == few

    async def test_the_review_list_reads_a_fixed_number_of_queries(self):
        user = await reader(METRICS, CONTENT, REVIEW)
        await a_turn()
        await OversightService.list_reviews(user=user)
        few = await sync_to_async(queries_of)(lambda: OversightService.list_reviews(user=user))
        for n in range(4):
            await a_turn(f"salary {n}")
        many = await sync_to_async(queries_of)(lambda: OversightService.list_reviews(user=user))
        assert many == few
        assert len((await OversightService.list_reviews(user=user)).items) == 5


@pytest.mark.django_db(transaction=True)
class TestReviews:
    async def test_a_reviewer_agrees_with_a_note(self):
        turn_id = await a_turn()
        user = await reader(METRICS, CONTENT, REVIEW)
        [item] = (await OversightService.list_reviews(user=user)).items
        assert item.turn.preview == "What about my salary?"
        resolved = await OversightService.resolve_review(
            item.id, ResolveIn(agree=True, note="Right call."), user=user
        )
        assert (resolved.status, resolved.agree, resolved.note) == ("done", True, "Right call.")
        assert resolved.resolved_by == str(user.pk)
        [review] = (await OversightService.get_turn(turn_id, user=user)).reviews
        assert review.note == "Right call."
        assert (await OversightService.list_reviews(user=user)).items == []
        assert (await OversightService.list_reviews(user=user, status="done")).items[0].id == (
            item.id
        )

    async def test_an_item_closes_once(self):
        await a_turn()
        user = await reader(REVIEW)
        item = await ReviewItem.objects.aget()
        await OversightService.resolve_review(str(item.id), ResolveIn(dismiss=True), user=user)
        with pytest.raises(ConflictError):
            await OversightService.resolve_review(str(item.id), ResolveIn(dismiss=True), user=user)
        await item.arefresh_from_db()
        assert item.status == ReviewItem.Status.DISMISSED

    async def test_nobody_reviews_a_turn_of_their_own(self):
        asker = await reader(REVIEW)
        turn_id = await a_turn(user=asker)
        item = await ReviewItem.objects.aget(turn_id=turn_id)
        for call in (
            OversightService.resolve_review(str(item.id), ResolveIn(agree=False), user=asker),
            OversightService.resolve_review(str(item.id), ResolveIn(dismiss=True), user=asker),
        ):
            with pytest.raises(PermissionDenied):
                await call
        await item.arefresh_from_db()
        assert item.status == ReviewItem.Status.OPEN
        # Nor a turn a subagent ran for it, which carries no actor of its own.
        child = await a_turn()
        await Turn.objects.filter(id=child).aupdate(actor_id="", parent_turn_id=turn_id)
        child_item = await ReviewItem.objects.aget(turn_id=child)
        with pytest.raises(PermissionDenied):
            await OversightService.resolve_review(
                str(child_item.id), ResolveIn(dismiss=True), user=asker
            )
        # Someone else may.
        other = await reader(REVIEW)
        done = await OversightService.resolve_review(
            str(item.id), ResolveIn(dismiss=True), user=other
        )
        assert done.status == "dismissed"

    async def test_a_verdict_or_a_dismissal_is_required(self):
        await a_turn()
        user = await reader(REVIEW)
        item = await ReviewItem.objects.aget()
        for given in (ResolveIn(), ResolveIn(agree=True, dismiss=True)):
            with pytest.raises(UserError):
                await OversightService.resolve_review(str(item.id), given, user=user)
        await item.arefresh_from_db()
        assert item.status == ReviewItem.Status.OPEN

    async def test_review_rows_follow_the_metrics_scope(self, settings):
        await a_turn()
        settings.AI_SDK_PERMISSIONS = {"oversight": [f"{__name__}.NoMetricTurns"]}
        [item] = (await OversightService.list_reviews(user=await reader(METRICS, REVIEW))).items
        assert item.turn is None

    async def test_review_pages_follow_a_cursor(self):
        for n in range(3):
            await a_turn(f"salary {n}")
        user = await reader(REVIEW)
        first = await OversightService.list_reviews(user=user, limit=2)
        second = await OversightService.list_reviews(user=user, limit=2, cursor=first.next_cursor)
        assert len(first.items) == 2
        assert len(second.items) == 1
        assert second.next_cursor == ""
        assert {i.id for i in first.items}.isdisjoint({i.id for i in second.items})

    async def test_queues_count_open_items(self):
        await a_turn()
        await a_turn("salary again")
        assert await OversightService.queues(user=await reader(REVIEW)) == {"hr": 2}


@pytest.mark.django_db(transaction=True)
class TestMetrics:
    async def test_counts_over_the_window(self):
        await a_turn()
        await a_turn("hello", "hi", channel="webhook:x")
        metrics = await OversightService.summarize(user=await reader(METRICS))
        assert metrics.turns == 2
        assert sum(metrics.by_outcome.values()) == 2
        assert metrics.by_report_outcome == {"blocked": 1}
        assert metrics.by_channel == {"api": 1, "webhook:x": 1}
        [agent_id] = {t.agent_id async for t in Turn.objects.all()}
        assert metrics.by_agent == {agent_id: 2}
        rules = {row["rule"]: row for row in metrics.rules}
        assert rules["hr_to_people"]["matches"] == 1
        assert rules["hr_to_people"]["acted"] == 1
        assert rules["would_block_hr"]["acted"] == 0
        assert metrics.queues == {"hr": 1}

    async def test_a_policy_narrows_it(self):
        await a_turn()
        metrics = await OversightService.summarize(user=await reader(METRICS), policy="other")
        assert metrics.turns == 0
        assert metrics.queues == {}


def test_visible_content_shows_the_turn_as_kept_and_nothing_beside_it():
    from django_ai_sdk.oversight.services.turns import visible_content

    kept = {
        "input": "q",
        "output": "a",
        "sources": [],
        "texts": {"x": {"rationale": "r"}},
    }
    assert visible_content(kept) == {"input": "q", "output": "a", "sources": []}
    assert visible_content(None) is None


class NoMetricTurns(OversightDefaultPermission):
    """Reviews every item, but reaches no turn's metrics."""

    def scope_turns(self, user, operation, queryset):
        return queryset.none() if operation == Operation.VIEW_OVERSIGHT_METRICS else queryset

    def scope_reviews(self, user, operation, queryset):
        return queryset


@pytest.mark.django_db(transaction=True)
async def test_an_empty_oversight_permission_list_falls_back_to_the_default(settings):
    turn_id = await a_turn()
    settings.AI_SDK_PERMISSIONS = {"oversight": []}
    with pytest.raises(PermissionDenied):
        await OversightService.get_turn(turn_id, user=await reader())


@pytest.mark.django_db(transaction=True)
async def test_the_admin_follows_the_metrics_scope(settings):
    from django.contrib.admin import AdminSite
    from django.test import RequestFactory, modify_settings

    with modify_settings(INSTALLED_APPS={"append": "django.contrib.admin"}):
        from django_ai_sdk.oversight import admin

    await a_turn()
    settings.AI_SDK_PERMISSIONS = {"oversight": [f"{__name__}.NoMetricTurns"]}
    request = RequestFactory().get("/")
    request.user = await reader(METRICS, REVIEW)

    for model, model_admin in (
        (Turn, admin.TurnAdmin),
        (LensAnswer, admin.LensAnswerAdmin),
        (RuleMatch, admin.RuleMatchAdmin),
        (ReviewItem, admin.ReviewItemAdmin),
    ):
        rows = await sync_to_async(model_admin(model, AdminSite()).get_queryset)(request)
        assert not await rows.aexists(), model
