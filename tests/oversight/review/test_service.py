"""OversightService: the turns feed and one turn, review queues, counts, and the access record."""

from __future__ import annotations

from datetime import timedelta
from unittest.mock import patch

import pytest
from asgiref.sync import async_to_sync, sync_to_async
from django.db import connection
from django.test.utils import CaptureQueriesContext
from django.utils import timezone
from pydantic import ValidationError
from django_ai_sdk.errors import UserError
from django_ai_sdk.oversight import L, Notify, Policy, Rule
from django_ai_sdk.oversight.models import AccessRecord, LensAnswer, ReviewItem, RuleMatch, Turn
from django_ai_sdk.oversight.schemas import ResolveIn, TurnFilter
from django_ai_sdk.oversight.services import OversightService
from django_ai_sdk.oversight.types import Stage
from django_ai_sdk.permissions import ConflictError, PermissionDenied, get_domain_permissions

from tests.oversight.helpers import FnLens
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

        # A new turn each time, so each read writes its access record.
        await a_turn()
        few = await queries()
        for _ in range(4):
            await a_turn()
        assert await queries() == few


def queries_of(call) -> int:
    """How many queries `call()` (a service call's coroutine) makes, its access record written.

    The access records go first: a repeat of the same read would write none.
    """
    AccessRecord.objects.all().delete()

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
class TestAccessRecord:
    async def test_a_list_records_the_ids_it_showed(self):
        first = await a_turn()
        second = await a_turn("hello", "hi")
        user = await reader(METRICS, CONTENT)
        await OversightService.list_turns(TurnFilter(outcome="flagged"), user=user)
        await OversightService.list_turns(user=user)
        records = [r async for r in AccessRecord.objects.order_by("created_at")]
        assert [r.operation for r in records] == ["list", "list"]
        assert records[0].query == {"outcome": "flagged"}
        assert set(records[1].turn_ids) == {first, second}
        assert records[1].content is True
        assert records[1].reader_id == str(user.pk)
        assert str(user.pk) in records[1].reader_label

    async def test_a_page_polling_is_one_record(self):
        await a_turn()
        user = await reader(METRICS)
        for _ in range(3):
            await OversightService.list_turns(user=user)
        [record] = [r async for r in AccessRecord.objects.all()]
        assert record.content is False

    async def test_a_view_and_a_review_list_are_recorded_counts_are_not(self):
        turn_id = await a_turn()
        user = await reader(METRICS, REVIEW)
        await OversightService.get_turn(turn_id, user=user)
        await OversightService.list_reviews(user=user)
        await OversightService.summarize(user=user)
        await OversightService.queues(user=user)
        records = [r async for r in AccessRecord.objects.order_by("created_at")]
        assert [(r.operation, r.turn_ids) for r in records] == [
            ("view", [turn_id]),
            ("reviews", [turn_id]),
        ]

    async def test_a_list_names_whose_content_it_showed_when_not_all(self):
        from django.test import override_settings

        inside = await a_turn(channel="api")
        outside = await a_turn(channel="webhook:x")
        path = "tests.oversight.review.test_permissions.NoContentForWebhooks"
        with override_settings(AI_SDK_PERMISSIONS={"oversight": [path]}):
            get_domain_permissions.cache_clear()
            user = await reader(METRICS, CONTENT, REVIEW)
            await OversightService.list_turns(user=user)
            await OversightService.list_reviews(user=user)
        for operation in ("list", "reviews"):
            record = await AccessRecord.objects.aget(operation=operation)
            assert set(record.turn_ids) == {inside, outside}
            assert record.content is True
            assert record.query["content_ids"] == [inside]

    async def test_a_list_that_showed_every_turns_content_names_none(self):
        await a_turn()
        await OversightService.list_turns(user=await reader(METRICS, CONTENT))
        record = await AccessRecord.objects.aget()
        assert record.content is True
        assert "content_ids" not in record.query


@pytest.mark.django_db(transaction=True)
class TestReviews:
    async def test_a_reviewer_resolves_an_item_with_an_answer_and_a_note(self):
        turn_id = await a_turn()
        user = await reader(METRICS, CONTENT, REVIEW)
        [item] = (await OversightService.list_reviews(user=user)).items
        assert item.turn.preview == "What about my salary?"
        resolved = await OversightService.resolve_review(
            item.id, ResolveIn(lens="topic", value="other", note="Not HR."), user=user
        )
        assert resolved.status == "done"
        assert resolved.resolution == {
            "by": str(user.pk),
            "value": "other",
            "agree": None,
            "note": "Not HR.",
        }
        answer = await LensAnswer.objects.aget(turn_id=turn_id, reviewer=str(user.pk))
        assert (answer.lens, answer.value, answer.answerer) == ("topic", "other", "reviewer")
        detail = await OversightService.get_turn(turn_id, user=user)
        assert detail.labels["topic"] == "other"
        [review] = detail.reviews
        assert review.resolution["note"] == "Not HR."
        assert (await OversightService.list_reviews(user=user)).items == []
        assert (await OversightService.list_reviews(user=user, status="done")).items[0].id == (
            item.id
        )
        records = [r async for r in AccessRecord.objects.filter(operation="review")]
        assert [r.turn_ids for r in records] == [[turn_id]]

    async def test_an_item_closes_once(self):
        await a_turn()
        user = await reader(REVIEW)
        item = await ReviewItem.objects.aget()
        await OversightService.resolve_review(str(item.id), ResolveIn(dismiss=True), user=user)
        with pytest.raises(ConflictError):
            await OversightService.resolve_review(str(item.id), ResolveIn(dismiss=True), user=user)
        await item.arefresh_from_db()
        assert item.status == ReviewItem.Status.DISMISSED

    async def test_a_call_on_a_rule_match(self):
        turn_id = await a_turn()
        user = await reader(REVIEW)
        item = await ReviewItem.objects.aget()
        assert item.rule == "hr_to_people"
        resolved = await OversightService.resolve_review(
            str(item.id), ResolveIn(agree=False), user=user
        )
        assert resolved.resolution["value"] is False
        assert resolved.resolution["agree"] is False
        answer = await LensAnswer.objects.aget(turn_id=turn_id, lens="rule:hr_to_people")
        assert answer.value is False
        turn = await Turn.objects.aget(id=turn_id)
        assert "rule:hr_to_people" not in turn.labels

    async def test_two_answers_at_once_match_a_rule_once(self):
        """Each finds the rule unmatched; the second to write finds it matched, under the lock."""
        from django_ai_sdk.oversight.persist import save_answers
        from django_ai_sdk.oversight.services.reviews import _reviewer_answer

        turn = await Turn.objects.select_related("content").aget(id=await a_turn("hello", "hi"))
        given = ResolveIn(lens="topic", value="hr")
        first = await _reviewer_answer(turn, "topic", given)
        second = await _reviewer_answer(turn, "topic", given)
        assert [m.rule.name for m in first[2]] == [m.rule.name for m in second[2]] != []
        for resolved, answered, matched in (first, second):
            await sync_to_async(save_answers)(turn, resolved, answered, matched, reviewer="7")
        assert second[2] == []
        assert await RuleMatch.objects.filter(turn=turn, rule="hr_to_people").acount() == 1
        assert await LensAnswer.objects.filter(turn=turn, reviewer="7").acount() == 2

    async def test_nobody_reviews_a_turn_of_their_own(self):
        asker = await reader(REVIEW)
        turn_id = await a_turn(user=asker)
        item = await ReviewItem.objects.aget(turn_id=turn_id)
        for call in (
            OversightService.resolve_review(
                str(item.id), ResolveIn(lens="topic", value="other"), user=asker
            ),
            OversightService.resolve_review(str(item.id), ResolveIn(dismiss=True), user=asker),
        ):
            with pytest.raises(PermissionDenied):
                await call
        await item.arefresh_from_db()
        assert item.status == ReviewItem.Status.OPEN
        assert not await LensAnswer.objects.filter(turn_id=turn_id).exclude(reviewer="").aexists()
        # Someone else may.
        other = await reader(REVIEW)
        done = await OversightService.resolve_review(
            str(item.id), ResolveIn(dismiss=True), user=other
        )
        assert done.status == "dismissed"

    async def test_a_turn_still_running_takes_no_answer(self):
        turn_id = await a_turn()
        await Turn.objects.filter(id=turn_id).aupdate(status=Turn.Status.OPEN)
        item = await ReviewItem.objects.aget(turn_id=turn_id)
        user = await reader(REVIEW)
        with pytest.raises(ConflictError):
            await OversightService.resolve_review(
                str(item.id), ResolveIn(lens="topic", value="other"), user=user
            )
        # Dismissing it writes no answer: that may happen while it runs.
        dismissed = await OversightService.resolve_review(
            str(item.id), ResolveIn(dismiss=True), user=user
        )
        assert dismissed.status == "dismissed"

    async def test_answers_that_do_not_fit_are_refused(self):
        await a_turn()
        user = await reader(REVIEW)
        item = await ReviewItem.objects.aget()
        for given in (
            ResolveIn(lens="nope", value=True),
            ResolveIn(lens="topic", value="not-a-label"),
            ResolveIn(lens="rule:never_matched", agree=True),
        ):
            with pytest.raises(UserError):
                await OversightService.resolve_review(str(item.id), given, user=user)
        await item.arefresh_from_db()
        assert item.status == ReviewItem.Status.OPEN

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
        "salt": "s",
        "texts": {"x": {"rationale": "r"}},
        "notes": {"y": "n"},
    }
    assert visible_content(kept) == {"input": "q", "output": "a", "sources": []}
    assert visible_content(None) is None


class Leaks(Policy):
    """A reviewer's call that a turn leaked alerts the log notifier."""

    name = "test-leaks"
    lenses = [
        FnLens(
            name="leak",
            stages=[Stage.REFLECT],
            fn=lambda text, view: False,
        )
    ]
    rules = [
        Rule(
            "leak_alert",
            when=L("leak").is_true,
            then=Notify(to=["log"], severity="critical"),
            mode="enforce",
        ),
        Rule("leak_seen", when=L("leak").is_true, then=Notify(), mode="report"),
    ]


@pytest.mark.django_db(transaction=True)
class TestReviewerMatchesNotify:
    @pytest.fixture(autouse=True)
    def _leaks(self, registry_snapshot):
        registry_snapshot.register_policy(Leaks)
        yield

    @staticmethod
    async def leak_item() -> ReviewItem:
        """A review item on the "leak" lens of a fresh turn."""
        turn = await Turn.objects.aget(id=await a_turn("hello", "hi", policy="test-leaks"))
        return await ReviewItem.objects.acreate(turn=turn, queue="leaks", lens="leak")

    async def test_an_answer_that_makes_a_notify_rule_match_sends_it(self):
        item = await self.leak_item()
        turn_id = str(item.turn_id)
        user = await reader(REVIEW)
        with patch("django_ai_sdk.oversight.persist.enqueue_notifications") as enqueue:
            await OversightService.resolve_review(str(item.id), ResolveIn(value=True), user=user)
        [[events]] = [call.args for call in enqueue.call_args_list]
        [(event, to)] = events
        assert to == ["log"]
        assert (event["rule"], event["stage"], event["severity"]) == (
            "leak_alert",
            "review",
            "critical",
        )
        assert event["turn_id"] == turn_id
        # Report-mode rules match but act on nothing, so they send nothing.
        assert await RuleMatch.objects.filter(turn_id=turn_id, rule="leak_seen").aexists()

    async def test_an_answer_that_matches_nothing_sends_nothing(self):
        item = await self.leak_item()
        user = await reader(REVIEW)
        with patch("django_ai_sdk.oversight.persist.enqueue_notifications") as enqueue:
            await OversightService.resolve_review(str(item.id), ResolveIn(value=False), user=user)
        enqueue.assert_not_called()
