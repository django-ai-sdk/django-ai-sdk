"""Who may read the record, its content, and the review queues: denied unless granted.

Each operation takes a model permission; `is_staff` grants nothing. The scope hooks
narrow which turns and review items a reader reaches.
"""

from __future__ import annotations

import pytest
from django.contrib.auth.models import AnonymousUser
from django.test import override_settings

from django_ai_sdk.oversight.models import ReviewItem
from django_ai_sdk.oversight.permissions import OversightDefaultPermission
from django_ai_sdk.oversight.schemas import ResolveIn
from django_ai_sdk.oversight.services import OversightService
from django_ai_sdk.permissions import PermissionDenied, get_domain_permissions
from tests.oversight.review.helpers import CONTENT, METRICS, REVIEW, a_turn, reader


async def allowed(user, turn_id: str) -> dict[str, bool]:
    """What `user` may do, one operation each."""
    item = await ReviewItem.objects.filter(turn_id=turn_id).afirst()
    assert item is not None

    async def ok(call) -> bool:
        try:
            await call
        except PermissionDenied:
            return False
        return True

    detail = None
    if await ok(OversightService.get_turn(turn_id, user=user)):
        detail = await OversightService.get_turn(turn_id, user=user)
    return {
        "list": await ok(OversightService.list_turns(user=user)),
        "view": detail is not None,
        "content": detail is not None and detail.content is not None,
        "metrics": await ok(OversightService.summarize(user=user)),
        "reviews": await ok(OversightService.list_reviews(user=user)),
        "queues": await ok(OversightService.queues(user=user)),
        "resolve": await ok(
            OversightService.resolve_review(
                str(item.id), ResolveIn(dismiss=True), user=user
            )
        ),
    }


NOTHING = dict.fromkeys(
    ("list", "view", "content", "metrics", "reviews", "queues", "resolve"), False
)


@pytest.mark.django_db(transaction=True)
class TestDefaultDeny:
    async def test_anonymous_is_refused_everything(self):
        assert await allowed(AnonymousUser(), await a_turn()) == NOTHING

    async def test_a_signed_in_member_is_refused_everything(self):
        assert await allowed(await reader(), await a_turn()) == NOTHING

    async def test_staff_alone_is_refused_everything(self):
        assert await allowed(await reader(staff=True), await a_turn()) == NOTHING

    async def test_metrics_reads_the_record_without_its_content(self):
        assert await allowed(await reader(METRICS), await a_turn()) == {
            **NOTHING,
            "list": True,
            "view": True,
            "metrics": True,
        }

    async def test_content_comes_on_top_of_metrics(self):
        assert await allowed(await reader(METRICS, CONTENT), await a_turn()) == {
            **NOTHING,
            "list": True,
            "view": True,
            "content": True,
            "metrics": True,
        }

    async def test_content_alone_reads_no_turn(self):
        assert await allowed(await reader(CONTENT), await a_turn()) == NOTHING

    async def test_review_works_the_queues_only(self):
        assert await allowed(await reader(REVIEW), await a_turn()) == {
            **NOTHING,
            "reviews": True,
            "queues": True,
            "resolve": True,
        }

    async def test_a_superuser_holds_every_model_permission(self):
        assert await allowed(await reader(superuser=True), await a_turn()) == dict.fromkeys(
            NOTHING, True
        )

    async def test_an_overridden_domain_replaces_the_default(self):
        turn_id = await a_turn()
        with override_settings(
            AI_SDK_PERMISSIONS={"oversight": ["django_ai_sdk.permissions.IsAdminUser"]}
        ):
            get_domain_permissions.cache_clear()
            staff = await reader(staff=True)
            page = await OversightService.list_turns(user=staff)
        assert [row.id for row in page.items] == [turn_id]


class OnlyAgent(OversightDefaultPermission):
    """Reaches the turns of one channel only."""

    def scope_turns(self, user, operation, queryset):
        return queryset.filter(channel="api")


class NoContentForWebhooks(OversightDefaultPermission):
    """Every turn's record, but no webhook turn's content."""

    def scope_turns(self, user, operation, queryset):
        if operation == "view_oversight_content":
            return queryset.exclude(channel__startswith="webhook")
        return queryset


SCOPED = {"oversight": ["tests.oversight.review.test_permissions.OnlyAgent"]}


@pytest.fixture
def scoped():
    with override_settings(AI_SDK_PERMISSIONS=SCOPED):
        get_domain_permissions.cache_clear()
        yield


@pytest.mark.django_db(transaction=True)
class TestScopeHooks:
    @pytest.mark.usefixtures("scoped")
    async def test_a_turn_outside_the_scope_is_not_found(self):
        inside = await a_turn(channel="api")
        outside = await a_turn(channel="webhook:x")
        get_domain_permissions.cache_clear()
        everything = await reader(METRICS, CONTENT, REVIEW)
        assert [row.id for row in (await OversightService.list_turns(user=everything)).items] == [
            inside
        ]
        assert await OversightService.get_turn(outside, user=everything) is None
        assert (await OversightService.get_turn(inside, user=everything)).content is not None

    @pytest.mark.usefixtures("scoped")
    async def test_review_items_follow_their_turns(self):
        await a_turn(channel="api")
        outside = await a_turn(channel="webhook:x")
        get_domain_permissions.cache_clear()
        reviewer = await reader(REVIEW)
        page = await OversightService.list_reviews(user=reviewer)
        assert outside not in {item.turn_id for item in page.items}
        assert len(page.items) == 1
        assert await OversightService.queues(user=reviewer) == {"hr": 1}
        item = await ReviewItem.objects.aget(turn_id=outside)
        resolved = await OversightService.resolve_review(
            str(item.id), ResolveIn(dismiss=True), user=reviewer
        )
        assert resolved is None
        await item.arefresh_from_db()
        assert item.status == ReviewItem.Status.OPEN

    async def test_content_follows_the_scope_for_its_own_operation(self):
        inside = await a_turn(channel="api")
        outside = await a_turn(channel="webhook:x")
        path = "tests.oversight.review.test_permissions.NoContentForWebhooks"
        with override_settings(AI_SDK_PERMISSIONS={"oversight": [path]}):
            get_domain_permissions.cache_clear()
            user = await reader(METRICS, CONTENT)
            rows = {row.id: row for row in (await OversightService.list_turns(user=user)).items}
            hidden = await OversightService.get_turn(outside, user=user)
        assert rows[inside].preview == "What about my salary?"
        assert rows[outside].preview == ""
        assert hidden.content is None
        assert hidden.content_state == "withheld"


@pytest.mark.django_db(transaction=True)
class TestWhatEachOperationShows:
    async def test_a_reviewer_without_metrics_sees_items_without_their_turns_rows(self):
        await a_turn()
        [item] = (await OversightService.list_reviews(user=await reader(REVIEW))).items
        assert item.turn is None
        assert item.queue == "hr"
        [item] = (await OversightService.list_reviews(user=await reader(REVIEW, METRICS))).items
        assert item.turn is not None
        assert item.turn.labels["topic"] == "hr"
        assert item.turn.preview == ""

    async def test_a_reviewer_without_content_resolves_without_the_question(self):
        await a_turn()
        reviewer = await reader(REVIEW, METRICS)
        [item] = (await OversightService.list_reviews(user=reviewer)).items
        assert item.turn.preview == ""
        resolved = await OversightService.resolve_review(
            item.id, ResolveIn(agree=False, note="Not HR."), user=reviewer
        )
        assert resolved.status == "done"
        # The note is content: a reviewer without content access writes it but does not read it.
        assert resolved.note == ""
        [done] = (await OversightService.list_reviews(user=reviewer, status="done")).items
        assert done.note == ""

    async def test_the_thread_filter_takes_content(self):
        from django_ai_sdk.oversight.schemas import TurnFilter

        turn_id = await a_turn(thread_id="0f8b2d4e-7a11-4c3b-9f00-000000000001")
        await a_turn("hello", "hi")
        by_thread = TurnFilter(thread_id="0f8b2d4e-7a11-4c3b-9f00-000000000001")
        with pytest.raises(PermissionDenied):
            await OversightService.list_turns(by_thread, user=await reader(METRICS))
        page = await OversightService.list_turns(by_thread, user=await reader(METRICS, CONTENT))
        assert [row.id for row in page.items] == [turn_id]

    async def test_the_thread_filter_stays_inside_the_content_scope(self):
        from django_ai_sdk.oversight.schemas import TurnFilter

        thread = "0f8b2d4e-7a11-4c3b-9f00-000000000002"
        await a_turn(channel="webhook:x", thread_id=thread)
        path = "tests.oversight.review.test_permissions.NoContentForWebhooks"
        with override_settings(AI_SDK_PERMISSIONS={"oversight": [path]}):
            get_domain_permissions.cache_clear()
            user = await reader(METRICS, CONTENT)
            page = await OversightService.list_turns(TurnFilter(thread_id=thread), user=user)
        assert page.items == []
