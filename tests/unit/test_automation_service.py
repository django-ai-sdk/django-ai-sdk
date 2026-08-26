"""AutomationService: the seam every HTTP layer sits on.

A router tells 403 from 404 by what this returns versus what it raises.
"""

from __future__ import annotations

from datetime import timedelta
from unittest.mock import AsyncMock, patch

import pytest
from django.utils import timezone

from django_ai_sdk.automations import Audience, Automation, AutomationService
from django_ai_sdk.automations.models import AutomationRun, AutomationState
from django_ai_sdk.automations.registry import register, reset_registry
from django_ai_sdk.automations.runner import AutomationBusy
from django_ai_sdk.permissions import PermissionDenied
from django_ai_sdk.workflows import WorkflowDefinition, WorkflowStep
from django_ai_sdk.workflows.registry import register as register_wf
from django_ai_sdk.workflows.registry import reset_registry as reset_workflows


@pytest.fixture(autouse=True)
def _clean_registries():
    reset_registry()
    reset_workflows()
    yield
    reset_registry()
    reset_workflows()


@pytest.fixture(autouse=True)
def _no_queue():
    """run_now goes through the real runner; only the queue hop is stubbed."""
    with patch(
        "django_ai_sdk.automations.runner._enqueue_all", new=AsyncMock(return_value=None)
    ):
        yield


def declare(name="example", **attrs):
    register_wf(
        WorkflowDefinition(
            name="wf", steps=[WorkflowStep(agent_id="a", output_key="result")]
        )
    )
    defaults = {"name": name, "cron": "0 9 * * *", "workflow": "wf"}
    register(type("Example", (Automation,), {**defaults, **attrs}))


async def make_user(name="staffer", **flags):
    from tests.factories.db import UserFactory

    return await UserFactory.acreate(email=f"{name}@example.com", **flags)


@pytest.mark.django_db(transaction=True)
class TestListForUser:
    async def test_it_describes_declared_automations(self):
        declare()
        user = await make_user(is_staff=True)

        [row] = await AutomationService.list_for_user(user)

        assert row.name == "example"
        assert row.workflow == "wf"
        assert row.audience == "app"
        assert row.enabled_source == "code"

    async def test_nothing_declared_is_an_empty_list(self):
        assert await AutomationService.list_for_user(await make_user()) == []

    async def test_it_reports_which_layer_disabled_an_automation(self, settings):
        # "Why is this off?" has a different answer each time.
        settings.AI_SDK_AUTOMATIONS = {"example": {"ENABLED": False}}
        declare()

        [row] = await AutomationService.list_for_user(await make_user(is_staff=True))

        assert (row.enabled, row.enabled_source) == (False, "settings")

    async def test_an_unusable_schedule_surfaces_as_detail(self):
        declare()
        user = await make_user(is_staff=True)

        with patch.object(Automation, "get_schedule", side_effect=RuntimeError("nope")):
            [row] = await AutomationService.list_for_user(user)

        # Better a visible "cannot run" than an automation that silently never fires.
        assert "nope" in row.detail


@pytest.mark.django_db(transaction=True)
class TestSetEnabled:
    async def test_it_writes_the_database_layer(self):
        declare()
        user = await make_user(is_staff=True)

        row = await AutomationService.set_enabled("example", enabled=False, user=user)

        assert (row.enabled, row.enabled_source) == (False, "db")
        state = await AutomationState.objects.aget(name="example")
        assert state.enabled is False

    async def test_an_unknown_name_returns_none_rather_than_raising(self):
        # The router answers 404 from this without catching an exception.
        user = await make_user(is_staff=True)
        assert await AutomationService.set_enabled("nope", enabled=True, user=user) is None

    async def test_a_non_staff_user_is_denied(self):
        declare()
        user = await make_user("regular")

        with pytest.raises(PermissionDenied):
            await AutomationService.set_enabled("example", enabled=False, user=user)


@pytest.mark.django_db(transaction=True)
class TestSetSubscribed:
    async def test_a_non_staff_user_may_subscribe_to_their_own_row(self):
        from django_ai_sdk.automations.models import AutomationSubscription

        declare()
        user = await make_user("regular")

        row = await AutomationService.set_subscribed("example", enabled=True, user=user)

        assert row.subscribed is True
        subscription = await AutomationSubscription.objects.aget(name="example", user=user)
        assert subscription.enabled is True

    async def test_unsubscribing_flips_the_same_row_rather_than_adding_another(self):
        from django_ai_sdk.automations.models import AutomationSubscription

        declare()
        user = await make_user("regular")

        await AutomationService.set_subscribed("example", enabled=True, user=user)
        row = await AutomationService.set_subscribed("example", enabled=False, user=user)

        assert row.subscribed is False
        assert await AutomationSubscription.objects.filter(name="example", user=user).acount() == 1

    async def test_an_unknown_name_returns_none_rather_than_raising(self):
        user = await make_user("regular")
        assert await AutomationService.set_subscribed("nope", enabled=True, user=user) is None

    async def test_it_never_touches_another_users_row(self):
        from django_ai_sdk.automations.models import AutomationSubscription

        declare()
        other = await make_user("other")
        await AutomationSubscription.objects.acreate(name="example", user=other, enabled=True)

        caller = await make_user("regular")
        await AutomationService.set_subscribed("example", enabled=True, user=caller)

        other_row = await AutomationSubscription.objects.aget(name="example", user=other)
        assert other_row.enabled is True


@pytest.mark.django_db(transaction=True)
class TestRunNow:
    async def test_it_dispatches_and_marks_the_trigger(self):
        declare()
        user = await make_user(is_staff=True)

        [run] = await AutomationService.run_now("example", user=user)

        assert run.trigger == AutomationRun.Trigger.MANUAL
        assert run.status == AutomationRun.Status.PENDING

    async def test_an_unknown_name_returns_none(self):
        user = await make_user(is_staff=True)
        assert await AutomationService.run_now("nope", user=user) is None

    async def test_a_non_staff_user_is_denied(self):
        declare()
        user = await make_user("regular")

        with pytest.raises(PermissionDenied):
            await AutomationService.run_now("example", user=user)

    async def test_it_runs_as_the_audience_not_the_caller(self):
        # The caller is who asked; the audience decides identity, exactly as on a tick.
        from django_ai_sdk.automations.models import AutomationSubscription

        declare(audience=Audience.SUBSCRIBED)
        target = await make_user("target")
        await AutomationSubscription.objects.acreate(
            name="example", user=target, enabled=True
        )
        caller = await make_user("staffer", is_staff=True)

        [run] = await AutomationService.run_now("example", user=caller)

        assert run.user_id != caller.pk

    async def test_it_fans_out_to_every_subscriber_not_just_the_caller(self):
        # "Run now" is the schedule firing early, not a private preview.
        from django_ai_sdk.automations.models import AutomationSubscription

        declare(audience=Audience.SUBSCRIBED)
        caller = await make_user("staffer", is_staff=True)
        for name in ("bosun", "cook", "lookout"):
            subscriber = await make_user(name)
            await AutomationSubscription.objects.acreate(
                name="example", user=subscriber, enabled=True
            )

        runs = await AutomationService.run_now("example", user=caller)

        assert len(runs) == 3
        assert caller.pk not in {r.user_id for r in runs}
        # One tick, one dispatch_id, however many principals it fanned out to.
        assert len({r.dispatch_id for r in runs}) == 1

    async def test_a_held_lease_raises_rather_than_starting_a_second_copy(self):
        declare()
        user = await make_user(is_staff=True)
        await AutomationService.run_now("example", user=user)

        # The router turns this into 409, so it needs its own exception type.
        with pytest.raises(AutomationBusy, match="already running"):
            await AutomationService.run_now("example", user=user)


@pytest.mark.django_db(transaction=True)
class TestRunHistory:
    async def test_it_lists_newest_first(self):
        declare()
        user = await make_user(is_staff=True)
        now = timezone.now()
        for offset in range(3):
            await AutomationRun.objects.acreate(
                name="example", scheduled_for=now - timedelta(hours=offset)
            )

        runs = await AutomationService.list_runs("example", user=user)

        assert [r.scheduled_for for r in runs] == sorted(
            (r.scheduled_for for r in runs), reverse=True
        )

    async def test_a_single_run_is_reachable_by_id(self):
        declare()
        user = await make_user(is_staff=True)
        created = await AutomationRun.objects.acreate(
            name="example", scheduled_for=timezone.now(), skip_reason="nobody connected"
        )

        run = await AutomationService.get_run(str(created.id), user=user)

        assert run.skip_reason == "nobody connected"

    async def test_an_unknown_run_returns_none(self):
        import uuid

        user = await make_user(is_staff=True)
        assert await AutomationService.get_run(str(uuid.uuid4()), user=user) is None

    async def test_history_for_an_undeclared_automation_is_still_permission_checked(self):
        # Removing the declaration must not turn its audit trail into an open endpoint.
        user = await make_user("anyone")
        await AutomationRun.objects.acreate(name="deleted", scheduled_for=timezone.now())

        with patch.object(
            type(user), "is_authenticated", property(lambda self: False)
        ), pytest.raises(PermissionDenied):
            await AutomationService.list_runs("deleted", user=user)

    async def test_a_run_of_an_undeclared_automation_is_still_permission_checked(self):
        user = await make_user("anyone")
        created = await AutomationRun.objects.acreate(
            name="deleted", scheduled_for=timezone.now()
        )

        with patch.object(
            type(user), "is_authenticated", property(lambda self: False)
        ), pytest.raises(PermissionDenied):
            await AutomationService.get_run(str(created.id), user=user)


@pytest.mark.django_db(transaction=True)
class TestRunHistoryIsScopedToTheCaller:
    """A run's output is the workflow's result, so a subscribed run is private content.

    VIEW_AUTOMATION is what puts an automation on a settings page for every
    authenticated user; it must not also hand them each other's results.
    """

    async def _two_users_one_run(self):
        declare(audience=Audience.SUBSCRIBED)
        alice = await make_user("alice")
        bob = await make_user("bob")
        run = await AutomationRun.objects.acreate(
            name="example",
            user=alice,
            scheduled_for=timezone.now(),
            status=AutomationRun.Status.SUCCEEDED,
            output={"secret": "alice only"},
        )
        return alice, bob, run

    async def test_another_user_cannot_list_someone_elses_run(self):
        _, bob, _ = await self._two_users_one_run()

        assert await AutomationService.list_runs("example", user=bob) == []

    async def test_a_user_sees_their_own_run(self):
        alice, _, _ = await self._two_users_one_run()

        [run] = await AutomationService.list_runs("example", user=alice)
        assert run.output == {"secret": "alice only"}

    async def test_a_manager_sees_the_whole_history(self):
        await self._two_users_one_run()
        boss = await make_user("boss", is_staff=True)

        assert len(await AutomationService.list_runs("example", user=boss)) == 1

    async def test_another_user_reading_a_run_by_id_gets_none_not_a_403(self):
        # A 403 would confirm the id exists, which the caller has no way to know.
        _, bob, run = await self._two_users_one_run()

        assert await AutomationService.get_run(str(run.id), user=bob) is None

    async def test_the_owner_reads_their_own_run_by_id(self):
        alice, _, run = await self._two_users_one_run()

        assert (await AutomationService.get_run(str(run.id), user=alice)) is not None

    async def test_an_app_level_run_is_not_readable_by_an_ordinary_user(self):
        # No user owns it, so nobody but a manager has a claim on its output.
        declare()
        await AutomationRun.objects.acreate(
            name="example", scheduled_for=timezone.now(), output={"internal": True}
        )
        onlooker = await make_user("onlooker")

        assert await AutomationService.list_runs("example", user=onlooker) == []


@pytest.mark.django_db(transaction=True)
class TestTickHealth:
    async def test_it_is_none_before_anything_dispatches(self):
        # A health endpoint points a dead-man's switch at this; the SDK cannot
        # otherwise notice its own scheduler being absent.
        assert await AutomationService.last_tick_at() is None

    async def test_it_reports_the_most_recent_dispatch(self):
        now = timezone.now()
        await AutomationState.objects.acreate(
            name="older", next_run_at=now, last_dispatched_at=now - timedelta(hours=2)
        )
        await AutomationState.objects.acreate(
            name="newer", next_run_at=now, last_dispatched_at=now
        )

        assert await AutomationService.last_tick_at() == now


@pytest.mark.django_db(transaction=True)
class TestAnAnonymousCallerIsNotAnOwner:
    """A host may permit anonymous VIEW; an app-level run must still not read as theirs."""

    @staticmethod
    def _allow_anonymous_view():
        from django_ai_sdk.permissions import BasePermission, Operation

        class ViewOnly(BasePermission):
            async def has_permission(self, user, operation, **kwargs):
                return operation == Operation.VIEW_AUTOMATION

        return ViewOnly

    async def test_listing_runs_does_not_blow_up_on_an_anonymous_user(self):
        from django.contrib.auth.models import AnonymousUser

        declare(permissions=[self._allow_anonymous_view()])
        await AutomationRun.objects.acreate(
            name="example", scheduled_for=timezone.now(), user=None
        )

        assert await AutomationService.list_runs("example", user=AnonymousUser()) == []

    async def test_an_app_level_run_is_not_readable_as_your_own(self):
        from django.contrib.auth.models import AnonymousUser

        declare(permissions=[self._allow_anonymous_view()])
        run = await AutomationRun.objects.acreate(
            name="example", scheduled_for=timezone.now(), user=None
        )

        assert await AutomationService.get_run(str(run.id), user=AnonymousUser()) is None


@pytest.mark.django_db(transaction=True)
class TestABrokenScheduleStillToggles:
    """An unusable cron is reported, not raised: the operator still needs the off switch."""

    async def test_it_writes_enabled_and_reports_the_reason(self):
        from django.core.exceptions import ImproperlyConfigured

        declare()
        user = await make_user(is_staff=True)
        broken = ImproperlyConfigured("'nope' is not a valid 5-field cron expression")

        with patch.object(Automation, "get_schedule", side_effect=broken):
            row = await AutomationService.set_enabled("example", enabled=False, user=user)

        assert row.enabled is False
        assert "not a valid" in row.detail
        assert await AutomationState.objects.filter(name="example", enabled=False).aexists()
