"""The claim, the lease, and audience fan-out, with `_enqueue_all` patched out.

What is under test is which rows get written, not whether django-tasks accepts them.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from django_ai_sdk.automations import Audience, Automation
from django_ai_sdk.automations.models import AutomationRun, AutomationState
from django_ai_sdk.automations.registry import register, reset_registry
from django_ai_sdk.automations.runner import claim, ensure_state, release, tick

NOW = datetime(2026, 8, 16, 12, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _clean_registry():
    from django_ai_sdk.automations import runner

    reset_registry()
    runner._warned_disabled = False
    yield
    reset_registry()


@pytest.fixture(autouse=True)
def _no_queue():
    """Nothing here needs a worker; assert on rows, not on the backend."""
    with patch(
        "django_ai_sdk.automations.runner._enqueue_all", new=AsyncMock(return_value=None)
    ) as enqueue:
        yield enqueue


def declare(name="example", **attrs) -> Automation:
    defaults = {
        "name": name,
        "cron": "0 9 * * *",
        "workflow": "some-workflow",
    }
    cls = type("Example", (Automation,), {**defaults, **attrs})
    register(cls)
    return cls()


@pytest.mark.django_db(transaction=True)
class TestClaim:
    async def test_two_ticks_racing_produce_exactly_one_winner(self):
        automation = declare(cron="0 * * * *")
        state, _ = await ensure_state(automation, now=NOW)
        # Make it due.
        await AutomationState.objects.filter(id=state.id).aupdate(next_run_at=NOW)
        state = await AutomationState.objects.aget(id=state.id)

        first = await claim(automation, state, now=NOW)
        second = await claim(automation, state, now=NOW)

        assert (first, second) == (True, False)
        # And the cursor advanced once, not twice — a double advance would silently
        # skip a whole window.
        state = await AutomationState.objects.aget(id=state.id)
        assert state.next_run_at == NOW + timedelta(hours=1)

    async def test_a_held_lease_blocks_the_next_tick(self):
        automation = declare()
        state, _ = await ensure_state(automation, now=NOW)
        await AutomationState.objects.filter(id=state.id).aupdate(
            next_run_at=NOW, locked_until=NOW + timedelta(minutes=30)
        )
        state = await AutomationState.objects.aget(id=state.id)

        assert await claim(automation, state, now=NOW) is False

    async def test_allow_overlap_ignores_the_lease(self):
        automation = declare(allow_overlap=True)
        state, _ = await ensure_state(automation, now=NOW)
        await AutomationState.objects.filter(id=state.id).aupdate(
            next_run_at=NOW, locked_until=NOW + timedelta(minutes=30)
        )
        state = await AutomationState.objects.aget(id=state.id)

        assert await claim(automation, state, now=NOW) is True

    async def test_an_expired_lease_is_reclaimed(self):
        # This is the crash-recovery path: a worker that died holding the lease must
        # not wedge the automation forever, and no sweeper process should be needed.
        automation = declare()
        state, _ = await ensure_state(automation, now=NOW)
        await AutomationState.objects.filter(id=state.id).aupdate(
            next_run_at=NOW, locked_until=NOW - timedelta(seconds=1)
        )
        state = await AutomationState.objects.aget(id=state.id)

        assert await claim(automation, state, now=NOW) is True

    async def test_release_clears_the_lease_and_records_success(self):
        automation = declare()
        state, _ = await ensure_state(automation, now=NOW)
        await claim(automation, state, now=NOW, force=True)

        await release(state, succeeded_at=NOW)

        state = await AutomationState.objects.aget(id=state.id)
        assert state.locked_until is None
        assert state.last_success_at == NOW


@pytest.mark.django_db(transaction=True)
class TestBootstrap:
    async def test_a_new_automation_is_scheduled_forward_not_backward(self):
        # Deploying an automation must not make it fire immediately just because its
        # schedule matched some moment before the deploy.
        automation = declare()
        state, created = await ensure_state(automation, now=NOW)

        assert created is True
        assert state.next_run_at > NOW

    async def test_the_first_tick_dispatches_nothing(self):
        declare()
        [result] = await tick(now=NOW)
        assert result.runs == []
        assert result.reason == "not due"


@pytest.mark.django_db(transaction=True)
class TestDueness:
    async def test_a_due_automation_dispatches_one_run(self):
        declare()
        await tick(now=NOW)
        await AutomationState.objects.aupdate(next_run_at=NOW)

        [result] = await tick(now=NOW)

        assert len(result.runs) == 1
        assert result.runs[0].status == AutomationRun.Status.PENDING

    async def test_disabled_in_settings_is_not_dispatched(self, settings):
        settings.AI_SDK_AUTOMATIONS = {"example": {"ENABLED": False}}
        declare()
        await AutomationState.objects.aupdate(next_run_at=NOW)

        [result] = await tick(now=NOW)

        assert result.runs == []
        assert "settings" in result.reason

    async def test_the_global_kill_switch_stops_everything(self, settings):
        settings.AI_SDK_AUTOMATIONS_ENABLED = False
        declare()
        assert await tick(now=NOW) == []

    async def test_naming_one_automation_and_forcing_it_bypasses_the_kill_switch(self, settings):
        # A person asking for this run is not the schedule firing, and it is what the
        # "run now" button does — the two paths must not disagree.
        settings.AI_SDK_AUTOMATIONS_ENABLED = False
        declare()

        [result] = await tick(now=NOW, only="example", force=True)

        assert len(result.runs) == 1

    async def test_a_blanket_force_still_respects_the_kill_switch(self, settings):
        # Otherwise one mistyped command sets a deliberately-quiet environment going.
        settings.AI_SDK_AUTOMATIONS_ENABLED = False
        declare()

        assert await tick(now=NOW, force=True) == []

    async def test_the_kill_switch_notice_is_logged_once_per_process(self, settings, caplog):
        # `run_automations --loop` calls tick() every minute; repeating the same line
        # is how a useful message turns into noise nobody reads.
        from django_ai_sdk.automations import runner

        settings.AI_SDK_AUTOMATIONS_ENABLED = False
        declare()

        with caplog.at_level("INFO", logger="django_ai_sdk.automations.runner"):
            for _ in range(3):
                await tick(now=NOW)

        assert caplog.text.count("AI_SDK_AUTOMATIONS_ENABLED is False") == 1

    async def test_dry_run_writes_no_runs(self):
        declare()
        await tick(now=NOW)
        await AutomationState.objects.aupdate(next_run_at=NOW)

        await tick(now=NOW, dry_run=True)

        assert await AutomationRun.objects.acount() == 0

    async def test_one_broken_automation_does_not_stop_the_rest(self):
        class Broken(Automation):
            name = "broken"
            cron = "0 9 * * *"
            workflow = "some-workflow"

            def get_schedule(self):
                raise RuntimeError("boom")

        declare("healthy")
        # Registered past validation, the way a schedule that breaks only at runtime
        # would be — the tick must isolate it.
        from django_ai_sdk.automations import registry as registry_module

        registry_module._registry["broken"] = Broken()

        results = {r.name: r for r in await tick(now=NOW)}

        assert "see logs" in results["broken"].reason
        assert results["healthy"].reason == "not due"


@pytest.mark.django_db(transaction=True)
class TestBehindSchedule:
    # Fires at :30 past every hour, so a stale cursor lands on a window that is
    # distinguishable from `now`.
    HOURLY = "30 * * * *"

    async def test_it_runs_once_for_the_window_it_missed(self):
        declare(cron=self.HOURLY)
        await tick(now=NOW)
        await AutomationState.objects.aupdate(next_run_at=NOW - timedelta(hours=4, minutes=30))

        [result] = await tick(now=NOW)

        pending = [r for r in result.runs if r.status == AutomationRun.Status.PENDING]
        assert len(pending) == 1
        # Stamped with the window it was for, not with `now` — otherwise the run
        # history loses which occurrence this was.
        assert pending[0].scheduled_for == NOW - timedelta(hours=4, minutes=30)

    async def test_the_claim_resumes_at_the_next_future_occurrence(self):
        declare(cron=self.HOURLY)
        await tick(now=NOW)
        await AutomationState.objects.aupdate(next_run_at=NOW - timedelta(hours=4, minutes=30))

        await tick(now=NOW)

        state = await AutomationState.objects.aget(name="example")
        assert state.next_run_at == NOW + timedelta(minutes=30)


@pytest.mark.django_db(transaction=True)
class TestDryRun:
    async def test_it_writes_nothing_at_all(self):
        declare()

        [result] = await tick(now=NOW, dry_run=True)

        assert result.runs == []
        assert not await AutomationState.objects.aexists()
        assert not await AutomationRun.objects.aexists()

    async def test_it_reports_a_due_automation_without_claiming_it(self):
        declare()
        await tick(now=NOW)
        await AutomationState.objects.aupdate(next_run_at=NOW)

        [result] = await tick(now=NOW, dry_run=True)

        assert "would dispatch" in result.reason
        state = await AutomationState.objects.aget(name="example")
        assert state.locked_until is None
        assert state.next_run_at == NOW


@pytest.mark.django_db(transaction=True)
class TestAudience:
    async def test_the_app_principal_yields_one_userless_run(self):
        declare()
        await tick(now=NOW)
        await AutomationState.objects.aupdate(next_run_at=NOW)

        [result] = await tick(now=NOW)

        assert len(result.runs) == 1
        assert result.runs[0].user_id is None

    async def test_subscribed_users_fan_out_one_run_each_sharing_a_dispatch_id(self):
        from django_ai_sdk.automations.models import AutomationSubscription
        from tests.factories.db import UserFactory

        for _ in range(3):
            user = await UserFactory.acreate()
            await AutomationSubscription.objects.acreate(name="example", user=user, enabled=True)

        declare(audience=Audience.SUBSCRIBED)
        await tick(now=NOW)
        await AutomationState.objects.aupdate(next_run_at=NOW)

        [result] = await tick(now=NOW)

        assert len(result.runs) == 3
        # One fan-out is one logical dispatch, so its runs have to be groupable.
        assert len({r.dispatch_id for r in result.runs}) == 1
        assert len({r.user_id for r in result.runs}) == 3

    async def test_an_empty_audience_writes_a_skipped_run_rather_than_nothing(self):
        # Silence is indistinguishable from a dead scheduler. A row with a reason is
        # the whole point.
        declare(audience=Audience.SUBSCRIBED)
        await tick(now=NOW)
        await AutomationState.objects.aupdate(next_run_at=NOW)

        [result] = await tick(now=NOW)

        assert len(result.runs) == 1
        assert result.runs[0].status == AutomationRun.Status.SKIPPED
        assert "nobody" in result.runs[0].skip_reason

    async def test_a_resolver_that_raises_is_treated_as_empty(self):
        class Exploding:
            async def resolve(self, automation):
                raise RuntimeError("upstream is down")

            def describe(self):
                return "exploding"

        declare(audience=Exploding())
        await tick(now=NOW)
        await AutomationState.objects.aupdate(next_run_at=NOW)

        [result] = await tick(now=NOW)

        assert result.runs[0].status == AutomationRun.Status.SKIPPED

    async def test_the_lease_is_released_when_the_audience_is_empty(self):
        # Otherwise one empty audience wedges the automation for a whole lease period.
        declare(audience=Audience.SUBSCRIBED)
        await tick(now=NOW)
        await AutomationState.objects.aupdate(next_run_at=NOW)

        await tick(now=NOW)

        state = await AutomationState.objects.aget(name="example")
        assert state.locked_until is None


@pytest.mark.django_db(transaction=True)
class TestTheLeaseSurvivesAFanOut:
    """One claim covers the whole fan-out, so one finisher must not free it.

    allow_overlap=False promises a second copy of a payload cannot start. With a
    subscribed audience the claim happens once and N runs follow, so releasing on the
    first completion broke that promise for every audience except APP.
    """

    async def _fan_out(self, subscribers=3):
        from django_ai_sdk.automations.models import AutomationSubscription
        from tests.factories.db import UserFactory

        for _ in range(subscribers):
            user = await UserFactory.acreate()
            await AutomationSubscription.objects.acreate(name="example", user=user, enabled=True)

        declare(audience=Audience.SUBSCRIBED, allow_overlap=False)
        await tick(now=NOW)
        await AutomationState.objects.aupdate(next_run_at=NOW)
        [result] = await tick(now=NOW)
        return result.runs

    async def _finish_one(self, run, status=AutomationRun.Status.SUCCEEDED):
        """End one run the way run_automation does: terminal status, then the lease."""
        from django_ai_sdk.automations.tasks import _finish, _release_if_dispatch_is_done

        loaded = await AutomationRun.objects.select_related("state").aget(id=run.id)
        await _finish(loaded, status=status, output={"ok": True})
        await _release_if_dispatch_is_done(loaded)

    async def test_the_first_finisher_does_not_release_the_lease(self):
        runs = await self._fan_out()
        await self._finish_one(runs[0])

        state = await AutomationState.objects.aget(name="example")
        assert state.locked_until is not None

    async def test_the_last_finisher_releases_it(self):
        runs = await self._fan_out()
        for run in runs:
            await self._finish_one(run)

        state = await AutomationState.objects.aget(name="example")
        assert state.locked_until is None

    async def test_a_half_finished_fan_out_cannot_be_re_claimed(self):
        runs = await self._fan_out()
        await self._finish_one(runs[0])

        automation = declare(audience=Audience.SUBSCRIBED, allow_overlap=False)
        state = await AutomationState.objects.aget(name="example")
        await AutomationState.objects.filter(name="example").aupdate(next_run_at=NOW)
        state.next_run_at = NOW

        assert await claim(automation, state, now=NOW) is False

    async def test_one_failure_among_many_still_advances_the_window(self):
        # last_success_at is per-automation while the audience is per-user, so one
        # principal failing must not make every other principal reprocess its window.
        runs = await self._fan_out()
        await self._finish_one(runs[0], status=AutomationRun.Status.FAILED)
        await self._finish_one(runs[1])
        await self._finish_one(runs[2])

        state = await AutomationState.objects.aget(name="example")
        assert state.last_success_at is not None

    async def test_a_fan_out_that_wholly_fails_does_not_advance_the_window(self):
        runs = await self._fan_out()
        for run in runs:
            await self._finish_one(run, status=AutomationRun.Status.FAILED)

        state = await AutomationState.objects.aget(name="example")
        assert state.locked_until is None
        assert state.last_success_at is None


@pytest.mark.django_db(transaction=True)
class TestALeaseIsOnlyDroppedByItsOwner:
    """Releasing is conditional on the lease value, so a straggler cannot free a new one."""

    async def test_a_straggler_does_not_free_a_lease_the_next_tick_took(self):
        automation = declare()
        state, _ = await ensure_state(automation, now=NOW)
        await AutomationState.objects.filter(id=state.id).aupdate(next_run_at=NOW)
        state = await AutomationState.objects.aget(id=state.id)
        assert await claim(automation, state, now=NOW)

        # The next tick claims after the first lease is gone, and takes its own.
        later = NOW + timedelta(hours=1)
        await AutomationState.objects.filter(id=state.id).aupdate(
            locked_until=None, next_run_at=later
        )
        fresh = await AutomationState.objects.aget(id=state.id)
        assert await claim(automation, fresh, now=later)

        # The straggler still holds the state it read when its own dispatch began.
        await release(state)

        current = await AutomationState.objects.aget(id=state.id)
        assert current.locked_until == fresh.locked_until

    async def test_a_success_still_advances_the_window(self):
        automation = declare()
        state, _ = await ensure_state(automation, now=NOW)
        await AutomationState.objects.filter(id=state.id).aupdate(next_run_at=NOW)
        state = await AutomationState.objects.aget(id=state.id)
        await claim(automation, state, now=NOW)

        await release(state, succeeded_at=NOW)

        current = await AutomationState.objects.aget(id=state.id)
        assert current.last_success_at == NOW
        assert current.locked_until is None


@pytest.mark.django_db(transaction=True)
class TestACancelledRunDoesNotHoldTheLease:
    """A task killed mid-run leaves its row RUNNING; the lease must not wait for expiry."""

    async def test_the_lease_is_dropped_even_though_the_row_is_not_terminal(self):
        from django_ai_sdk.automations.tasks import _release_if_dispatch_is_done

        declare()
        await tick(now=NOW)
        await AutomationState.objects.aupdate(next_run_at=NOW)
        [result] = await tick(now=NOW)
        [run] = result.runs
        await AutomationRun.objects.filter(id=run.id).aupdate(status=AutomationRun.Status.RUNNING)

        loaded = await AutomationRun.objects.select_related("state").aget(id=run.id)
        await _release_if_dispatch_is_done(loaded)

        state = await AutomationState.objects.aget(name="example")
        assert state.locked_until is None

    async def test_a_sibling_still_running_keeps_it(self):
        from django_ai_sdk.automations.tasks import _release_if_dispatch_is_done

        declare()
        await tick(now=NOW)
        await AutomationState.objects.aupdate(next_run_at=NOW)
        [result] = await tick(now=NOW)
        [run] = result.runs
        sibling = await AutomationRun.objects.acreate(
            name=run.name,
            state=run.state,
            dispatch_id=run.dispatch_id,
            status=AutomationRun.Status.RUNNING,
            scheduled_for=run.scheduled_for,
        )
        assert sibling.pk

        loaded = await AutomationRun.objects.select_related("state").aget(id=run.id)
        await _release_if_dispatch_is_done(loaded)

        state = await AutomationState.objects.aget(name="example")
        assert state.locked_until is not None


@pytest.mark.django_db(transaction=True)
class TestEnqueueWaitsForTheCommit:
    """A worker must never be handed a run id that is still inside an open transaction."""

    @pytest.fixture(autouse=True)
    def _no_queue(self):
        # This class exercises the real _enqueue_all.
        yield

    def test_nothing_is_enqueued_until_the_transaction_commits(self):
        from asgiref.sync import async_to_sync
        from django.db import transaction

        from django_ai_sdk.automations.runner import _enqueue_all

        automation = declare()
        state, _ = async_to_sync(ensure_state)(automation, now=NOW)
        run = AutomationRun.objects.create(name=automation.name, state=state, scheduled_for=NOW)

        with patch("django_ai_sdk.automations.tasks.execute_automation") as task:
            enqueue = task.enqueue
            enqueue.return_value.id = "task-1"
            with transaction.atomic():
                async_to_sync(_enqueue_all)([run])
                assert enqueue.call_count == 0
            assert enqueue.call_count == 1

        run.refresh_from_db()
        assert run.task_id == "task-1"

    def test_a_backend_that_refuses_fails_the_run(self):
        from asgiref.sync import async_to_sync

        from django_ai_sdk.automations.runner import _enqueue_all

        automation = declare()
        state, _ = async_to_sync(ensure_state)(automation, now=NOW)
        run = AutomationRun.objects.create(name=automation.name, state=state, scheduled_for=NOW)

        with patch("django_ai_sdk.automations.tasks.execute_automation") as task:
            task.enqueue.side_effect = RuntimeError("no backend")
            async_to_sync(_enqueue_all)([run])

        run.refresh_from_db()
        assert run.status == AutomationRun.Status.FAILED
        assert "task backend" in run.error


@pytest.mark.django_db(transaction=True)
class TestAManualDispatchSaysSo:
    """A run records the trigger it was dispatched by, including when it is skipped."""

    async def test_a_manual_run_with_nobody_to_run_for_is_still_manual(self):
        from django_ai_sdk.automations.runner import run_now

        declare(audience=Audience.SUBSCRIBED)

        [run] = await run_now("example")

        assert run.status == AutomationRun.Status.SKIPPED
        assert run.trigger == AutomationRun.Trigger.MANUAL

    async def test_a_second_manual_run_reports_the_lease_by_name(self):
        from django_ai_sdk.automations.runner import AutomationBusy, run_now

        declare()
        await run_now("example")

        with pytest.raises(AutomationBusy):
            await run_now("example")
