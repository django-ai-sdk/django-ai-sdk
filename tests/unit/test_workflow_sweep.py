"""A run whose worker died: the sweep marks it, resume queues it, and it continues.

Also a run reaching a worker more than once, and the enqueue that marks a run
nobody will pick up.
"""

from __future__ import annotations

from datetime import timedelta
from io import StringIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from django.core.management import call_command
from django.utils import timezone

from django_ai_sdk.workflows.models import StepRun, WorkflowRun
from django_ai_sdk.workflows.runs import claim_run, enqueue_run, execute_run, open_run
from tests.mocks.workflow import AGENT_GET, ENQUEUE, fake_agent, two_steps

LONG_AGO = timedelta(hours=2)


def sweep(*args: str) -> str:
    out = StringIO()
    call_command("workflows_sweep", *args, stdout=out)
    return out.getvalue().strip()


async def died_after_the_first_step() -> WorkflowRun:
    """A run whose worker completed `first`, started `second`, and was killed."""
    run = await claim_run(str((await open_run(two_steps())).pk))
    assert run is not None
    await StepRun.objects.acreate(
        run=run, sequence=0, step_name="first", status="completed", output="stored"
    )
    await StepRun.objects.acreate(run=run, sequence=1, step_name="second", status="running")
    await WorkflowRun.objects.filter(pk=run.pk).aupdate(heartbeat_at=timezone.now() - LONG_AGO)
    return run


@pytest.mark.django_db(transaction=True)
class TestTheSweep:
    async def test_a_run_with_a_stale_heartbeat_is_interrupted_with_its_step(self):
        run = await died_after_the_first_step()

        assert await sync(sweep) == "1 run interrupted"

        await run.arefresh_from_db()
        assert run.status == WorkflowRun.Status.INTERRUPTED
        assert run.completed_at is not None
        second = await StepRun.objects.aget(run=run, sequence=1)
        assert (second.status, second.detail) == ("failed", "interrupted")
        assert (await StepRun.objects.aget(run=run, sequence=0)).status == "completed"

    async def test_a_live_run_and_a_queued_one_are_left_alone(self):
        live = await claim_run(str((await open_run(two_steps())).pk))
        queued = await open_run(two_steps())
        await WorkflowRun.objects.filter(pk=queued.pk).aupdate(created_at=timezone.now() - LONG_AGO)

        assert await sync(sweep) == "0 runs interrupted"

        assert (await WorkflowRun.objects.aget(pk=live.pk)).status == "running"
        assert (await WorkflowRun.objects.aget(pk=queued.pk)).status == "pending"

    async def test_a_run_that_was_never_queued_is_interrupted(self):
        lost = await open_run(two_steps())
        queued = await open_run(two_steps())
        await WorkflowRun.objects.filter(pk=queued.pk).aupdate(task_id="task-1")
        await WorkflowRun.objects.filter(pk__in=[lost.pk, queued.pk]).aupdate(
            updated_at=timezone.now() - LONG_AGO
        )

        assert await sync(sweep) == "1 run interrupted"

        await lost.arefresh_from_db()
        assert lost.status == WorkflowRun.Status.INTERRUPTED
        assert "never queued" in lost.error
        # A run in the queue's hands waits there, however long the backlog.
        assert (await WorkflowRun.objects.aget(pk=queued.pk)).status == "pending"

    async def test_a_dead_workers_run_reads_as_interrupted_by_its_worker(self):
        run = await died_after_the_first_step()

        await sync(sweep)

        await run.arefresh_from_db()
        assert "heartbeats" in run.error

    async def test_a_dry_run_counts_and_changes_nothing(self):
        run = await died_after_the_first_step()

        assert await sync(sweep, "--dry-run") == "1 run would be interrupted"

        assert (await WorkflowRun.objects.aget(pk=run.pk)).status == "running"

    async def test_the_limit_takes_the_oldest_first(self):
        first = await died_after_the_first_step()
        await died_after_the_first_step()

        assert await sync(sweep, "--limit", "1") == "1 run interrupted"

        assert (await WorkflowRun.objects.aget(pk=first.pk)).status == "interrupted"
        assert await WorkflowRun.objects.filter(status="running").acount() == 1

    async def test_only_the_swept_runs_steps_are_closed(self):
        run = await died_after_the_first_step()
        # An older interrupted run whose running step row is not this sweep's to touch.
        earlier = await WorkflowRun.objects.acreate(status=WorkflowRun.Status.INTERRUPTED)
        await StepRun.objects.acreate(run=earlier, sequence=0, step_name="x", status="running")

        assert await sync(sweep) == "1 run interrupted"

        assert (await StepRun.objects.aget(run=run, sequence=1)).status == "failed"
        assert (await StepRun.objects.aget(run=earlier)).status == "running"

    async def test_stale_after_is_how_old_a_heartbeat_may_be(self):
        await died_after_the_first_step()

        assert await sync(sweep, "--stale-after", "3h") == "0 runs interrupted"


@pytest.mark.django_db(transaction=True)
class TestResumingAnInterruptedRun:
    async def test_it_continues_past_what_completed(self):
        from django_ai_sdk.workflows.services import WorkflowService
        from tests.factories.db import MANAGE_WORKFLOWS, UserFactory, agrant

        owner = await agrant(await UserFactory.acreate(), MANAGE_WORKFLOWS)
        run = await died_after_the_first_step()
        await WorkflowRun.objects.filter(pk=run.pk).aupdate(user=owner)
        await sync(sweep)

        with patch(ENQUEUE) as enqueue:
            await WorkflowService.resume(str(run.pk), user=owner)
        enqueue.assert_called_once_with(str(run.pk))
        model = fake_agent("again")
        with patch(AGENT_GET, AsyncMock(return_value=model)):
            await execute_run(str(run.pk))

        await run.arefresh_from_db()
        assert run.status == WorkflowRun.Status.COMPLETED
        assert run.attempt == 2
        assert run.outputs == {"first": "stored", "second": "again"}
        # Only the step the dead worker did not finish ran again.
        assert model.run.await_count == 1


@pytest.mark.django_db(transaction=True)
class TestOneRunOneWorker:
    @pytest.mark.parametrize("status", ["running", "completed", "failed"])
    async def test_a_run_that_is_not_pending_claims_nothing(self, status):
        # A redelivery of a dead, finished or failed attempt calls no agent.
        run = await open_run(two_steps())
        await WorkflowRun.objects.filter(pk=run.pk).aupdate(status=status, attempt=1)
        model = fake_agent()

        with patch(AGENT_GET, AsyncMock(return_value=model)):
            assert await execute_run(str(run.pk)) is None

        model.run.assert_not_awaited()
        assert (await WorkflowRun.objects.aget(pk=run.pk)).status == status

    async def test_the_claim_counts_attempts_and_beats(self):
        run = await claim_run(str((await open_run(two_steps())).pk))

        assert run is not None
        assert (run.status, run.attempt) == ("running", 1)
        assert run.heartbeat_at is not None
        assert run.started_at is not None


@pytest.mark.django_db
class TestEnqueue:
    def test_it_waits_for_the_transaction_to_commit(self, django_capture_on_commit_callbacks):
        run = WorkflowRun.objects.create()

        with patch(ENQUEUE) as enqueue:
            with django_capture_on_commit_callbacks() as callbacks:
                enqueue_run(run)
            enqueue.assert_not_called()
            assert len(callbacks) == 1
            callbacks[0]()
        enqueue.assert_called_once_with(str(run.pk))

    def test_it_records_the_task_it_queued(self, django_capture_on_commit_callbacks):
        run = WorkflowRun.objects.create()

        with (
            patch(
                "django_ai_sdk.workflows.tasks.execute_workflow",
                MagicMock(enqueue=MagicMock(return_value=MagicMock(id="task-1"))),
            ),
            django_capture_on_commit_callbacks(execute=True),
        ):
            enqueue_run(run)

        run.refresh_from_db()
        assert (run.status, run.task_id) == ("pending", "task-1")

    def test_a_run_that_cannot_be_queued_is_marked_failed(self, django_capture_on_commit_callbacks):
        run = WorkflowRun.objects.create()

        with (
            patch(
                "django_ai_sdk.workflows.tasks.execute_workflow",
                MagicMock(enqueue=MagicMock(side_effect=ConnectionError("broker down"))),
            ),
            django_capture_on_commit_callbacks(execute=True),
        ):
            enqueue_run(run)

        run.refresh_from_db()
        assert run.status == WorkflowRun.Status.FAILED
        assert "broker down" in run.error


async def sync(fn, *args):
    from asgiref.sync import sync_to_async

    return await sync_to_async(fn)(*args)
