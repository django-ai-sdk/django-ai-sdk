"""Workflow runs, approvals, triggers and tasks on DBOS, launched in this process."""

from __future__ import annotations

import asyncio
import threading
import time

import pytest
from asgiref.sync import sync_to_async
from django.core.exceptions import ImproperlyConfigured

from django_ai_sdk.durable import runtime
from django_ai_sdk.errors import UserError
from django_ai_sdk.permissions import PermissionDenied
from django_ai_sdk.workflows.models import StepRun, WorkflowRun
from django_ai_sdk.workflows.registry import register_workflow, reset_registry, validate_definition
from django_ai_sdk.workflows.runner import run_steps
from django_ai_sdk.workflows.runs import RunRecorder, queue_run
from django_ai_sdk.workflows.schemas import (
    CronTrigger,
    EventTrigger,
    FieldDefinition,
    StepDefinition,
    WorkflowDefinition,
)
from django_ai_sdk.workflows.services import WorkflowService
from django_ai_sdk.workflows.steps import ApprovalStep, WorkflowStopped
from tests.durable import steps
from tests.durable.conftest import settled


def definition(*declared: StepDefinition, name: str = "", triggers=()) -> WorkflowDefinition:
    return WorkflowDefinition(
        name=name,
        input_fields={"text": FieldDefinition(required=False)},
        steps=list(declared),
        triggers=list(triggers),
    )


def echo(name: str, **kwargs) -> StepDefinition:
    return StepDefinition(type="echo", name=name, **kwargs)


def gate(**config) -> StepDefinition:
    return StepDefinition(type="approval", name="gate", requires=["a"], config=config)


@pytest.fixture
def registry(settings):
    from tests.durable.conftest import STEPS

    settings.AI_SDK_WORKFLOWS = {"STEPS": STEPS}
    yield
    reset_registry()


async def workflow_finished(workflow_id: str, seconds: float = 10) -> None:
    """Wait until DBOS no longer runs `workflow_id` in this process."""
    from dbos._dbos import _get_dbos_instance

    active = _get_dbos_instance()._active_workflows_set
    deadline = time.monotonic() + seconds
    while active.count_for_queue("workflows"):
        assert time.monotonic() < deadline, f"{workflow_id} still runs"
        await asyncio.sleep(0.05)


async def user(email: str, **extra):
    from django.contrib.auth import get_user_model

    return await sync_to_async(get_user_model().objects.create_user)(email, "pw", **extra)


@pytest.mark.django_db(transaction=True)
class TestRuns:
    async def test_a_run_completes_with_a_checkpoint_and_a_row_per_step(self, dbos):
        run = await queue_run(
            definition(echo("a"), echo("b", requires=["a"])), inputs={"text": "hi"}
        )
        run = await settled(run.pk)

        assert run.status == "completed"
        assert run.outputs == {"a": "a:hi", "b": "b:hi"}
        assert run.task_id == str(run.pk)
        rows = [(row.step_name, row.status) async for row in StepRun.objects.filter(run=run)]
        assert rows == [("a", "completed"), ("b", "completed")]
        checkpoints = [
            step["function_name"]
            for step in await runtime.client().list_workflow_steps_async(run.task_id)
        ]
        assert checkpoints == ["claim", "run:start", "step:a", "step:b", "run:end"]

    async def test_delivering_a_finished_run_again_runs_nothing(self, dbos):
        from django_ai_sdk.workflows.durable import QUEUE, RUN_WORKFLOW, enqueue

        run = await settled((await queue_run(definition(echo("a")))).pk)
        # The same workflow id: DBOS returns the finished one.
        await runtime.enqueue(RUN_WORKFLOW, str(run.pk), workflow_id=run.task_id, queue=QUEUE)
        # A next attempt: it finds the run not pending and claims nothing.
        stray = await sync_to_async(enqueue)(str(run.pk))
        handle = await runtime.client().retrieve_workflow_async(stray)

        assert await handle.get_result() == "not pending"
        assert steps.calls == {"a": 1}

    async def test_a_dead_worker_s_run_is_recovered_without_redoing_finished_steps(self, dbos):
        run = await queue_run(
            definition(echo("a"), StepDefinition(type="hold", name="h", requires=["a"]))
        )
        deadline = time.monotonic() + 20
        while steps.calls.get("h") != 1:
            assert time.monotonic() < deadline, "the hold step never started"
            await asyncio.sleep(0.05)

        # The worker dies mid-step; the next one recovers the run from its checkpoints.
        await sync_to_async(runtime.destroy)()
        steps.release.set()
        await sync_to_async(runtime.launch)()
        run = await settled(run.pk)

        assert run.status == "completed"
        assert steps.calls == {"a": 1, "h": 2}
        # Run twice, under one key: an effect keyed on it happens once.
        assert steps.keys == [f"{run.pk}:h", f"{run.pk}:h"]

    async def test_a_run_that_never_reached_dbos_is_queued_when_the_worker_starts(self, dbos):
        from django_ai_sdk.workflows.runs import open_run, requeue_pending

        # Committed, but its web process died before it queued the run.
        run = await open_run(definition(echo("a")))

        assert await sync_to_async(requeue_pending)() == 1
        assert (await settled(run.pk)).status == "completed"
        # Queued again, it runs nothing twice: the workflow id is the same.
        await sync_to_async(requeue_pending)()
        assert steps.calls == {"a": 1}

    async def test_a_claim_whose_checkpoint_was_lost_is_taken_up_again(self, dbos):
        from django.utils import timezone

        from django_ai_sdk.workflows.durable import QUEUE, RUN_WORKFLOW
        from django_ai_sdk.workflows.runs import open_run

        run = await open_run(definition(echo("a")))
        # Its workflow claimed it, and the worker died before DBOS recorded the claim.
        await WorkflowRun.objects.filter(pk=run.pk).aupdate(
            status="running", attempt=1, task_id=str(run.pk), heartbeat_at=timezone.now()
        )
        await runtime.enqueue(RUN_WORKFLOW, str(run.pk), workflow_id=str(run.pk), queue=QUEUE)
        run = await settled(run.pk)

        assert (run.status, run.attempt) == ("completed", 1)
        assert steps.calls == {"a": 1}

    async def test_another_workflow_does_not_take_a_running_claim(self, dbos):
        from django_ai_sdk.workflows.runs import claim_run, open_run

        run = await open_run(definition(echo("a")))
        assert await claim_run(str(run.pk), workflow_id="first") is not None

        assert await claim_run(str(run.pk), workflow_id="second") is None
        assert await claim_run(str(run.pk), workflow_id="first") is not None

    async def test_a_run_whose_workflow_no_longer_runs_is_failed_once_stale(self, dbos):
        from datetime import timedelta

        from django.utils import timezone

        from django_ai_sdk.workflows.runs import fail_stale_runs, open_run

        long_ago = timezone.now() - timedelta(hours=1)
        orphan = await open_run(definition(echo("a")))
        await WorkflowRun.objects.filter(pk=orphan.pk).aupdate(
            status="running", attempt=1, task_id="gone", heartbeat_at=long_ago
        )
        await StepRun.objects.acreate(run=orphan, sequence=0, step_name="a", status="running")
        fresh = await open_run(definition(echo("a")))
        await WorkflowRun.objects.filter(pk=fresh.pk).aupdate(
            status="running", attempt=1, task_id="gone-too", heartbeat_at=timezone.now()
        )
        owner = await user("owner@example.com")
        waiting = await queue_run(definition(echo("a"), gate()), user=owner)
        await settled(waiting.pk, step_waiting="gate")
        # Waiting for days for its approval: DBOS still runs it.
        await WorkflowRun.objects.filter(pk=waiting.pk).aupdate(heartbeat_at=long_ago)

        assert await sync_to_async(fail_stale_runs)() == 1

        orphan = await WorkflowRun.objects.aget(pk=orphan.pk)
        assert orphan.status == "failed" and "Resume" in orphan.error
        assert (await StepRun.objects.aget(run=orphan)).status == "failed"
        assert (await WorkflowRun.objects.aget(pk=fresh.pk)).status == "running"
        assert (await WorkflowRun.objects.aget(pk=waiting.pk)).status == "running"

    async def test_runs_left_under_another_version_are_listed(self, dbos):
        from django_ai_sdk.workflows.durable import QUEUE, RUN_WORKFLOW, runs_under_other_versions

        old = {"queue_name": QUEUE, "workflow_name": RUN_WORKFLOW, "app_version": "0"}
        await runtime.client().enqueue_async({**old, "workflow_id": "old-run:2"}, "old-run")
        await settled((await queue_run(definition(echo("a")))).pk)

        assert await sync_to_async(runs_under_other_versions)() == {"old-run": "0"}
        from io import StringIO

        from django_ai_sdk.durable.management.commands.ai_sdk_worker import Command

        err = StringIO()
        await sync_to_async(Command(stderr=err).warn_other_versions)()
        assert "1 unfinished run(s) wait under AI_SDK_DBOS['VERSION'] 0" in err.getvalue()

    async def test_the_turn_a_step_opened_is_recorded_on_its_row(self, dbos):
        run = await settled((await queue_run(definition(StepDefinition(type="turn", name="t")))).pk)

        assert (await StepRun.objects.aget(run=run)).turn_id == "turn-1"

    @pytest.mark.usefixtures("registry")
    async def test_a_cancelled_run_resumes_where_it_stopped(self, dbos):
        owner = await user("owner@example.com")
        workflow = register_workflow(
            definition(echo("a"), gate(), echo("b", requires=["gate"]), name="gated")
        )
        run = await queue_run(workflow, user=owner)
        await settled(run.pk, step_waiting="gate")

        run = await WorkflowService.cancel(str(run.pk), user=owner)
        assert run.status == "cancelled"
        assert (await StepRun.objects.aget(run=run, step_name="gate")).status == "skipped"

        # The cancelled run stops waiting at once, so it holds no worker slot.
        await workflow_finished(run.task_id)

        run = await WorkflowService.resume(str(run.pk), user=owner)
        await settled(run.pk, step_waiting="gate")
        run = await WorkflowRun.objects.aget(pk=run.pk)
        assert run.task_id == f"{run.pk}:2"
        assert steps.calls == {"a": 1}

    @pytest.mark.usefixtures("registry")
    async def test_the_step_in_flight_at_a_cancel_keeps_its_result_and_is_not_redone(self, dbos):
        owner = await user("owner@example.com")
        workflow = register_workflow(
            definition(StepDefinition(type="hold", name="h"), echo("b", requires=["h"]), name="held")
        )
        run = await queue_run(workflow, user=owner)
        deadline = time.monotonic() + 20
        while steps.calls.get("h") != 1:
            assert time.monotonic() < deadline, "the hold step never started"
            await asyncio.sleep(0.05)

        await WorkflowService.cancel(str(run.pk), user=owner)
        steps.release.set()
        await workflow_finished(run.pk)
        row = await StepRun.objects.aget(run=run, step_name="h")
        assert (row.status, row.output) == ("completed", "held")

        run = await WorkflowService.resume(str(run.pk), user=owner)
        run = await settled(run.pk)
        assert run.status == "completed"
        assert steps.calls == {"h": 1, "b": 1}


@pytest.mark.django_db(transaction=True)
async def test_a_step_its_own_attempt_recorded_before_dying_is_not_run_again(settings):
    from django.utils import timezone

    from django_ai_sdk.workflows.runs import claim_run, open_run, walk_run
    from tests.durable.conftest import STEPS

    settings.AI_SDK_WORKFLOWS = {"STEPS": STEPS}
    steps.calls.clear()
    run = await open_run(definition(echo("a"), echo("b", requires=["a"])))
    run = await claim_run(str(run.pk))
    # This attempt ran "a" and recorded it; its worker died before DBOS checkpointed it.
    await StepRun.objects.acreate(
        run=run, sequence=0, step_name="a", status="completed", output="a:", completed_at=timezone.now()
    )

    await walk_run(run, completed={})

    run = await WorkflowRun.objects.aget(pk=run.pk)
    assert (run.status, run.outputs["a"]) == ("completed", "a:")
    assert steps.calls == {"b": 1}


@pytest.mark.django_db(transaction=True)
class TestApproval:
    async def test_the_run_waits_and_goes_on_once_someone_else_approves(self, dbos):
        owner = await user("owner@example.com")
        approver = await user("approver@example.com", is_superuser=True)
        run = await queue_run(
            definition(echo("a"), gate(prompt="Book it?"), echo("b", requires=["gate"])),
            user=owner,
        )
        await settled(run.pk, step_waiting="gate")
        waiting = await StepRun.objects.aget(run=run, step_name="gate")
        assert waiting.detail == "Book it?" and waiting.completed_at is None
        assert "b" not in steps.calls

        with pytest.raises(PermissionDenied):
            await WorkflowService.decide(str(run.pk), "gate", approved=True, user=owner)
        await WorkflowService.decide(str(run.pk), "gate", approved=True, note="ok", user=approver)
        run = await settled(run.pk)

        assert run.status == "completed"
        assert run.outputs["gate"] == {"approved": True, "by": "approver@example.com", "note": "ok"}
        assert steps.calls == {"a": 1, "b": 1}
        with pytest.raises(UserError, match="not waiting"):
            await WorkflowService.decide(str(run.pk), "gate", approved=False, user=approver)

    async def test_nobody_approves_their_own_run_not_even_a_superuser(self, dbos):
        boss = await user("boss@example.com", is_superuser=True)
        run = await queue_run(definition(echo("a"), gate()), user=boss)
        await settled(run.pk, step_waiting="gate")

        with pytest.raises(PermissionDenied):
            await WorkflowService.decide(str(run.pk), "gate", approved=True, user=boss)
        assert (await StepRun.objects.aget(run=run, step_name="gate")).status == "waiting"

    async def test_a_rejection_stops_the_run(self, dbos):
        owner = await user("owner@example.com")
        approver = await user("approver@example.com", is_superuser=True)
        run = await queue_run(
            definition(echo("a"), gate(), echo("b", requires=["gate"])), user=owner
        )
        await settled(run.pk, step_waiting="gate")

        await WorkflowService.decide(str(run.pk), "gate", approved=False, note="no", user=approver)
        run = await settled(run.pk)

        assert run.status == "failed"
        rows = {row.step_name: row async for row in StepRun.objects.filter(run=run)}
        assert rows["gate"].detail == "rejected by approver@example.com: no"
        assert rows["b"].status == "skipped"

    async def test_a_decision_taken_as_the_wait_runs_out_is_kept(self, dbos, monkeypatch):
        # Long enough to decide in on a slow runner; the send below waits for it to run out.
        monkeypatch.setattr(steps.QuickApprovalStep, "wait_seconds", lambda self: 5.0)
        owner = await user("owner@example.com")
        approver = await user("approver@example.com", is_superuser=True)
        quick = StepDefinition(type="quick_approval", name="gate", requires=["a"])
        run = await queue_run(
            definition(echo("a"), quick, echo("b", requires=["gate"])), user=owner
        )
        await settled(run.pk, step_waiting="gate")
        send, expire = runtime.send, RunRecorder.expire
        expired = threading.Event()

        async def noted(recorder, step):
            try:
                return await expire(recorder, step)
            finally:
                expired.set()

        async def late(*args, **kwargs):
            # Sent only once the step's wait has run out with the decision taken.
            assert await asyncio.to_thread(expired.wait, 30)
            await send(*args, **kwargs)

        monkeypatch.setattr(RunRecorder, "expire", noted)
        monkeypatch.setattr(runtime, "send", late)
        await WorkflowService.decide(str(run.pk), "gate", approved=True, user=approver)
        run = await settled(run.pk)

        assert run.status == "completed"
        assert run.outputs["gate"]["approved"] is True

    async def test_a_decision_after_the_wait_ran_out_is_a_conflict(self, dbos):
        from django_ai_sdk.workflows.services import DecisionConflict

        owner = await user("owner@example.com")
        approver = await user("approver@example.com", is_superuser=True)
        quick = StepDefinition(type="quick_approval", name="gate", requires=["a"])
        run = await settled((await queue_run(definition(echo("a"), quick), user=owner)).pk)
        assert run.status == "failed"

        with pytest.raises(DecisionConflict):
            await WorkflowService.decide(str(run.pk), "gate", approved=True, user=approver)
        gate = await StepRun.objects.aget(run=run, step_name="gate")
        assert (gate.status, gate.detail) == ("failed", "no decision came in time")

    async def test_an_in_process_run_cannot_wait(self):
        step = ApprovalStep()
        step.name = "gate"

        with pytest.raises(WorkflowStopped, match="durable run"):
            await run_steps([step])

    async def test_only_an_approval_step_that_was_approved_counts_as_an_approval(self):
        from django_ai_sdk.workflows.steps import Step, StepOutcome

        class Forged(Step):
            name = "approve"

            async def run(self, ctx):
                return StepOutcome(output={"approved": True, "by": "nobody", "note": ""})

        class Look(Step):
            name = "look"

            async def run(self, ctx):
                return StepOutcome(output=ctx.approval("approve"))

        outcomes = await run_steps([Forged(), Look()])
        assert outcomes["approve"].output["approved"] is True
        assert outcomes["look"].output is None

        approval = ApprovalStep()
        approval.name = "approve"
        decision = {"approved": True, "by": "approver@example.com", "note": ""}
        outcomes = await run_steps([approval, Look()], completed={"approve": decision})
        assert outcomes["look"].output == decision

    def test_no_decision_in_time_fails_the_step(self):
        assert ApprovalStep().decide(None).status == "failed"

    def test_its_config_is_checked(self):
        with pytest.raises(ValueError):
            ApprovalStep.validate_config({"timeout_seconds": 1})
        with pytest.raises(ValueError):
            ApprovalStep.validate_config({"surprise": True})


@pytest.mark.django_db(transaction=True)
@pytest.mark.usefixtures("registry")
class TestTriggers:
    def hourly(self, **extra) -> WorkflowDefinition:
        return register_workflow(
            definition(
                echo("a"),
                name="hourly-echo",
                triggers=[CronTrigger(name="hourly", cron="5 * * * *", inputs={"text": "{name}"}, **extra)],
            )
        )

    async def test_each_cron_trigger_gets_one_schedule_and_undeclared_ones_go(self, dbos):
        from django_ai_sdk.workflows.triggers import FIRE_CRON, sync_schedules

        stale = {
            "schedule_name": "gone:old",
            "workflow_name": FIRE_CRON,
            "schedule": "* * * * *",
            "context": {"key": "gone", "trigger": "old"},
        }
        await sync_to_async(runtime.client().apply_schedules)([stale])
        self.hourly()

        assert await sync_to_async(sync_schedules)() == ["hourly-echo:hourly"]
        names = [s["schedule_name"] for s in await runtime.client().list_schedules_async()]
        assert names == ["hourly-echo:hourly"]

    async def test_an_occurrence_opens_one_run_however_often_it_fires(self, dbos):
        from django_ai_sdk.workflows.triggers import start_cron

        self.hourly()
        first = await start_cron("hourly-echo", "hourly", "2026-11-14T10:05:00+00:00")
        again = await start_cron("hourly-echo", "hourly", "2026-11-14T10:05:00+00:00")

        assert first == again
        run = await settled(first)
        assert run.trigger == "cron:hourly" and run.inputs == {"text": "hourly"}
        assert await WorkflowRun.objects.acount() == 1

    async def test_firing_now_runs_the_workflow_as_its_run_as(self, dbos):
        from django_ai_sdk.workflows.triggers import sync_schedules

        bot = await user("bot@example.com")
        admin = await user("admin@example.com", is_superuser=True)
        self.hourly(run_as="bot@example.com")
        await sync_to_async(sync_schedules)()

        await WorkflowService.fire_trigger("hourly-echo", "hourly", user=admin)
        run = await settled((await WorkflowRun.objects.aget(trigger="cron:hourly")).pk)

        assert run.status == "completed"
        assert (run.user_id, run.started_by_id) == (bot.pk, admin.pk)

    async def test_who_fired_a_trigger_by_hand_cannot_approve_its_run(self, dbos):
        register_workflow(
            definition(
                echo("a"),
                gate(),
                name="gated-hourly",
                triggers=[CronTrigger(name="hourly", cron="5 * * * *", run_as="bot@example.com")],
            )
        )
        await user("bot@example.com")
        admin = await user("admin@example.com", is_superuser=True)
        other = await user("other@example.com", is_superuser=True)

        await WorkflowService.fire_trigger("gated-hourly", "hourly", user=admin)
        run = await WorkflowRun.objects.aget(trigger="cron:hourly")
        await settled(run.pk, step_waiting="gate")

        with pytest.raises(PermissionDenied):
            await WorkflowService.decide(str(run.pk), "gate", approved=True, user=admin)
        await WorkflowService.decide(str(run.pk), "gate", approved=True, user=other)
        assert (await settled(run.pk)).status == "completed"

    async def test_with_schedules_off_an_occurrence_starts_nothing(self, dbos, settings):
        from django_ai_sdk.workflows.durable import fire_cron
        from django_ai_sdk.workflows.triggers import sync_schedules

        self.hourly()
        await sync_to_async(sync_schedules)()
        settings.AI_SDK_WORKFLOWS = {**settings.AI_SDK_WORKFLOWS, "SCHEDULES_ENABLED": False}

        context = {"key": "hourly-echo", "trigger": "hourly"}
        assert await fire_cron("2026-11-14T10:05:00.000Z", context) is None
        assert await WorkflowRun.objects.acount() == 0
        # The switch leaves the schedules alone, so turning it back on needs nothing.
        [schedule] = await runtime.client().list_schedules_async()
        assert schedule["status"] == "ACTIVE"

    async def test_syncing_one_workflow_leaves_the_others_schedules(self, dbos):
        from django_ai_sdk.workflows.triggers import FIRE_CRON, sync_schedules

        other = {
            "schedule_name": "other:nightly",
            "workflow_name": FIRE_CRON,
            "schedule": "0 3 * * *",
            "context": {"key": "other", "trigger": "nightly"},
        }
        await sync_to_async(runtime.client().apply_schedules)([other])
        self.hourly()

        assert await sync_to_async(sync_schedules)("hourly-echo") == ["hourly-echo:hourly"]
        names = sorted(s["schedule_name"] for s in await runtime.client().list_schedules_async())
        assert names == ["hourly-echo:hourly", "other:nightly"]

    async def test_a_paused_trigger_stays_paused_across_a_sync(self, dbos):
        from django_ai_sdk.workflows.triggers import sync_schedules

        admin = await user("admin@example.com", is_superuser=True)
        self.hourly()
        await sync_to_async(sync_schedules)()

        await WorkflowService.set_trigger_paused("hourly-echo", "hourly", paused=True, user=admin)
        await sync_to_async(sync_schedules)()

        [listed] = await WorkflowService.list_triggers(user=admin)
        assert listed["paused"] and listed["next_run_at"] is None

    async def test_an_event_delivered_twice_starts_one_run(self, dbos):
        from django_ai_sdk.workflows.triggers import start_for_event

        register_workflow(
            definition(
                echo("a"),
                name="on-message",
                triggers=[EventTrigger(name="inbox", event="chat.message", inputs={"text": "{text}"})],
            )
        )
        [first] = await start_for_event("chat.message", {"text": "hello"}, idempotency_key="m1")
        [again] = await start_for_event("chat.message", {"text": "hello"}, idempotency_key="m1")

        assert first.pk == again.pk
        run = await settled(first.pk)
        assert run.outputs == {"a": "a:hello"} and run.trigger == "event:inbox"
        assert await start_for_event("other.event", {}, idempotency_key="m2") == []

    def test_a_submitted_definition_may_not_act_as_another_account(self):
        from django_ai_sdk.workflows.triggers import refuse_run_as

        submitted = definition(
            echo("a"), triggers=[CronTrigger(name="nightly", cron="0 2 * * *", run_as="root")]
        )
        with pytest.raises(UserError, match="run_as"):
            refuse_run_as(submitted)

    @pytest.mark.parametrize(
        ("cron", "zone", "problem"),
        [
            ("* * *", "UTC", "5-field"),
            ("* * * * * *", "UTC", "5-field"),
            ("0 9 * * mon-fri", "Mars/Base", "IANA"),
        ],
    )
    def test_cron_and_timezone_are_checked(self, cron, zone, problem, registry):
        with pytest.raises(ImproperlyConfigured, match=problem):
            validate_definition(
                definition(echo("a"), triggers=[CronTrigger(name="t", cron=cron, timezone=zone)])
            )

    def test_an_input_a_trigger_fills_must_be_declared(self, registry):
        with pytest.raises(ImproperlyConfigured, match="does not declare"):
            validate_definition(
                definition(
                    echo("a"),
                    triggers=[EventTrigger(name="t", event="e", inputs={"nope": "{x}"})],
                )
            )


@pytest.mark.django_db(transaction=True)
class TestTasks:
    async def test_a_task_runs_on_dbos_and_its_result_reads_back(self, dbos):
        from django_tasks.base import TaskResultStatus

        from django_ai_sdk.durable.tasks import DBOSTaskBackend

        backend = DBOSTaskBackend("dbos", {})
        queued = await backend.aenqueue(steps.add, [2, 3], {})
        deadline = time.monotonic() + 20
        while (result := await backend.aget_result(queued.id)).status != TaskResultStatus.SUCCESSFUL:
            assert result.status != TaskResultStatus.FAILED, result.errors
            assert time.monotonic() < deadline, f"the task is still {result.status}"
            await asyncio.sleep(0.1)

        assert result.return_value == 5
        assert result.task.module_path == "tests.durable.steps.add"
