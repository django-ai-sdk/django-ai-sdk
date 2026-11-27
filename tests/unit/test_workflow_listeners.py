import pytest
from django.test import override_settings
from pydantic import BaseModel

from django_ai_sdk.workflows.listeners import RunListener, get_listener_registry
from django_ai_sdk.workflows.models import WorkflowRun, StepRun
from django_ai_sdk.workflows.runner import run_steps
from django_ai_sdk.workflows.runs import RunRecorder
from django_ai_sdk.workflows.steps import RunSuperseded, StepOutcome, StepAlreadyRecorded, WorkflowContext
from tests.mocks.workflow import FakeStep

CTX = WorkflowContext()


@pytest.mark.django_db(transaction=True)
class TestRunRecorder:
    async def _recorder(self, *names):
        run = await WorkflowRun.objects.acreate(status=WorkflowRun.Status.RUNNING, attempt=1)
        steps = [FakeStep(name) for name in names]
        return run, RunRecorder(run, steps), steps

    async def test_start_then_end_leaves_one_completed_row(self):
        run, recorder, (ocr,) = await self._recorder("ocr")

        await recorder.on_step_start(CTX, ocr)
        row = await run.steps.aget(step_name="ocr")
        assert row.status == StepRun.Status.RUNNING

        await recorder.on_step_end(CTX, ocr, StepOutcome(output="the text"))
        await row.arefresh_from_db()

        assert row.status == StepRun.Status.COMPLETED
        assert row.output == "the text"
        assert row.error == ""
        assert row.completed_at is not None

    async def test_a_failed_outcome_is_recorded_on_the_row(self):
        run, recorder, (extract,) = await self._recorder("extract")

        await recorder.on_step_start(CTX, extract)
        await recorder.on_step_end(
            CTX, extract, StepOutcome(status="failed", detail="nothing readable")
        )

        row = await run.steps.aget(step_name="extract")
        assert row.status == StepRun.Status.FAILED
        assert row.error == "nothing readable"

    async def test_a_skipped_outcome_is_a_row_carrying_its_reason(self):
        run, recorder, (extract,) = await self._recorder("extract")

        await recorder.on_step_end(
            CTX, extract, StepOutcome(status="skipped", detail="not an invoice")
        )

        row = await run.steps.aget(step_name="extract")
        assert row.status == StepRun.Status.SKIPPED
        assert row.detail == "not an invoice"
        assert row.error == ""

    async def test_a_pydantic_output_is_stored_json_safe(self):
        class Lines(BaseModel):
            items: list[str]

        run, recorder, (extract,) = await self._recorder("extract")

        await recorder.on_step_start(CTX, extract)
        await recorder.on_step_end(CTX, extract, StepOutcome(output=Lines(items=["a"])))

        row = await run.steps.aget(step_name="extract")
        assert row.output == {"items": ["a"]}

    async def test_a_run_records_every_step_it_walked(self):
        run = await WorkflowRun.objects.acreate(status=WorkflowRun.Status.RUNNING)
        steps = [
            FakeStep("ocr", outcome=StepOutcome(output="t")),
            FakeStep("triage", requires=("ocr",)),
        ]

        await run_steps(steps, recorder=RunRecorder(run, steps))

        rows = {row.step_name: row.status async for row in run.steps.all()}
        assert rows == {
            "ocr": StepRun.Status.COMPLETED,
            "triage": StepRun.Status.COMPLETED,
        }

    async def test_every_step_refreshes_the_runs_heartbeat(self):
        run, recorder, (ocr,) = await self._recorder("ocr")

        await recorder.on_step_start(CTX, ocr)
        await run.arefresh_from_db()
        started = run.heartbeat_at
        await recorder.on_step_end(CTX, ocr, StepOutcome(output="t"))
        await run.arefresh_from_db()

        assert started is not None
        assert run.heartbeat_at >= started

    async def test_the_recorder_writes_the_runs_ending(self):
        run, recorder, _steps = await self._recorder("ocr")
        await WorkflowRun.objects.filter(pk=run.pk).aupdate(status=WorkflowRun.Status.RUNNING)

        await recorder.on_run_end(CTX, RuntimeError("provider 503"))
        await run.arefresh_from_db()

        assert run.status == WorkflowRun.Status.FAILED
        assert run.error == "provider 503"
        assert recorder.closed

    async def test_a_superseded_attempt_leaves_the_ending_to_the_newer_one(self):
        run, recorder, _steps = await self._recorder("ocr")
        await WorkflowRun.objects.filter(pk=run.pk).aupdate(status=WorkflowRun.Status.RUNNING)

        await recorder.on_run_end(CTX, RunSuperseded("taken over"))
        await run.arefresh_from_db()

        assert run.status == WorkflowRun.Status.RUNNING
        assert not recorder.closed


@pytest.mark.django_db(transaction=True)
class TestTheRecorderHoldsTheRun:
    async def _recorder(self, *names, attempt=1):
        run = await WorkflowRun.objects.acreate(status=WorkflowRun.Status.RUNNING, attempt=attempt)
        steps = [FakeStep(name) for name in names]
        return run, RunRecorder(run, steps), steps

    async def test_a_step_row_an_earlier_attempt_left_running_is_taken_over(self):
        run, recorder, (ocr,) = await self._recorder("ocr")
        await StepRun.objects.acreate(
            run=run, sequence=0, step_name="ocr", status=StepRun.Status.RUNNING
        )

        await recorder.on_step_start(CTX, ocr)

        assert await run.steps.acount() == 1

    async def test_two_independent_runs_do_not_contend(self):
        run_a, recorder_a, (ocr_a,) = await self._recorder("ocr")
        run_b, recorder_b, (ocr_b,) = await self._recorder("ocr")

        await recorder_a.on_step_start(CTX, ocr_a)
        await recorder_b.on_step_start(CTX, ocr_b)

        assert await run_a.steps.acount() == 1
        assert await run_b.steps.acount() == 1

    async def test_a_failed_step_is_reclaimed_on_a_resume(self):
        run, recorder, (ocr,) = await self._recorder("ocr")

        await recorder.on_step_start(CTX, ocr)
        await recorder.on_step_end(CTX, ocr, StepOutcome(status="failed", detail="503"))

        await recorder.on_step_start(CTX, ocr)

        row = await run.steps.aget(step_name="ocr")
        assert row.status == StepRun.Status.RUNNING

    async def test_a_completed_step_is_never_reclaimed(self):
        run, recorder, (ocr,) = await self._recorder("ocr")

        await recorder.on_step_start(CTX, ocr)
        await recorder.on_step_end(CTX, ocr, StepOutcome(output="the text"))

        with pytest.raises(StepAlreadyRecorded) as recorded:
            await recorder.on_step_start(CTX, ocr)

        assert recorded.value.outcome.output == "the text"
        assert (await run.steps.aget(step_name="ocr")).status == StepRun.Status.COMPLETED

    async def test_a_cancelled_run_stops_its_worker_before_it_starts_a_step(self):
        run, recorder, (ocr,) = await self._recorder("ocr")
        await WorkflowRun.objects.filter(pk=run.pk).aupdate(
            status=WorkflowRun.Status.CANCELLED
        )

        with pytest.raises(RunSuperseded):
            await recorder.on_step_start(CTX, ocr)

        assert await run.steps.acount() == 0

    async def test_a_step_in_flight_when_its_run_was_cancelled_keeps_its_outcome(self):
        run, recorder, (ocr,) = await self._recorder("ocr")
        await recorder.on_step_start(CTX, ocr)
        await WorkflowRun.objects.filter(pk=run.pk).aupdate(
            status=WorkflowRun.Status.CANCELLED
        )

        with pytest.raises(RunSuperseded):
            await recorder.on_step_end(CTX, ocr, StepOutcome(output="late"))

        row = await run.steps.aget(step_name="ocr")
        assert (row.status, row.output) == (StepRun.Status.COMPLETED, "late")

    async def test_a_later_attempt_s_run_gets_nothing_from_an_earlier_one(self):
        run, recorder, (ocr,) = await self._recorder("ocr")
        await WorkflowRun.objects.filter(pk=run.pk).aupdate(attempt=2)

        with pytest.raises(RunSuperseded):
            await recorder.on_step_end(CTX, ocr, StepOutcome(output="late"))

        assert await run.steps.acount() == 0

    async def test_a_later_attempt_stops_the_earlier_one(self):
        run, recorder, (ocr,) = await self._recorder("ocr", attempt=1)
        await WorkflowRun.objects.filter(pk=run.pk).aupdate(attempt=2)

        with pytest.raises(RunSuperseded):
            await recorder.on_step_start(CTX, ocr)


@pytest.mark.django_db
def test_an_existing_step_row_does_not_break_the_callers_transaction():
    """The claim inserts with ON CONFLICT DO NOTHING: on Postgres an IntegrityError
    would abort the transaction the caller is in."""
    from asgiref.sync import async_to_sync

    run = WorkflowRun.objects.create(status=WorkflowRun.Status.RUNNING)
    ocr = FakeStep("ocr")
    StepRun.objects.create(run=run, sequence=0, step_name="ocr", status=StepRun.Status.FAILED)

    async_to_sync(RunRecorder(run, [ocr]).on_step_start)(CTX, ocr)

    assert StepRun.objects.get(run=run).status == StepRun.Status.RUNNING


class Loud(RunListener):
    description = "says so"


class NotAnListener:
    pass


class TestTheListenerRegistry:
    @override_settings(AI_SDK_WORKFLOWS={"LISTENERS": {"loud": "tests.unit.test_workflow_listeners.Loud"}})
    def test_a_declared_listener_is_composable(self):
        # By name, not identity: import_string reaches this module under its own
        # name, which is not the one pytest collected it under.
        assert [cls.__name__ for cls in get_listener_registry().values()] == ["Loud"]

    @override_settings(AI_SDK_WORKFLOWS={"LISTENERS": {}})
    def test_the_package_ships_none(self):
        assert get_listener_registry() == {}

    @override_settings(AI_SDK_WORKFLOWS={"LISTENERS": {"gone": "nowhere.NoSuchListener"}})
    def test_a_path_that_will_not_import_is_left_out(self):
        assert get_listener_registry() == {}

    @override_settings(AI_SDK_WORKFLOWS={"LISTENERS": {"wrong": "tests.unit.test_workflow_listeners.NotAnListener"}})
    def test_a_class_that_is_not_a_listener_is_left_out(self):
        assert get_listener_registry() == {}


class TestConfig:
    def test_a_listener_is_built_from_the_config_its_spec_carries(self):
        from django_ai_sdk.workflows.definitions import compile_listeners
        from django_ai_sdk.workflows.schemas import ListenerDefinition

        with override_settings(AI_SDK_WORKFLOWS={"LISTENERS": {"loud": "tests.unit.test_workflow_listeners.Loud"}}):
            (listener,) = compile_listeners(
                [ListenerDefinition(type="loud", config={"to": "#ops"})], "a workflow"
            )

        assert listener.config == {"to": "#ops"}

    def test_a_listener_built_in_code_takes_no_config(self):
        assert Loud().config == {}
