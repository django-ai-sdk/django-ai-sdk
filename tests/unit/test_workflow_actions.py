"""Actions: the run recorder's row lifecycle, and the registry that gates the rest."""

import pytest
from django.test import override_settings
from pydantic import BaseModel

from django_ai_sdk.workflows import (
    RunRecorder,
    StepAlreadyRunning,
    WorkflowAction,
    StepOutcome,
    WorkflowContext,
    WorkflowRun,
    WorkflowRunStep,
    get_action_registry,
    run_steps,
)
from tests.mocks.workflow import FakeStep

CTX = WorkflowContext()


@pytest.mark.django_db(transaction=True)
class TestRunRecorder:
    """The one action the package ships, and the one the executor always attaches."""

    async def _recorder(self, *names):
        """A recorder over a pipeline of `names`, which fixes each row's sequence."""
        run = await WorkflowRun.objects.acreate()
        steps = [FakeStep(name) for name in names]
        return run, RunRecorder(run, steps), steps

    async def test_start_then_end_leaves_one_completed_row(self):
        run, recorder, (ocr,) = await self._recorder("ocr")

        await recorder.on_step_start(CTX, ocr)
        row = await run.steps.aget(step_name="ocr")
        assert row.status == WorkflowRunStep.Status.RUNNING

        await recorder.on_step_end(CTX, ocr, StepOutcome(output="the text"))
        await row.arefresh_from_db()

        assert row.status == WorkflowRunStep.Status.COMPLETED
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
        assert row.status == WorkflowRunStep.Status.FAILED
        assert row.error == "nothing readable"

    async def test_a_skipped_outcome_is_a_row_carrying_its_reason(self):
        """A step that never ran is visible, not absent."""
        run, recorder, (extract,) = await self._recorder("extract")

        await recorder.on_step_end(
            CTX, extract, StepOutcome(status="skipped", detail="not an invoice")
        )

        row = await run.steps.aget(step_name="extract")
        assert row.status == WorkflowRunStep.Status.SKIPPED
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
        run = await WorkflowRun.objects.acreate()
        steps = [
            FakeStep("ocr", outcome=StepOutcome(output="t")),
            FakeStep("triage", requires=("ocr",)),
        ]

        await run_steps(steps, actions=[RunRecorder(run, steps)])

        rows = {row.step_name: row.status async for row in run.steps.all()}
        assert rows == {
            "ocr": WorkflowRunStep.Status.COMPLETED,
            "triage": WorkflowRunStep.Status.COMPLETED,
        }

    async def test_the_run_row_is_marked_running_then_failed(self):
        run, recorder, _steps = await self._recorder("ocr")

        await recorder.on_run_start(CTX)
        await run.arefresh_from_db()
        assert run.status == WorkflowRun.Status.RUNNING
        assert run.started_at is not None

        await recorder.on_run_end(CTX, RuntimeError("provider 503"))
        await run.arefresh_from_db()

        assert run.status == WorkflowRun.Status.FAILED
        assert run.error == "provider 503"

    async def test_a_clean_ending_leaves_the_run_row_to_the_executor(self):
        """Only the executor knows the outputs, so only it stamps completion."""
        run, recorder, _steps = await self._recorder("ocr")

        await recorder.on_run_start(CTX)
        await recorder.on_run_end(CTX, None)
        await run.arefresh_from_db()

        assert run.status == WorkflowRun.Status.RUNNING


@pytest.mark.django_db(transaction=True)
class TestTheStepRowIsAClaim:
    """A duplicate delivery of the same run reaches the same step twice."""

    async def _recorder(self, *names):
        run = await WorkflowRun.objects.acreate()
        steps = [FakeStep(name) for name in names]
        return run, RunRecorder(run, steps), steps

    async def test_a_second_start_of_a_running_step_is_refused(self):
        """Two workers racing a redelivery: the second loses the claim."""
        run, recorder, (ocr,) = await self._recorder("ocr")

        await recorder.on_step_start(CTX, ocr)

        with pytest.raises(StepAlreadyRunning):
            await recorder.on_step_start(CTX, ocr)

        row = await run.steps.aget(step_name="ocr")
        assert row.status == WorkflowRunStep.Status.RUNNING

    async def test_two_independent_runs_do_not_contend(self):
        """The claim is per-run: a step name repeated across runs is not a race."""
        run_a, recorder_a, (ocr_a,) = await self._recorder("ocr")
        run_b, recorder_b, (ocr_b,) = await self._recorder("ocr")

        await recorder_a.on_step_start(CTX, ocr_a)
        await recorder_b.on_step_start(CTX, ocr_b)  # does not raise

        assert await run_a.steps.acount() == 1
        assert await run_b.steps.acount() == 1

    async def test_a_failed_step_may_be_reclaimed_on_a_legitimate_retry(self):
        """A resume retries a FAILED step; that is not the race this guards against."""
        run, recorder, (ocr,) = await self._recorder("ocr")

        await recorder.on_step_start(CTX, ocr)
        await recorder.on_step_end(CTX, ocr, StepOutcome(status="failed", detail="503"))

        await recorder.on_step_start(CTX, ocr)  # does not raise

        row = await run.steps.aget(step_name="ocr")
        assert row.status == WorkflowRunStep.Status.RUNNING

    async def test_a_second_start_of_a_completed_step_is_refused(self):
        """Belt-and-braces: even if replay logic ever missed it, the row itself refuses."""
        run, recorder, (ocr,) = await self._recorder("ocr")

        await recorder.on_step_start(CTX, ocr)
        await recorder.on_step_end(CTX, ocr, StepOutcome(output="the text"))

        with pytest.raises(StepAlreadyRunning):
            await recorder.on_step_start(CTX, ocr)


class Loud(WorkflowAction):
    description = "says so"


class NotAnAction:
    pass


class TestTheActionRegistry:
    """The gate on what a JSON definition may name."""

    @override_settings(AI_SDK_WORKFLOW_ACTIONS={"loud": "tests.unit.test_workflow_actions.Loud"})
    def test_a_declared_action_is_composable(self):
        # By name, not identity: import_string reaches this module under its own
        # name, which is not the one pytest collected it under.
        assert [cls.__name__ for cls in get_action_registry().values()] == ["Loud"]

    @override_settings(AI_SDK_WORKFLOW_ACTIONS={})
    def test_the_package_ships_none(self):
        """Delivering a result somewhere is the host's business, not the SDK's."""
        assert get_action_registry() == {}

    @override_settings(AI_SDK_WORKFLOW_ACTIONS={"gone": "nowhere.NoSuchAction"})
    def test_a_path_that_will_not_import_is_left_out(self):
        assert get_action_registry() == {}

    @override_settings(AI_SDK_WORKFLOW_ACTIONS={"wrong": "tests.unit.test_workflow_actions.NotAnAction"})
    def test_a_class_that_is_not_a_action_is_left_out(self):
        assert get_action_registry() == {}


class TestConfig:
    """One registered class serves every definition that names it."""

    def test_an_action_is_built_from_the_config_its_spec_carries(self):
        from django_ai_sdk.workflows.definitions import compile_actions
        from django_ai_sdk.workflows.schemas import ActionDefinition

        with override_settings(
            AI_SDK_WORKFLOW_ACTIONS={"loud": "tests.unit.test_workflow_actions.Loud"}
        ):
            (action,) = compile_actions(
                [ActionDefinition(type="loud", config={"to": "#ops"})], "a workflow"
            )

        assert action.config == {"to": "#ops"}

    def test_an_action_built_in_code_takes_no_config(self):
        assert Loud().config == {}
