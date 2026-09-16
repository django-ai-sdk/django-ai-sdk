"""Hooks: the run recorder's row lifecycle, and the registry that gates the rest."""

import pytest
from django.test import override_settings
from pydantic import BaseModel

from django_ai_sdk.workflows import (
    RunRecorder,
    WorkflowHook,
    StepOutcome,
    WorkflowContext,
    WorkflowRun,
    WorkflowRunStep,
    get_hook_registry,
    run_steps,
)
from tests.mocks.workflow import FakeStep

CTX = WorkflowContext()


@pytest.mark.django_db(transaction=True)
class TestRunRecorder:
    """The one hook the package ships, and the one the executor always attaches."""

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

        await run_steps(steps, hooks=[RunRecorder(run, steps)])

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


class Loud(WorkflowHook):
    description = "says so"


class NotAHook:
    pass


class TestTheHookRegistry:
    """The gate on what a JSON definition may name."""

    @override_settings(AI_SDK_WORKFLOW_HOOKS={"loud": "tests.unit.test_workflow_hooks.Loud"})
    def test_a_declared_hook_is_composable(self):
        # By name, not identity: import_string reaches this module under its own
        # name, which is not the one pytest collected it under.
        assert [cls.__name__ for cls in get_hook_registry().values()] == ["Loud"]

    @override_settings(AI_SDK_WORKFLOW_HOOKS={})
    def test_the_package_ships_none(self):
        """Delivering a result somewhere is the host's business, not the SDK's."""
        assert get_hook_registry() == {}

    @override_settings(AI_SDK_WORKFLOW_HOOKS={"gone": "nowhere.NoSuchHook"})
    def test_a_path_that_will_not_import_is_left_out(self):
        assert get_hook_registry() == {}

    @override_settings(AI_SDK_WORKFLOW_HOOKS={"wrong": "tests.unit.test_workflow_hooks.NotAHook"})
    def test_a_class_that_is_not_a_hook_is_left_out(self):
        assert get_hook_registry() == {}


class TestConfig:
    """One registered class serves every definition that names it."""

    def test_a_hook_is_built_from_the_config_its_spec_carries(self):
        from django_ai_sdk.workflows.definitions import compile_hooks
        from django_ai_sdk.workflows.schemas import HookDefinition

        with override_settings(
            AI_SDK_WORKFLOW_HOOKS={"loud": "tests.unit.test_workflow_hooks.Loud"}
        ):
            (hook,) = compile_hooks(
                [HookDefinition(type="loud", config={"to": "#ops"})], "a workflow"
            )

        assert hook.config == {"to": "#ops"}

    def test_a_hook_built_in_code_takes_no_config(self):
        assert Loud().config == {}
