"""The default sink's row lifecycle."""

import pytest
from pydantic import BaseModel

from django_ai_sdk.workflows import (
    StepOutcome,
    WorkflowRun,
    WorkflowRunStep,
    WorkflowRunStepSink,
    run_steps,
)
from tests.mocks.workflow import FakeStep


@pytest.mark.django_db(transaction=True)
class TestWorkflowRunStepSink:
    async def _sink(self, *names):
        """A sink over a pipeline of `names`, which is what fixes each row's sequence."""
        run = await WorkflowRun.objects.acreate()
        steps = [FakeStep(name) for name in names]
        return run, WorkflowRunStepSink(run, steps)

    async def test_open_then_close_leaves_one_completed_row(self):
        run, sink = await self._sink("ocr")

        await sink.open("ocr")
        row = await run.steps.aget(step_name="ocr")
        assert row.status == WorkflowRunStep.Status.RUNNING
        assert row.output_key == "ocr"

        await sink.close("ocr", StepOutcome(output="the text"))
        await row.arefresh_from_db()

        assert row.status == WorkflowRunStep.Status.COMPLETED
        assert row.output == "the text"
        assert row.error == ""
        assert row.completed_at is not None

    async def test_a_failed_outcome_is_recorded_on_the_row(self):
        run, sink = await self._sink("extract")

        await sink.open("extract")
        await sink.close(
            "extract", StepOutcome(status="failed", detail="nothing readable")
        )

        row = await run.steps.aget(step_name="extract")
        assert row.status == WorkflowRunStep.Status.FAILED
        assert row.error == "nothing readable"

    async def test_a_skipped_outcome_is_a_row_carrying_its_reason(self):
        """A step that never ran is visible, not absent."""
        run, sink = await self._sink("extract")

        await sink.close("extract", StepOutcome(status="skipped", detail="not an invoice"))

        row = await run.steps.aget(step_name="extract")
        assert row.status == WorkflowRunStep.Status.SKIPPED
        assert row.detail == "not an invoice"
        assert row.error == ""

    async def test_fail_records_the_exception_on_the_row(self):
        run, sink = await self._sink("ocr")

        await sink.open("ocr")
        await sink.fail("ocr", RuntimeError("provider 503"))

        row = await run.steps.aget(step_name="ocr")
        assert row.status == WorkflowRunStep.Status.FAILED
        assert row.error == "provider 503"

    async def test_a_pydantic_output_is_stored_json_safe(self):
        class Lines(BaseModel):
            items: list[str]

        run, sink = await self._sink("extract")

        await sink.open("extract")
        await sink.close("extract", StepOutcome(output=Lines(items=["a"])))

        row = await run.steps.aget(step_name="extract")
        assert row.output == {"items": ["a"]}

    async def test_completed_reports_this_runs_outputs_for_resume(self):
        _first, sink = await self._sink("ocr")
        second = await WorkflowRun.objects.acreate()

        await sink.open("ocr")
        await sink.close("ocr", StepOutcome(output="the text"))

        assert await sink.completed() == {"ocr": "the text"}
        assert await WorkflowRunStepSink(second, []).completed() == {}

    async def test_a_run_records_every_step_it_walked(self):
        run = await WorkflowRun.objects.acreate()
        steps = [
            FakeStep("ocr", provides="text", outcome=StepOutcome(output="t")),
            FakeStep("triage", requires=("text",)),
        ]

        await run_steps(steps, sink=WorkflowRunStepSink(run, steps))

        rows = {row.step_name: row.status async for row in run.steps.all()}
        assert rows == {
            "ocr": WorkflowRunStep.Status.COMPLETED,
            "triage": WorkflowRunStep.Status.COMPLETED,
        }
