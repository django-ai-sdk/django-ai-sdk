"""What the runner does with a step's outcome, and with the rest of the run."""

import pytest
from django.core.exceptions import ImproperlyConfigured

from django_ai_sdk.workflows import (
    OnError,
    StepFailed,
    StepOutcome,
    WorkflowContext,
    run_steps,
)
from tests.mocks.workflow import FakeStep, RecordingHook


class TestOrdering:
    """Steps run in the order they were declared, and read what came before."""

    async def test_steps_run_in_the_order_they_were_declared(self):
        journal: list[str] = []
        steps = [
            FakeStep("ocr", journal=journal),
            FakeStep("extract", requires=("ocr",), journal=journal),
            FakeStep("match", requires=("extract",), journal=journal),
        ]

        await run_steps(steps)

        assert journal == ["ocr", "extract", "match"]

    async def test_a_step_declared_ahead_of_its_producer_is_refused(self):
        """Order is the author's, so a list that cannot run says so up front."""
        steps = [FakeStep("match", requires=("extract",)), FakeStep("extract")]

        with pytest.raises(ImproperlyConfigured, match="no earlier step"):
            await run_steps(steps)

    async def test_a_step_reads_what_an_earlier_one_produced(self):
        """Keyed by the producing step's name, so nothing has to be declared twice."""
        seen = {}

        class Reader(FakeStep):
            async def run(self, ctx):
                seen["text"] = ctx.step("ocr")
                return await super().run(ctx)

        steps = [
            FakeStep("ocr", outcome=StepOutcome(output="the text")),
            Reader("triage", requires=("ocr",)),
        ]

        await run_steps(steps)

        assert seen["text"] == "the text"

    async def test_a_step_reads_what_the_caller_passed_in(self):
        """Inputs are their own namespace, so they need no `requires`."""
        seen = {}

        class Reader(FakeStep):
            async def run(self, ctx):
                seen["doc"] = ctx.input("doc")
                return await super().run(ctx)

        await run_steps([Reader("ocr")], inputs={"doc": "a file"})

        assert seen["doc"] == "a file"

    async def test_a_step_name_and_an_input_name_do_not_contend(self):
        """Two namespaces: one key can mean both without either being renamed."""
        seen = {}

        class Reader(FakeStep):
            async def run(self, ctx):
                seen["input"] = ctx.input("document")
                seen["step"] = ctx.step("document")
                return await super().run(ctx)

        steps = [
            FakeStep("document", outcome=StepOutcome(output="what the step made")),
            Reader("triage", requires=("document",)),
        ]

        await run_steps(steps, inputs={"document": "what the caller passed"})

        assert seen["input"] == "what the caller passed"
        assert seen["step"] == "what the step made"


class TestFailure:
    async def test_a_fail_step_stops_the_run(self):
        steps = [
            FakeStep("ocr", raises=RuntimeError("provider 503")),
            FakeStep("triage", requires=("ocr",)),
        ]

        with pytest.raises(RuntimeError, match="provider 503"):
            await run_steps(steps)

    async def test_a_returned_failure_obeys_on_error_too(self):
        """`on_error=FAIL` applies whether a step raises or reports it."""
        steps = [FakeStep("ocr", outcome=StepOutcome(status="failed", detail="empty text"))]

        with pytest.raises(StepFailed, match="empty text"):
            await run_steps(steps)

    async def test_a_nested_runs_failure_obeys_this_steps_on_error(self):
        """A step wrapping its own runner is a step, not a second run."""
        steps = [
            FakeStep(
                "nested",
                raises=StepFailed("inner step failed"),
                on_error=OnError.CONTINUE,
            ),
            FakeStep("after"),
        ]

        outcomes = await run_steps(steps)

        assert outcomes["nested"].status == "failed"
        assert outcomes["after"].status == "completed"

    async def test_a_continue_failure_reaches_only_what_read_its_output(self):
        """The triage case: a step nothing depends on must not cost the rest."""
        journal: list[str] = []
        steps = [
            FakeStep("ocr", journal=journal),
            FakeStep(
                "extract",
                requires=("ocr",),
                raises=RuntimeError("provider 503"),
                on_error=OnError.CONTINUE,
                journal=journal,
            ),
            FakeStep("match", requires=("extract",), journal=journal),
            FakeStep("triage", requires=("ocr",), journal=journal),
        ]

        outcomes = await run_steps(steps)

        assert "provider 503" in outcomes["extract"].detail
        assert outcomes["match"].status == "skipped"
        assert outcomes["triage"].status == "completed"
        assert journal == ["ocr", "extract", "triage"]

    async def test_a_hook_sees_the_failure_a_handler_step_used_to_read(self):
        """What `error_key` was for: reacting to a failure is a hook's job now."""
        hook = RecordingHook()
        steps = [
            FakeStep(
                "extract",
                raises=RuntimeError("provider 503"),
                on_error=OnError.CONTINUE,
            ),
            FakeStep("match", requires=("extract",)),
        ]

        outcomes = await run_steps(steps, hooks=[hook])

        assert ("step_end", "extract", "failed") in hook.events
        assert outcomes["match"].status == "skipped"

    async def test_a_skipped_step_names_the_step_it_waited_for(self):
        steps = [
            FakeStep(
                "extract",
                outcome=StepOutcome(status="failed", detail="unreadable"),
                on_error=OnError.CONTINUE,
            ),
            FakeStep("match", requires=("extract",)),
        ]

        outcomes = await run_steps(steps)

        assert outcomes["match"].detail == "extract produced nothing"

    async def test_a_step_that_produced_none_still_counts_as_run(self):
        """Completion, not a truthy value: None is an answer."""
        steps = [
            FakeStep("ocr", outcome=StepOutcome(output=None)),
            FakeStep("triage", requires=("ocr",)),
        ]

        outcomes = await run_steps(steps)

        assert outcomes["triage"].status == "completed"

    async def test_a_failure_cascades_without_the_runner_tracking_it(self):
        """A skipped step produces nothing, so its own readers skip in turn."""
        steps = [
            FakeStep("ocr", raises=RuntimeError("down"), on_error=OnError.CONTINUE),
            FakeStep("extract", requires=("ocr",)),
            FakeStep("match", requires=("extract",)),
        ]

        outcomes = await run_steps(steps)

        assert outcomes["extract"].status == "skipped"
        assert outcomes["match"].status == "skipped"

    async def test_every_step_that_will_not_run_is_recorded(self):
        """Nothing is merely absent, so a reader can see the whole run."""
        hook = RecordingHook()
        steps = [
            FakeStep("ocr", raises=RuntimeError("down")),
            FakeStep("triage", requires=("ocr",)),
            FakeStep("report", requires=("ocr",)),
        ]

        with pytest.raises(RuntimeError):
            await run_steps(steps, hooks=[hook])

        settled = {name: status for event, name, status in hook.events if event == "step_end"}
        assert settled == {"ocr": "failed", "triage": "skipped", "report": "skipped"}


class TestSkipping:
    async def test_a_skipped_step_is_recorded_with_its_reason(self):
        hook = RecordingHook()
        steps = [
            FakeStep("ocr"),
            FakeStep("extract", requires=("ocr",), skip_reason="not an invoice"),
            FakeStep("match", requires=("extract",)),
        ]

        outcomes = await run_steps(steps, hooks=[hook])

        assert outcomes["extract"].detail == "not an invoice"
        assert outcomes["match"].detail == "extract produced nothing"
        assert ("step_start", "extract", None) not in hook.events

    async def test_skip_when_may_read_the_run_state(self):
        class Conditional(FakeStep):
            async def skip_when(self, ctx: WorkflowContext) -> str:
                return "" if ctx.input("kind") == "invoice" else "not an invoice"

        outcomes = await run_steps([Conditional("extract")], inputs={"kind": "letter"})

        assert outcomes["extract"].status == "skipped"


class TestResume:
    async def test_a_completed_step_is_not_re_run_and_its_output_returns(self):
        seen = {}

        class Reader(FakeStep):
            async def run(self, ctx):
                seen["text"] = ctx.step("ocr")
                return await super().run(ctx)

        ocr = FakeStep("ocr")
        triage = Reader("triage", requires=("ocr",))

        await run_steps([ocr, triage], completed={"ocr": "stored text"})

        assert ocr.calls == 0
        assert seen["text"] == "stored text"

    async def test_a_replayed_step_is_not_recorded_again(self):
        """Recording it would date a row to a run the step took no part in."""
        hook = RecordingHook()

        await run_steps(
            [FakeStep("ocr"), FakeStep("triage", requires=("ocr",))],
            hooks=[hook],
            completed={"ocr": "stored text"},
        )

        assert [name for _event, name, _status in hook.events if name] == ["triage", "triage"]

    async def test_a_replayed_step_appears_in_the_runs_outputs(self):
        """Resume must not drop replayed output from the returned map."""
        from django_ai_sdk.workflows.executor import _published

        outcomes = await run_steps(
            [
                FakeStep("ocr"),
                FakeStep("triage", requires=("ocr",), outcome=StepOutcome(output="SUM")),
            ],
            completed={"ocr": "stored text"},
        )

        assert outcomes["ocr"].output == "stored text"
        assert _published(outcomes) == {"ocr": "stored text", "triage": "SUM"}


class TestHooks:
    async def test_they_are_optional(self):
        outcomes = await run_steps([FakeStep("ocr")])

        assert outcomes["ocr"].status == "completed"

    async def test_two_runs_can_use_different_hooks(self):
        """No process-wide hook: the record follows the call."""
        first, second = RecordingHook(), RecordingHook()

        await run_steps([FakeStep("ocr")], hooks=[first])
        await run_steps([FakeStep("triage")], hooks=[second])

        assert [name for _e, name, _s in first.events if name] == ["ocr", "ocr"]
        assert [name for _e, name, _s in second.events if name] == ["triage", "triage"]

    async def test_a_workflow_hook_fires_for_every_step(self):
        hook = RecordingHook()

        await run_steps([FakeStep("ocr"), FakeStep("triage")], hooks=[hook])

        assert [name for event, name, _s in hook.events if event == "step_end"] == [
            "ocr",
            "triage",
        ]

    async def test_a_step_hook_fires_for_that_step_alone(self):
        """The difference the two attachment points buy."""
        on_the_step = RecordingHook()

        await run_steps([FakeStep("ocr", hooks=(on_the_step,)), FakeStep("triage")])

        assert [name for _e, name, _s in on_the_step.events] == ["ocr", "ocr"]

    async def test_a_step_hook_is_told_nothing_about_the_run(self):
        """Run-level callbacks belong to hooks attached to the workflow."""
        on_the_step = RecordingHook()

        await run_steps([FakeStep("ocr", hooks=(on_the_step,))])

        assert [event for event, _n, _s in on_the_step.events] == ["step_start", "step_end"]

    async def test_a_broken_hook_does_not_fail_work_that_succeeded(self):
        """A hook watches the run; a notification nobody received is not a failure."""
        from django_ai_sdk.workflows import WorkflowHook

        class Broken(WorkflowHook):
            async def on_step_end(self, ctx, step, outcome):
                raise RuntimeError("slack is down")

        outcomes = await run_steps([FakeStep("ocr")], hooks=[Broken()])

        assert outcomes["ocr"].status == "completed"

    async def test_a_broken_hook_does_not_stop_the_hooks_after_it(self):
        from django_ai_sdk.workflows import WorkflowHook

        class Broken(WorkflowHook):
            async def on_step_end(self, ctx, step, outcome):
                raise RuntimeError("slack is down")

        after = RecordingHook()

        await run_steps([FakeStep("ocr")], hooks=[Broken(), after])

        assert ("step_end", "ocr", "completed") in after.events

    async def test_a_hook_refuses_a_duplicate_delivery_by_raising(self):
        """The one exception: a hook whose rows are also a claim has to reach the caller."""
        from django_ai_sdk.workflows import WorkflowHook
        from django_ai_sdk.workflows.steps import StepAlreadyRunning

        class Claim(WorkflowHook):
            async def on_step_start(self, ctx, step):
                raise StepAlreadyRunning(step.name)

        with pytest.raises(StepAlreadyRunning):
            await run_steps([FakeStep("ocr")], hooks=[Claim()])

    async def test_a_crash_reaches_the_hooks_before_it_propagates(self):
        hook = RecordingHook()

        with pytest.raises(RuntimeError):
            await run_steps([FakeStep("ocr", raises=RuntimeError("boom"))], hooks=[hook])

        assert ("step_end", "ocr", "failed") in hook.events


class TestNamesAreResolvable:
    """The one name rule left: a step's name is its key, and it is unique."""

    async def test_a_required_name_with_no_source_is_refused(self):
        """The typo case: no earlier step goes by that name."""
        steps = [FakeStep("ocr"), FakeStep("triage", requires=("ocrr",))]

        with pytest.raises(ImproperlyConfigured, match="ocrr"):
            await run_steps(steps)

    async def test_a_duplicate_step_name_is_refused(self):
        with pytest.raises(ImproperlyConfigured, match="duplicate step name"):
            await run_steps([FakeStep("ocr"), FakeStep("ocr")])

    async def test_an_unnamed_step_is_refused(self):
        with pytest.raises(ImproperlyConfigured, match="needs a `name`"):
            await run_steps([FakeStep("")])

    async def test_an_empty_pipeline_is_refused(self):
        with pytest.raises(ImproperlyConfigured, match="no steps to run"):
            await run_steps([])


class TestTheRunLevelBracket:
    """A workflow hook is told the run started, and how it ended."""

    async def test_run_start_fires_before_the_first_step(self):
        hook = RecordingHook()

        await run_steps([FakeStep("ocr")], hooks=[hook])

        assert hook.events[0] == ("run_start", "", None)

    async def test_run_end_carries_no_error_when_the_run_finished(self):
        hook = RecordingHook()

        await run_steps([FakeStep("ocr")], hooks=[hook])

        assert hook.events[-1] == ("run_end", "", None)

    async def test_run_end_carries_what_ended_the_run(self):
        hook = RecordingHook()

        with pytest.raises(RuntimeError):
            await run_steps([FakeStep("ocr", raises=RuntimeError("boom"))], hooks=[hook])

        assert hook.events[-1] == ("run_end", "", "boom")
