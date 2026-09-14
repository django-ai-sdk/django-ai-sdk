"""What the runner does with a step's outcome, and with the rest of the run."""

import pytest
from django.core.exceptions import ImproperlyConfigured

from django_ai_sdk.workflows import (
    OnError,
    StepContext,
    StepFailed,
    StepOutcome,
    run_steps,
)
from tests.mocks.workflow import FakeStep, RecordingSink


class TestOrdering:
    """Steps run in the order they were declared, and read what came before."""

    async def test_steps_run_in_the_order_they_were_declared(self):
        journal: list[str] = []
        steps = [
            FakeStep("ocr", provides="text", journal=journal),
            FakeStep("extract", requires=("text",), provides="kvk", journal=journal),
            FakeStep("match", requires=("kvk",), journal=journal),
        ]

        await run_steps(steps)

        assert journal == ["ocr", "extract", "match"]

    async def test_a_step_declared_ahead_of_its_producer_is_refused(self):
        """Order is the author's, so a list that cannot run says so up front."""
        steps = [
            FakeStep("match", requires=("kvk",)),
            FakeStep("extract", provides="kvk"),
        ]

        with pytest.raises(ImproperlyConfigured, match="no earlier step provides"):
            await run_steps(steps)

    async def test_a_step_reads_what_an_earlier_one_provided(self):
        seen = {}

        class Reader(FakeStep):
            async def run(self, ctx):
                seen["text"] = ctx.get("text")
                return await super().run(ctx)

        steps = [
            FakeStep("ocr", provides="text", outcome=StepOutcome(output="the text")),
            Reader("triage", requires=("text",)),
        ]

        await run_steps(steps)

        assert seen["text"] == "the text"

    async def test_a_step_reads_what_the_caller_passed_in(self):
        seen = {}

        class Reader(FakeStep):
            async def run(self, ctx):
                seen["doc"] = ctx.get("doc")
                return await super().run(ctx)

        await run_steps([Reader("ocr", requires=("doc",))], inputs={"doc": "a file"})

        assert seen["doc"] == "a file"


class TestFailure:
    async def test_a_fail_step_stops_the_run(self):
        steps = [
            FakeStep("ocr", provides="text", raises=RuntimeError("provider 503")),
            FakeStep("triage", requires=("text",)),
        ]

        with pytest.raises(RuntimeError, match="provider 503"):
            await run_steps(steps)

    async def test_a_returned_failure_obeys_on_error_too(self):
        """`on_error=FAIL` applies whether a step raises or reports it."""
        steps = [
            FakeStep(
                "ocr",
                provides="text",
                outcome=StepOutcome(status="failed", detail="empty text"),
            ),
        ]

        with pytest.raises(StepFailed, match="empty text"):
            await run_steps(steps)

    async def test_a_nested_runs_failure_obeys_this_steps_on_error(self):
        """A step wrapping its own runner is a step, not a second run."""
        steps = [
            FakeStep(
                "nested",
                provides="text",
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
            FakeStep("ocr", provides="text", journal=journal),
            FakeStep(
                "extract",
                requires=("text",),
                provides="kvk",
                raises=RuntimeError("provider 503"),
                on_error=OnError.CONTINUE,
                journal=journal,
            ),
            FakeStep("match", requires=("kvk",), journal=journal),
            FakeStep("triage", requires=("text",), journal=journal),
        ]

        outcomes = await run_steps(steps)

        assert "provider 503" in outcomes["extract"].detail
        assert outcomes["match"].status == "skipped"
        assert outcomes["triage"].status == "completed"
        assert journal == ["ocr", "extract", "triage"]

    async def test_continue_with_error_key_feeds_a_handler(self):
        seen = {}

        class Handler(FakeStep):
            async def run(self, ctx):
                seen["err"] = ctx.get("extract_err")
                return await super().run(ctx)

        steps = [
            FakeStep(
                "extract",
                provides="kvk",
                raises=RuntimeError("provider 503"),
                on_error=OnError.CONTINUE,
                error_key="extract_err",
            ),
            Handler("handler", requires=("extract_err",)),
            FakeStep("match", requires=("kvk",)),
        ]

        outcomes = await run_steps(steps)

        assert outcomes["handler"].status == "completed"
        assert outcomes["match"].status == "skipped"
        assert seen["err"]["step"] == "extract"
        assert "provider 503" in seen["err"]["error"]

    async def test_a_skipped_step_names_the_input_it_waited_for(self):
        steps = [
            FakeStep(
                "extract",
                provides="kvk",
                outcome=StepOutcome(status="failed", detail="unreadable"),
                on_error=OnError.CONTINUE,
            ),
            FakeStep("match", requires=("kvk",)),
        ]

        outcomes = await run_steps(steps)

        assert outcomes["match"].detail == "kvk was not produced"

    async def test_a_failure_cascades_without_the_runner_tracking_it(self):
        """A skipped step publishes nothing, so its own readers skip in turn."""
        steps = [
            FakeStep(
                "ocr",
                provides="text",
                raises=RuntimeError("down"),
                on_error=OnError.CONTINUE,
            ),
            FakeStep("extract", requires=("text",), provides="kvk"),
            FakeStep("match", requires=("kvk",)),
        ]

        outcomes = await run_steps(steps)

        assert outcomes["extract"].status == "skipped"
        assert outcomes["match"].status == "skipped"

    async def test_every_step_that_will_not_run_is_recorded(self):
        """Nothing is merely absent, so a reader can see the whole run."""
        sink = RecordingSink()
        steps = [
            FakeStep("ocr", provides="text", raises=RuntimeError("down")),
            FakeStep("triage", requires=("text",)),
            FakeStep("report", requires=("text",)),
        ]

        with pytest.raises(RuntimeError):
            await run_steps(steps, sink=sink)

        closed = {name: status for event, name, status in sink.events if event == "close"}
        assert closed == {"triage": "skipped", "report": "skipped"}
        # The crashed step keeps the row `sink.fail` wrote for it.
        assert ("fail", "ocr", "down") in sink.events
        assert "ocr" not in closed


class TestSkipping:
    async def test_a_skipped_step_is_recorded_with_its_reason(self):
        sink = RecordingSink()
        steps = [
            FakeStep("ocr", provides="text"),
            FakeStep(
                "extract",
                requires=("text",),
                provides="kvk",
                skip_reason="not an invoice",
            ),
            FakeStep("match", requires=("kvk",)),
        ]

        outcomes = await run_steps(steps, sink=sink)

        assert outcomes["extract"].detail == "not an invoice"
        assert outcomes["match"].detail == "kvk was not produced"
        assert ("open", "extract", None) not in sink.events

    async def test_skip_when_may_read_the_run_state(self):
        class Conditional(FakeStep):
            async def skip_when(self, ctx: StepContext) -> str:
                return "" if ctx.get("kind") == "invoice" else "not an invoice"

        outcomes = await run_steps(
            [Conditional("extract", requires=("kind",))], inputs={"kind": "letter"}
        )

        assert outcomes["extract"].status == "skipped"


class TestResume:
    async def test_a_completed_step_is_not_re_run_and_its_output_returns(self):
        seen = {}

        class Reader(FakeStep):
            async def run(self, ctx):
                seen["text"] = ctx.get("text")
                return await super().run(ctx)

        ocr = FakeStep("ocr", provides="text")
        triage = Reader("triage", requires=("text",))
        sink = RecordingSink(completed_names={"ocr": "stored text"})

        await run_steps([ocr, triage], sink=sink)

        assert ocr.calls == 0
        assert seen["text"] == "stored text"

    async def test_a_replayed_step_is_not_recorded_again(self):
        """Closing it would date a row to a run the step took no part in."""
        ocr = FakeStep("ocr", provides="text")
        sink = RecordingSink(completed_names={"ocr": "stored text"})

        await run_steps([ocr, FakeStep("triage", requires=("text",))], sink=sink)

        assert [name for _event, name, _status in sink.events if name] == ["triage", "triage"]

    async def test_a_replayed_step_appears_in_outcomes_and_published(self):
        """Resume must not drop replayed provides from the returned/persisted map."""
        from django_ai_sdk.workflows.executor import _published

        ocr = FakeStep("ocr", provides="text")
        triage = FakeStep(
            "triage",
            provides="summary",
            requires=("text",),
            outcome=StepOutcome(output="SUM"),
        )
        sink = RecordingSink(completed_names={"ocr": "stored text"})

        outcomes = await run_steps([ocr, triage], sink=sink)

        assert outcomes["ocr"].status == "completed"
        assert outcomes["ocr"].output == "stored text"
        assert _published([ocr, triage], outcomes) == {
            "text": "stored text",
            "summary": "SUM",
        }


class TestSink:
    async def test_it_is_optional(self):
        outcomes = await run_steps([FakeStep("ocr")])

        assert outcomes["ocr"].status == "completed"

    async def test_two_runs_can_use_different_sinks(self):
        """No process-wide sink: the record follows the call."""
        first, second = RecordingSink(), RecordingSink()

        await run_steps([FakeStep("ocr")], sink=first)
        await run_steps([FakeStep("triage")], sink=second)

        assert [name for _e, name, _s in first.events if name] == ["ocr", "ocr"]
        assert [name for _e, name, _s in second.events if name] == ["triage", "triage"]

    async def test_a_crash_reaches_the_sink_before_it_propagates(self):
        sink = RecordingSink()

        with pytest.raises(RuntimeError):
            await run_steps([FakeStep("ocr", raises=RuntimeError("boom"))], sink=sink)

        assert ("fail", "ocr", "boom") in sink.events


class TestNamesAreResolvable:
    """A name a step reads has one source, and the runner says which."""

    async def test_a_required_name_with_no_source_is_refused(self):
        """The typo case: nothing provides it and no input carries it."""
        steps = [FakeStep("ocr", requires=("extracted_txt",))]

        with pytest.raises(ImproperlyConfigured, match="extracted_txt"):
            await run_steps(steps, inputs={"extracted_text": "the text"})

    async def test_a_provided_name_may_not_shadow_an_input(self):
        steps = [FakeStep("ocr", provides="document")]

        with pytest.raises(ImproperlyConfigured, match="one source"):
            await run_steps(steps, inputs={"document": "a file"})

    async def test_two_steps_may_not_provide_one_name(self):
        steps = [FakeStep("first", provides="text"), FakeStep("second", provides="text")]

        with pytest.raises(ImproperlyConfigured, match="One name, one producer"):
            await run_steps(steps)

    async def test_a_required_name_the_caller_passed_is_accepted(self):
        outcomes = await run_steps(
            [FakeStep("ocr", requires=("document",))], inputs={"document": "a file"}
        )

        assert outcomes["ocr"].status == "completed"

    async def test_a_duplicate_step_name_is_refused(self):
        with pytest.raises(ImproperlyConfigured, match="Duplicate step name"):
            await run_steps([FakeStep("ocr"), FakeStep("ocr")])

    async def test_an_empty_pipeline_is_refused(self):
        with pytest.raises(ImproperlyConfigured, match="at least one step"):
            await run_steps([])


class TestTheRunLevelBracket:
    """A sink is told the run started, and how it ended."""

    async def test_begin_fires_before_the_first_step(self):
        sink = RecordingSink()

        await run_steps([FakeStep("ocr")], sink=sink)

        assert sink.events[0] == ("begin", "", None)

    async def test_end_carries_no_error_when_the_run_finished(self):
        sink = RecordingSink()

        await run_steps([FakeStep("ocr")], sink=sink)

        assert sink.events[-1] == ("end", "", None)

    async def test_end_carries_what_ended_the_run(self):
        sink = RecordingSink()

        with pytest.raises(RuntimeError):
            await run_steps([FakeStep("ocr", raises=RuntimeError("boom"))], sink=sink)

        assert sink.events[-1] == ("end", "", "boom")
