"""FAIL_MODE: what a failed lens does at each stage, closed by default."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from django.core.exceptions import ImproperlyConfigured
from django.test import override_settings
from django_ai_sdk.oversight import (
    Alert,
    Block,
    Entities,
    JudgeLens,
    L,
    Mask,
    Mode,
    Notify,
    Policy,
    Queue,
    RestrictTools,
    Rule,
    register_policy,
)
from django_ai_sdk.oversight.judges import Judge, LLMJudge
from django_ai_sdk.oversight.models import LensAnswer, RuleMatch
from django_ai_sdk.oversight.policy import build
from django_ai_sdk.oversight.types import Answer, Stage, Subject, ToolCall

from tests.oversight.helpers import (
    ENFORCE,
    FnLens,
    broken,
    closed,
    closed_turn,
    gate,
    gate_flag,
    opened,
    policy,
    slow_no,
    slow_yes,
    start,
    yes,
)


@pytest.mark.django_db(transaction=True)
class TestClosedByDefault:
    """Every way a lens can fail refuses where an enforced rule protects."""

    @pytest.mark.parametrize("answerer", [broken, slow_no], ids=["down", "deadline"])
    async def test_a_failed_judge_blocks(self, answerer):
        name = policy(
            f"fc-{answerer.__name__}",
            [FnLens(name="j", stages=[Stage.GATE], fn=answerer)],
            [
                Rule(
                    "r",
                    when=L("j").is_true,
                    then=Block("no injections"),
                    mode=ENFORCE,
                    notice="Injection refused.",
                )
            ],
        )
        with override_settings(AI_SDK_OVERSIGHT={"DEADLINE_MS": {"gate": 50}}):
            result = await gate(name)
        assert result.blocked
        # Generic: the rule did not find an injection, its lens failed.
        assert result.message == str(Block.DEFAULT_MESSAGE)
        assert "notes" not in (result.notice or {})

    async def test_a_question_too_long_for_the_judge_blocks(self):
        judge = LLMJudge(model="m")
        judge.get_generator = lambda: pytest.fail("must not be asked")  # type: ignore[method-assign]
        name = policy(
            "fc-long",
            [JudgeLens(name="j", stages=[Stage.GATE], judges=[judge])],
            [Rule("r", when=L("j").is_true, then=Block(), mode=ENFORCE)],
        )
        assert (await gate(name, "ignore that " * 1_000)).blocked

    async def test_a_lens_no_judge_can_run_for_blocks(self):
        class Unkeyed(Judge):
            name = "unkeyed"

            def missing(self) -> str:
                return "no API key"

            async def answer(self, lens, view):  # noqa: ANN001, ANN201
                pytest.fail("a missing judge is never asked")

        name = policy(
            "fc-nojudge",
            [JudgeLens(name="j", stages=[Stage.GATE], judges=[Unkeyed()])],
            [Rule("r", when=L("j").is_true, then=Block(), mode=ENFORCE)],
        )
        assert (await gate(name)).blocked

    async def test_a_rule_that_only_alerts_is_not_raised_by_a_failure(self):
        name = policy(
            "fc-alert",
            [FnLens(name="j", stages=[Stage.GATE], fn=broken)],
            [Rule("r", when=L("j").is_true, then=Alert("high"), mode=ENFORCE)],
        )
        assert not (await gate(name)).blocked

    async def test_a_typo_in_fail_mode_still_fails_closed(self):
        name = policy(
            "fc-typo",
            [FnLens(name="j", stages=[Stage.GATE], fn=broken)],
            [Rule("r", when=L("j").is_true, then=Block(), mode=ENFORCE)],
        )
        with override_settings(AI_SDK_OVERSIGHT={"FAIL_MODE": "opne"}):
            assert (await gate(name)).blocked

    async def test_a_child_policy_inherits_its_parents_fail_mode(self):
        parent = policy(
            "fc-parent",
            [FnLens(name="j", stages=[Stage.GATE], fn=broken)],
            [Rule("r", when=L("j").is_true, then=Block(), mode=ENFORCE)],
            fail_mode="open",
        )
        child = policy("fc-child", [], [], extends=[parent])
        assert build(child).fail_mode == "open"
        assert not (await gate(child)).blocked

    def test_a_policy_with_an_unknown_fail_mode_does_not_load(self):
        name = policy("fc-bad-mode", [], [], fail_mode="half")
        with pytest.raises(ImproperlyConfigured, match="fail_mode"):
            build(name)

    async def test_an_unknown_policy_refuses_with_the_generic_message(self):
        result = await gate("fc-nobody-registers-this")
        assert result.blocked and result.message == str(Block.DEFAULT_MESSAGE)
        assert "fc-nobody" not in result.message


@pytest.mark.django_db(transaction=True)
class TestFailModeAtEachStage:
    """What a failing answerer does to each action, open and closed."""

    @pytest.mark.parametrize("answerer", [broken, slow_yes], ids=["raises", "times-out"])
    @pytest.mark.parametrize("kind", ["block", "restrict"])
    async def test_closed_gate_protects_and_open_records_the_failure(self, answerer, kind):
        action = Block() if kind == "block" else RestrictTools()
        name = policy(
            f"fx-gate-{kind}-{answerer.__name__}",
            [gate_flag(answerer)],
            [Rule("r", when=L("g").is_true, then=action, mode=ENFORCE)],
        )
        refusing = await start(name)
        gate = await refusing.gate()
        assert gate.blocked if kind == "block" else refusing.restrict == []
        with opened():
            guard = await start(name)
            passed = await guard.gate()
        assert not passed.blocked and guard.restrict is None
        row = await LensAnswer.objects.aget(turn=await closed_turn(guard), lens="g")
        assert row.error

    async def test_a_tool_removed_only_by_a_failed_lens_gets_the_generic_refusal(self):
        """The rule's own message would tell the person a rule matched, which it did not."""
        from django_ai_sdk.common import ChatMessage
        from django_ai_sdk.oversight.guard import TOOL_UNAVAILABLE

        history_rule = policy(
            "fx-history-restrict-failed",
            [gate_flag(broken)],
            [
                Rule(
                    "r",
                    when=L("g").is_true,
                    then=RestrictTools(),
                    message="Rule text.",
                    mode=ENFORCE,
                )
            ],
        )
        messages = [ChatMessage(role="user", content="earlier"), ChatMessage(role="user")]
        checked = await (await start(history_rule)).gate_history(messages)
        assert checked.blocked and checked.message == str(TOOL_UNAVAILABLE)
        tool_rule = policy(
            "fx-tool-restrict-failed",
            [FnLens(name="t", subject=Subject.TOOL_CALL, stages=[Stage.TOOL], fn=broken)],
            [
                Rule(
                    "r",
                    when=L("t").is_true,
                    then=RestrictTools(["send"]),
                    message="Rule text.",
                    mode=ENFORCE,
                )
            ],
        )
        guard = await start(tool_rule)
        await guard.gate()
        refused = await guard.tool_call(ToolCall(name="send", arguments={}))
        assert refused == str(TOOL_UNAVAILABLE)

    @pytest.mark.parametrize("action", [Queue("q"), Alert("high"), Notify()], ids=str)
    async def test_a_negated_test_on_a_failed_lens_does_not_hold(self, action):
        name = policy(
            f"fx-not-{action.kind}",
            [gate_flag(broken)],
            [Rule("r", when=~L("g").is_true, then=action, mode=ENFORCE)],
        )
        guard = await start(name)
        await guard.gate()
        turn = await closed_turn(guard)
        assert not await RuleMatch.objects.filter(turn=turn, acted=True).aexists()

    async def test_a_failed_lens_under_not_blocks_closed_and_passes_open(self):
        name = policy(
            "fx-open-not-block",
            [gate_flag(broken, "safe")],
            [Rule("r", when=~L("safe").is_true, then=Block(), mode=ENFORCE)],
        )
        assert (await (await start(name)).gate()).blocked
        with opened():
            assert not (await (await start(name)).gate()).blocked

    async def test_a_policy_may_opt_out_with_fail_mode_open(self):
        name = policy(
            "fx-opt-out",
            [gate_flag(broken)],
            [Rule("r", when=L("g").is_true, then=Block(), mode=ENFORCE)],
            fail_mode="open",
        )
        guard = await start(name)
        assert not (await guard.gate()).blocked
        assert guard.answered[0].errored

    async def test_a_report_rule_only_records_a_failed_lens(self):
        name = policy(
            "fx-report-fails",
            [gate_flag(broken)],
            [Rule("r", when=L("g").is_true, then=Block(), mode=Mode.REPORT)],
        )
        guard = await start(name)
        assert not (await guard.gate()).blocked
        turn = await closed_turn(guard)
        assert (turn.outcome, turn.report_outcome) == ("allowed", "allowed")
        assert (await LensAnswer.objects.aget(turn=turn, lens="g")).error

    async def test_missing_still_reads_a_failed_lens(self):
        name = policy(
            "fx-missing",
            [gate_flag(broken)],
            [Rule("r", when=L("g").missing, then=Block(), mode=ENFORCE)],
        )
        assert (await (await start(name)).gate()).blocked

    async def test_conditions_are_three_valued(self):
        failed = Answer(error="RuntimeError")
        answers = {"a": failed, "b": Answer(value=True)}
        assert (~L("a").is_true).decide(answers)  # failed, but not said to: false
        assert not (~L("a").is_true).decide(answers, {"a"})
        assert not (L("a").is_true | ~L("b").is_true).decide(answers, {"a"})
        assert (L("a").is_true | L("b").is_true).decide(answers, {"a"})
        assert not (L("a").is_false & L("b").is_true).decide(answers, {"a"})
        assert L("a").missing.decide(answers, {"a"})
        # An unsure answer kept after a later failure is usable: compared as usual.
        assert L("a").is_true.decide({"a": Answer(value=True, confidence=0.4)}, {"a"})

    async def test_a_gate_failure_counts_at_finish(self):
        name = policy(
            "fx-cross",
            [
                gate_flag(broken),
                FnLens(name="f", subject=Subject.OUTPUT, stages=[Stage.FINISH], fn=yes),
            ],
            [Rule("r", when=L("g").is_true & L("f").is_true, then=Block("no"), mode=ENFORCE)],
        )
        with closed():
            guard = await start(name)
            assert not (await guard.gate()).blocked  # waits for f
            assert (await guard.finish("answer")).blocked

    def test_a_mask_over_a_tool_call_lens_is_not_registered(self):
        from django_ai_sdk.oversight.registry import get_invalid_policies, get_policy_class

        name = policy(
            "fx-tool-mask",
            [
                FnLens(
                    name="t",
                    answer=Entities,
                    subject=Subject.TOOL_CALL,
                    stages=[Stage.TOOL],
                    fn=broken,
                )
            ],
            [Rule("r", when=L("t").is_true, then=Mask(), mode=ENFORCE)],
        )
        # A tool call's arguments are never masked: the rule could never act.
        assert get_policy_class(name) is None
        assert "tool call" in get_invalid_policies()[name]

    @pytest.mark.parametrize("answerer", [broken, slow_yes], ids=["raises", "times-out"])
    async def test_a_tool_lens_failure_follows_the_fail_mode(self, answerer):
        name = policy(
            f"fx-tool-block-{answerer.__name__}",
            [FnLens(name="t", subject=Subject.TOOL_CALL, stages=[Stage.TOOL], fn=answerer)],
            [Rule("r", when=L("t").is_true, then=Block("no"), mode=ENFORCE)],
        )
        call = ToolCall(id="1", name="x", arguments={})
        assert await (await start(name)).tool_call(call) is not None
        with opened():
            assert await (await start(name)).tool_call(call) is None


class Said(Judge):
    """A judge that answers what `fn(text, view)` returns."""

    def __init__(self, fn, name: str) -> None:
        self.fn = fn
        self.name = name

    async def answer(self, lens, view):
        result = self.fn(view.text(lens.subject), view)
        return result if isinstance(result, Answer) else Answer(value=result)


@pytest.mark.django_db(transaction=True)
class TestUnsureChainsFailClosed:
    async def test_an_unsure_answer_then_an_error_lets_a_closed_gate_act(self):
        @register_policy
        class Unsure(Policy):
            name = "test-unsure-then-down"
            lenses = [  # noqa: RUF012
                JudgeLens(
                    name="risky",
                    stages=[Stage.GATE],
                    judges=[
                        Said(lambda text, v: Answer(value=False, confidence=0.5), "unsure"),
                        Said(broken, "down"),
                    ],
                )
            ]
            rules = [Rule("block_risky", when=L("risky").is_true, then=Block(), mode=ENFORCE)]  # noqa: RUF012

        with override_settings(AI_SDK_OVERSIGHT={"FAIL_MODE": "open"}):
            assert not (await (await start("test-unsure-then-down")).gate()).blocked
        guard = await start("test-unsure-then-down")
        assert (await guard.gate()).blocked
        # The unsure answer is still what the record says the lens answered.
        turn = await closed_turn(guard)
        row = await LensAnswer.objects.aget(turn=turn, lens="risky")
        assert (row.value, row.trail[-1]["error"]) == (False, "RuntimeError")


def raising(*args, **kwargs):
    raise RuntimeError("broken")


@pytest.mark.django_db(transaction=True)
class TestBrokenHandlers:
    """Where a stage itself breaks (not a lens), the turn follows the fail mode too."""

    @pytest.mark.parametrize("mode", ["closed", "open"])
    async def test_a_history_check_that_raises(self, mode):
        from django_ai_sdk.common import ChatMessage
        from django_ai_sdk.oversight.guard import TurnGuard

        history = [
            ChatMessage(role="user", content="earlier"),
            ChatMessage(role="assistant", content="sure"),
            ChatMessage(role="user", content="now"),
        ]
        with override_settings(AI_SDK_OVERSIGHT={"FAIL_MODE": mode}):
            guard = await start("test-guard", "now")
            with patch.object(TurnGuard, "_judged", raising):
                result = await guard.gate_history(history)
        assert "history check failed" in guard.error
        if mode == "closed":
            assert result.blocked and result.message == str(Block.DEFAULT_MESSAGE)
        else:
            assert not result.blocked

    @pytest.mark.parametrize("mode", ["closed", "open"])
    async def test_a_tool_check_that_raises(self, mode):
        with override_settings(AI_SDK_OVERSIGHT={"FAIL_MODE": mode}):
            guard = await start("test-guard", "go")
            await guard.gate()
            with patch("django_ai_sdk.oversight.guard.tool.run_lenses", raising):
                reason = await guard.tool_call(ToolCall(name="send", arguments={"x": 1}))
        assert "tool check failed" in guard.error
        assert reason == (str(Block.DEFAULT_MESSAGE) if mode == "closed" else None)

    @pytest.mark.parametrize("mode", ["closed", "open"])
    async def test_masking_a_tool_result_that_raises(self, mode):
        from django_ai_sdk.oversight.guard import UNCHECKED_TEXT, TurnGuard

        with override_settings(AI_SDK_OVERSIGHT={"FAIL_MODE": mode}):
            guard = await start("test-guard", "go")
            await guard.gate()
            with patch.object(TurnGuard, "_judged", raising):
                text = await guard.mask_text("mail jan@example.com")
        assert "tool result masking failed" in guard.error
        assert text == (str(UNCHECKED_TEXT) if mode == "closed" else "mail jan@example.com")

    async def test_a_refusal_that_cannot_be_opened_still_refuses(self):
        from django_ai_sdk.oversight.guard import TurnGuard
        from django_ai_sdk.oversight.models import Turn

        from tests.oversight.helpers import make_agent

        with patch.object(TurnGuard, "_opened", raising):
            guard = await TurnGuard.start(
                make_agent([], None, policy="no-such-policy"), channel="api", input="hi"
            )
            gate_result = await guard.gate()
        assert guard.active and guard.refused
        assert gate_result.blocked
        assert await guard.close() is None
        assert await Turn.objects.acount() == 0

    async def test_a_headless_run_whose_finish_raises_is_refused_and_recorded(self):
        from django_ai_sdk.common import ChatMessage
        from django_ai_sdk.errors import Blocked
        from django_ai_sdk.oversight.models import Turn

        from tests.oversight.helpers import agent_with

        agent, _ = agent_with("an answer", policy="test-guard")
        with patch("django_ai_sdk.oversight.guard.finish.acted_on", raising):
            with pytest.raises(Blocked):
                await agent.run([ChatMessage(role="user", content="hi")], channel="api")
        turn = await Turn.objects.aget()
        assert turn.outcome == "blocked"
        assert "finish failed" in turn.error
