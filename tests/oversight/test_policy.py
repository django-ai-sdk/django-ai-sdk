"""Conditions, rules and policies: one engine for code and settings, with sources."""

from __future__ import annotations

from typing import Any

import pytest
from django.core.exceptions import ImproperlyConfigured
from django.test import override_settings
from django_ai_sdk.oversight import (
    Alert,
    Block,
    Choice,
    Entities,
    JudgeLens,
    L,
    Mask,
    Mode,
    Notify,
    PatternLens,
    PatternJudge,
    PIILens,
    Policy,
    RestrictTools,
    Rule,
    Score,
    LensPack,
    RulePack,
    register_policy,
)
from django_ai_sdk.oversight import registry as resolution
from django_ai_sdk.oversight.rules import Condition
from django_ai_sdk.oversight.content import found_literals
from django_ai_sdk.oversight.guard.judging import Answered, match_rules, answer_lenses
from django_ai_sdk.oversight.judges.base import Judge
from django_ai_sdk.oversight.packs.security import SecurityPack
from django_ai_sdk.oversight.registry import build_policy
from django_ai_sdk.oversight.policy import validate_policy
from django_ai_sdk.oversight.registry import get_invalid_policies
from django_ai_sdk.oversight.registry import invalidate_policy
from django_ai_sdk.oversight.types import (
    Answer,
    Capture,
    Stage,
    Subject,
    TextSpan,
    TurnContext,
)


def ans(value, confidence=1.0, error=""):
    return Answer(value=value, confidence=confidence, error=error)


@register_policy
class ParentPolicy(Policy):
    name = "test-parent"
    # A Flag, not Entities: it never says where, so it cannot be masked.
    lenses = [JudgeLens(name="mentions_x", question="X?", judges=[PatternJudge(keywords={"x": ["x"]})])]
    rules = [Rule("block_x", when=L("mentions_x").is_true, then=Block("no x"))]


@register_policy
class ChildPolicy(Policy):
    name = "test-child"
    extends = ["test-parent"]  # noqa: RUF012
    capture = Capture.MASKED
    lenses = [
        JudgeLens(
            name="topic",
            answer=Choice,
            labels=["a", "b"],
            default="b",
            judges=[PatternJudge(keywords={"a": ["a"]})],
        )
    ]
    rules = [Rule("note_b", when=L("topic") == "b", then=Alert(), mode=Mode.ENFORCE)]


@register_policy
class PickedPolicy(Policy):
    name = "test-picked"
    lenses = [LensPack(SecurityPack.lenses, "injection_question", "secrets_question")]
    rules = [
        RulePack(SecurityPack.rules, "injection_in_question"),
        RulePack(SecurityPack.rules, "secrets_in_question", mode=Mode.ENFORCE),
    ]


@register_policy
class HeldParent(Policy):
    name = "test-held-parent"
    stream_mode = "live"
    retention = {"content": 7}  # noqa: RUF012


@register_policy
class HeldChild(Policy):
    name = "test-held-child"
    extends = ["test-held-parent"]  # noqa: RUF012


def down(lens, context):
    raise RuntimeError("down")


class Said(Judge):
    """A judge that answers what `fn(lens, context)` returns."""

    def __init__(self, fn, name: str = "said") -> None:
        self.fn = fn
        self.name = name

    async def answer(self, lens, context):
        import inspect

        result = self.fn(lens, context)
        if inspect.isawaitable(result):
            result = await result
        return result if result is None or isinstance(result, Answer) else Answer(value=result)


class TestConditions:
    def test_operators(self):
        answers = {"inj": ans(True, 0.95), "tox": ans(0.9), "topic": ans("hr"), "pii": ans(["bsn"])}
        assert (L("inj").is_true & L("inj").confident(0.9)).matches(answers)
        assert (L("tox") > 0.8).matches(answers)
        assert not (L("tox") < 0.8).matches(answers)
        assert (L("topic") == "hr").matches(answers)
        assert L("topic").in_({"hr", "legal"}).matches(answers)
        assert L("pii").contains("bsn").matches(answers)
        assert L("pii").is_true.matches(answers)
        assert (~(L("topic") == "billing")).matches(answers)

    def test_a_missing_answer_matches_nothing_but_missing(self):
        answers = {"broken": ans(None, error="timeout")}
        assert not L("broken").is_true.matches(answers)
        assert not (L("absent") > 0).matches(answers)
        assert L("broken").missing.matches(answers)
        assert L("absent").missing.matches(answers)

    def test_a_test_that_cannot_be_evaluated_is_refused(self):
        bad = [
            {"lens": "a", "op": "confident"},
            {"lens": "a", "op": "unsure", "value": 2},
            {"lens": "a", "op": "gt", "value": "high"},
            {"lens": "a", "op": "in", "value": "hr"},
        ]
        for node in bad:
            with pytest.raises(ValueError):
                Condition(node)

    def test_round_trips_through_json(self):
        condition = (L("a").is_true & (L("b") >= 0.5)) | ~L("c").in_(["x"])
        again = Condition(condition.to_dict())
        assert again.to_dict() == condition.to_dict()
        assert again.lenses() == {"a", "b", "c"}

    def test_rule_round_trip(self):
        rule = Rule("r", when=L("a").is_true, then=[Block("no"), Alert("high")], mode="enforce")
        again = Rule.from_dict(rule.to_dict())
        assert again.to_dict() == rule.to_dict()
        assert [a.kind for a in again.ordered_actions()] == ["block", "alert"]


class TestPolicies:
    def test_extends_merges_with_sources(self):
        resolved = build_policy("test-child")
        assert set(resolved.lenses) == {"mentions_x", "topic"}
        assert set(resolved.rules) == {"block_x", "note_b"}
        # Each value names the policy that declared it.
        assert resolved.sources["rule:block_x"] == "test-parent"
        assert resolved.sources["lens:topic"] == "test-child"
        assert build_policy("default").sources["lens:injection_question"] == "security"
        assert resolved.capture is Capture.MASKED

    def test_inherited_rules_start_in_report_mode(self):
        assert build_policy("test-child").rules["block_x"].mode is Mode.REPORT

    @override_settings(AI_SDK_OVERSIGHT={"CAPTURE": "metadata"})
    def test_the_deployment_capture_is_a_ceiling(self):
        assert build_policy("test-child").capture is Capture.METADATA

    def test_incomplete_policies_are_not_registered(self):
        class NoName(Policy):
            pass

        with pytest.raises(ImproperlyConfigured):
            validate_policy(NoName)

        @register_policy
        class Bad(Policy):
            name = "test-bad"
            lenses = [JudgeLens(name="c", answer=Choice, judges=["judge"])]

        assert "test-bad" in get_invalid_policies()

    def test_a_policy_picks_single_lenses_and_rules_from_a_pack(self):
        resolved = build_policy("test-picked")
        assert list(resolved.lenses) == ["injection_question", "secrets_question"]
        assert list(resolved.rules) == ["injection_in_question", "secrets_in_question"]
        assert resolved.rules["secrets_in_question"].mode is Mode.ENFORCE
        assert not get_invalid_policies().get("test-picked")

    def test_picking_leaves_the_pack_alone(self):
        pack = {rule.name: rule for rule in SecurityPack.rules}
        assert pack["secrets_in_question"].mode is Mode.REPORT
        assert RulePack(SecurityPack.rules, "injection_in_question")[0] is not pack["injection_in_question"]

    def test_picking_an_unknown_name_says_what_there_is(self):
        with pytest.raises(ImproperlyConfigured, match="injection_question"):
            LensPack(SecurityPack.lenses, "prompt_injektion")

    def test_mode_only_applies_to_rules(self):
        with pytest.raises(ImproperlyConfigured, match="mode"):
            LensPack(SecurityPack.lenses, "injection_question", mode=Mode.ENFORCE)

    def test_a_pack_without_names_takes_everything(self):
        assert [rule.name for rule in RulePack(SecurityPack.rules)] == [
            rule.name for rule in SecurityPack.rules
        ]

    def test_a_lens_pack_sets_overrides_on_copies(self):
        [taken] = LensPack(SecurityPack.lenses, "secrets_question", stages=(Stage.REFLECT,))
        [own] = [lens for lens in SecurityPack.lenses if getattr(lens, "name", "") == "secrets_question"]
        assert taken.stages == (Stage.REFLECT,) and own.stages != (Stage.REFLECT,)

    def test_packs_and_lists_in_a_policy_are_flattened(self):
        class Nested(Policy):
            name = "test-nested"
            lenses = [LensPack(SecurityPack.lenses, "pii_answer"), PickedPolicy.lenses]
            rules = [PickedPolicy.rules, RulePack(SecurityPack.rules, "pii_in_answer")]

        assert [rule.name for rule in Nested.rules] == [
            "injection_in_question",
            "secrets_in_question",
            "pii_in_answer",
        ]
        assert [getattr(lens, "name", "") for lens in Nested.lenses] == [
            "pii_answer",
            "injection_question",
            "secrets_question",
        ]

    def test_the_packs_are_registered_and_resolve(self):
        resolved = build_policy("default")
        assert {"injection_question", "pii_question", "groundedness", "self_harm_question"} <= set(
            resolved.lenses
        )
        assert all(rule.mode is Mode.REPORT for rule in resolved.rules.values())


class TestRuleEvaluation:
    def test_a_rule_waits_for_lenses_still_to_come(self):
        resolved = build_policy("default")
        early = match_rules(resolved, {"pii_question": ans([])}, Stage.GATE, done=set())
        assert "ungrounded_answer" not in {m.rule.name for m in early}
        later = match_rules(resolved, {"groundedness": ans(0.1, 0.9)}, Stage.REFLECT, done=set())
        assert "ungrounded_answer" in {m.rule.name for m in later}

    def test_a_block_that_is_already_decided_does_not_wait(self):
        resolved = build_policy("default")
        rule = resolved.rules["ungrounded_answer"]
        rule.mode = Mode.ENFORCE
        found = {"pii_question": ans(["email"])}

        def at_gate(when, *then):
            rule.when, rule.then = when, list(then)
            return [
                m for m in match_rules(resolved, found, Stage.GATE, done=set()) if m.rule is rule
            ]

        either = L("pii_question").is_true | (L("groundedness") < 0.5)
        [match] = at_gate(either, Block())
        assert [a.kind for a in match.acted] == ["block"]
        # Not decided yet: an `all`, or a `missing` on the later lens. A mask waits anyway.
        assert not at_gate(L("pii_question").is_true & (L("groundedness") < 0.5), Block())
        assert not at_gate(L("pii_question").is_false | L("groundedness").missing, Block())
        assert not at_gate(either, Mask())
        assert at_gate(either, RestrictTools(["search"]))
        # Restricting tools early would lose the mask at finish; blocking ends the turn.
        assert not at_gate(either, RestrictTools(["search"]), Mask())
        assert at_gate(either, Block(), Mask())

    def test_report_matches_record_but_do_not_act(self):
        resolved = build_policy("test-parent")
        [match] = match_rules(resolved, {"mentions_x": ans(True)}, Stage.GATE, done=set())
        assert not match.acted
        assert match.outcome(report=True).value == "blocked"

    def test_enforced_actions_only_where_they_still_can(self):
        resolved = build_policy("test-parent")
        resolved.rules["block_x"].mode = Mode.ENFORCE
        resolved.rules["block_x"].then = [Block(), Mask()]
        [gate] = match_rules(resolved, {"mentions_x": ans(True)}, Stage.GATE, done=set())
        assert [a.kind for a in gate.acted] == ["block", "mask"]
        [late] = match_rules(resolved, {"mentions_x": ans(True)}, Stage.REFLECT, done=set())
        assert [a.kind for a in late.acted] == ["mask"]

    def test_fail_closed_treats_an_error_as_a_match(self):
        resolved = build_policy("test-parent")
        broken = {"mentions_x": ans(None, error="timeout")}

        def closed():
            return match_rules(
                resolved, broken, Stage.GATE, done=set(), errored={"mentions_x"}, fail_closed=True
            )

        # In report mode, a slow answerer would only fill the record with false matches.
        assert not closed()
        resolved.rules["block_x"].mode = Mode.ENFORCE
        assert not match_rules(resolved, broken, Stage.GATE, done=set())
        assert [m.rule.name for m in closed()] == ["block_x"]

    def test_fail_closed_never_raises_an_alert(self):
        resolved = build_policy("test-parent")
        resolved.rules["block_x"].mode = Mode.ENFORCE
        resolved.rules["block_x"].then = [Alert("high")]
        broken = {"mentions_x": ans(None, error="timeout")}
        assert not match_rules(
            resolved, broken, Stage.GATE, done=set(), errored={"mentions_x"}, fail_closed=True
        )

    def test_retention_takes_content_and_metadata_only(self):
        from django_ai_sdk.oversight.conf import validate_retention

        assert build_policy("test-parent").retention == {"content": 90, "metadata": 730}
        with pytest.raises(ValueError, match="content.*metadata"):
            validate_retention({"security": 90})


class TestSeverity:
    def test_unknown_severities_are_refused(self):
        assert Notify(severity="critical").severity == "critical"
        with pytest.raises(ValueError, match="Unknown severity"):
            Alert("loud")

    def test_the_most_severe_wins(self):
        from django_ai_sdk.oversight.types import most_severe

        assert most_severe("info", "high", "warning") == "high"
        assert most_severe() == ""


class TestResolving:
    async def test_policies_resolve_from_code_and_are_cached(self):
        resolved = await resolution.get_resolved_policy("test-child")
        assert resolved.rules["block_x"].mode is Mode.REPORT
        assert await resolution.get_resolved_policy("test-child") is resolved

    async def test_invalidating_a_parent_forgets_its_children(self):
        resolved = await resolution.get_resolved_policy("test-child")
        invalidate_policy("test-parent")
        assert await resolution.get_resolved_policy("test-child") is not resolved


class TestPolicyFields:
    def test_stream_mode_and_retention_are_inherited(self):
        resolved = build_policy("test-held-child")
        assert resolved.stream_mode == "live"
        assert resolved.retention == {"content": 7, "metadata": 730}
        assert resolved.sources["policy.stream_mode"] == "test-held-parent"

    def test_an_unknown_stream_mode_is_refused(self):
        from unittest.mock import patch

        from django_ai_sdk.oversight import registry

        class Typo(Policy):
            name = "test-typo"
            stream_mode = "held"

        class Hold(Policy):
            name = "test-hold"
            stream_mode = "hold"

        with patch.dict(registry._registry, {"test-typo": Typo, "test-hold": Hold}):
            with pytest.raises(ImproperlyConfigured, match="stream_mode 'held'"):
                build_policy("test-typo")
            # "hold" is gone: a client that needs the whole answer buffers it itself.
            with pytest.raises(ImproperlyConfigured, match="stream_mode 'hold'"):
                build_policy("test-hold")

    @override_settings(AI_SDK_OVERSIGHT={"STREAM_MODE": "held"})
    def test_an_unknown_default_stream_mode_is_refused(self):
        with pytest.raises(ImproperlyConfigured, match="STREAM_MODE"):
            build_policy("test-parent")


class TestDeclarationsThatCannotWork:
    """What could never do what it says is refused where it is declared, not on a turn."""

    def test_a_mask_over_a_lens_that_never_says_where_is_refused(self):
        class FlagMask(Policy):
            name = "test-flag-mask"
            lenses = [JudgeLens(name="f", judges=[PatternJudge(keywords={"x": ["x"]})])]  # noqa: RUF012
            rules = [Rule("mask_f", when=L("f").is_true, then=Mask(), mode=Mode.ENFORCE)]  # noqa: RUF012

        with pytest.raises(ImproperlyConfigured, match="only an entities lens"):
            validate_policy(FlagMask)

        class NamedMask(FlagMask):
            name = "test-named-mask"
            lenses = [  # noqa: RUF012
                *FlagMask.lenses,
                PIILens(name="mail", kinds=["email"]),
            ]
            rules = [  # noqa: RUF012
                Rule("mask_mail", when=L("f").is_true, then=Mask(lenses=["mail"]))
            ]

        validate_policy(NamedMask)

    def test_a_rule_masking_an_inherited_lens_that_never_says_where_does_not_resolve(self):
        @register_policy
        class MasksParent(Policy):
            name = "test-masks-parent"
            extends = ["test-parent"]  # noqa: RUF012
            rules = [Rule("mask_x", when=L("mentions_x").is_true, then=Mask())]  # noqa: RUF012

        with pytest.raises(ImproperlyConfigured, match="only an entities lens"):
            build_policy("test-masks-parent")

    def test_a_rule_reading_a_lens_the_policy_lacks_does_not_resolve(self):
        @register_policy
        class Typo(Policy):
            name = "test-unknown-lens"
            extends = ["test-parent"]  # noqa: RUF012
            rules = [Rule("r", when=L("mentions_y").is_true, then=Alert())]  # noqa: RUF012

        @register_policy
        class MaskTypo(Policy):
            name = "test-unknown-mask-lens"
            extends = ["test-parent"]  # noqa: RUF012
            rules = [  # noqa: RUF012
                Rule("r", when=L("mentions_x").is_true, then=Mask(lenses=["nope"]))
            ]

        for name in ("test-unknown-lens", "test-unknown-mask-lens"):
            with pytest.raises(ImproperlyConfigured, match="unknown lenses"):
                build_policy(name)

    def test_a_choice_default_must_be_one_of_its_labels(self):
        class OddDefault(Policy):
            name = "test-odd-default"
            lenses = [  # noqa: RUF012
                JudgeLens(
                    name="c",
                    answer=Choice,
                    labels=["a", "b"],
                    default="other",
                    judges=[PatternJudge(keywords={"a": ["a"]})],
                )
            ]

        with pytest.raises(ImproperlyConfigured, match="default"):
            validate_policy(OddDefault)

    def test_an_attribute_the_lens_kind_does_not_have_is_refused(self):
        with pytest.raises(ImproperlyConfigured, match="has no attribute"):
            PatternLens(name="p", keywords=["x"], judges=["judge"])

    def test_a_score_is_read_on_the_scale_the_judge_was_told(self):
        lens = JudgeLens(name="s", answer=Score)
        assert Score.parse(lens, "1") == 1.0
        assert Score.parse(lens, 0.25) == 0.25
        for off_the_scale in (7, "10", -0.1, 1.5):
            with pytest.raises(ValueError):
                Score.parse(lens, off_the_scale)


class TestAnswerChains:
    async def test_a_chain_that_ends_on_an_error_after_unsure_answers_failed(self):
        unsure = Said(lambda lens, v: Answer(value=False, confidence=0.5), name="quick")
        judge = Said(down, name="judge")
        lens = JudgeLens(name="f", judges=[unsure, judge])
        [answered] = await answer_lenses([lens], TurnContext(input="x"), Stage.GATE)
        # Kept for the record, failed for FAIL_MODE.
        assert (answered.answer.value, answered.answer.answerer) == (False, "quick")
        assert answered.failed and answered.errored
        # Ending on an unsure answer is not failing: the last one to answer decides.
        lens = JudgeLens(name="g", judges=[judge, unsure])
        [answered] = await answer_lenses([lens], TurnContext(input="x"), Stage.GATE)
        assert answered.answer.answerer == "quick"
        assert not answered.failed and not answered.errored

    def test_reflect_asks_again_when_the_service_after_an_unsure_answer_was_down(self):
        from django_ai_sdk.oversight.guard.judging import should_retry

        assert should_retry("", ["unavailable: ConnectionError"])
        assert should_retry("unavailable: ConnectionError")
        assert not should_retry("", ["unavailable: ConnectionError", "RuntimeError"])

    async def test_a_cut_off_unsure_answer_failed(self):
        import asyncio

        async def slow(lens, v):
            await asyncio.sleep(2)

        unsure = Said(lambda lens, v: Answer(value=False, confidence=0.5), name="quick")
        lens = JudgeLens(name="f", judges=[unsure, Said(slow)])
        [answered] = await answer_lenses([lens], TurnContext(input="x"), Stage.GATE, deadline_ms=50)
        assert answered.answer.answerer == "quick" and answered.failed

    @pytest.mark.parametrize("deadline_ms", [None, 50])
    async def test_no_judge_outlives_its_stage(self, deadline_ms):
        import asyncio

        stopped = asyncio.Event()

        async def slow(lens, v):
            try:
                await asyncio.sleep(10)
            finally:
                stopped.set()

        lens = JudgeLens(name="f", judges=[Said(slow)])
        stage = asyncio.ensure_future(
            answer_lenses([lens], TurnContext(input="x"), Stage.GATE, deadline_ms=deadline_ms)
        )
        if deadline_ms is None:
            # The stage itself is cancelled: the person went away mid-stream.
            await asyncio.sleep(0.05)
            stage.cancel()
            with pytest.raises(asyncio.CancelledError):
                await stage
        else:
            await stage
        # Stopped by the time the stage is done, not some time later.
        assert stopped.is_set()


class TestDeclarations:
    def test_default_stages_follow_the_subject(self):
        assert JudgeLens(name="a").stages == (Stage.GATE,)
        assert JudgeLens(name="b", subject=Subject.ANSWER).stages == (Stage.FINISH,)
        assert JudgeLens(name="c", subject=Subject.EXCHANGE).stages == (Stage.FINISH,)
        assert JudgeLens(name="d", subject=Subject.SOURCES).stages == (Stage.FINISH,)
        assert JudgeLens(name="e", subject=Subject.TOOL_CALL).stages == (Stage.TOOL,)
        assert JudgeLens(name="f", stages=[Stage.REFLECT]).stages == (Stage.REFLECT,)

    def test_there_is_no_watch_stage(self):
        assert [str(s) for s in Stage] == ["gate", "tool", "finish", "reflect"]
        with pytest.raises(ValueError):
            Stage("watch")

    def test_a_lens_at_a_stage_that_never_reads_its_subject_is_refused(self):
        misplaced = [
            {"subject": Subject.ANSWER, "stages": [Stage.TOOL]},
            {"stages": [Stage.FINISH]},
            {"subject": Subject.TOOL_CALL, "stages": [Stage.REFLECT]},
        ]
        for kwargs in misplaced:
            with pytest.raises(ImproperlyConfigured, match="never reads it"):
                JudgeLens(name="a", judges=["judge"], **kwargs).validate()
        JudgeLens(
            name="d", judges=["judge"], subject=Subject.ANSWER, stages=[Stage.REFLECT]
        ).validate()
        JudgeLens(name="e", judges=["judge"], subject=Subject.TOOL_CALL).validate()

    def test_sample_and_min_confidence_are_fractions(self):
        with pytest.raises(ImproperlyConfigured, match="sample"):
            JudgeLens(name="a", judges=["judge"], sample=2).validate()
        with pytest.raises(ImproperlyConfigured, match="min_confidence"):
            JudgeLens(name="a", judges=["judge"], min_confidence=-0.1).validate()

    def test_exfiltration_is_judged_on_the_whole_answer_only(self):
        from django_ai_sdk.oversight.packs.security import Exfiltration, SecurityPack

        assert Exfiltration().stages == (Stage.FINISH,)
        pii_tool_call, secrets_tool_call = LensPack(SecurityPack.lenses, "pii_tool_call", "secrets_tool_call")
        assert pii_tool_call.stages == secrets_tool_call.stages == (Stage.TOOL,)

    def test_only_entities_spans_are_masked_in_what_capture_keeps(self):
        text = "What about my salary, Jan?"
        answered = [
            Answered(
                JudgeLens(name="topic"),
                Stage.GATE,
                Answer(value=True, spans=[TextSpan(start=14, end=20, label="hr")]),
            ),
            Answered(
                JudgeLens(name="names", answer=Entities),
                Stage.GATE,
                Answer(value=["person"], spans=[TextSpan(start=22, end=25, label="person")]),
            ),
        ]
        assert found_literals(answered, {Subject.QUESTION: text}) == {"Jan": "person"}

    def test_a_mask_over_a_tool_call_lens_is_refused(self):
        from django_ai_sdk.oversight.policy import validate_rule

        lens = PatternLens(name="args", subject=Subject.TOOL_CALL, keywords=["x"])
        rule = Rule("mask_args", when=L("args").is_true, then=Mask(), mode=Mode.ENFORCE)
        with pytest.raises(ImproperlyConfigured, match="tool call"):
            validate_rule(rule, {"args": lens})

    def test_a_mask_over_a_sources_lens_is_refused(self):
        from django_ai_sdk.oversight.policy import validate_rule

        lens = PatternLens(name="src", subject=Subject.SOURCES, keywords=["x"])
        rule = Rule("mask_src", when=L("src").is_true, then=Mask(), mode=Mode.ENFORCE)
        with pytest.raises(ImproperlyConfigured, match="only the question and the answer"):
            validate_rule(rule, {"src": lens})

    def test_reset_registry_forgets_cached_resolutions(self, registry_snapshot: Any):
        from django_ai_sdk.oversight import registry

        registry._cache["x"] = (None, frozenset())
        registry_snapshot.reset_registry()
        assert registry._cache == {}
