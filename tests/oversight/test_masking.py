"""Masks: what happens when a mask cannot point, or changes nothing."""

from __future__ import annotations

import pytest
from django.test import override_settings
from django_ai_sdk.oversight import (
    Block,
    Choice,
    Entities,
    JudgeLens,
    L,
    Mask,
    Patterns,
    PIILens,
    Policy,
    Rule,
    register_policy,
)
from django_ai_sdk.oversight.models import RuleMatch
from django_ai_sdk.oversight.types import (
    Answer,
    Stage,
    Subject,
    TextSpan,
)

from tests.oversight.helpers import (
    ENFORCE,
    FnLens,
    broken,
    closed_turn,
    found_without_spans,
    masking_policy,
    policy,
    start,
)


@pytest.mark.django_db(transaction=True)
class TestMasksThatCannotPoint:
    """A mask that cannot say where to mask blocks instead; one with nothing to mask did nothing."""

    async def test_a_lens_that_found_something_without_saying_where_blocks_the_question(self):
        guard = await start(masking_policy("test-no-spans", found_without_spans), "Jan asked")
        gate = await guard.gate()
        assert gate.blocked
        assert gate.message == str(Block.DEFAULT_MESSAGE)
        assert gate.notice["outcome"] == "blocked"
        # The question is not echoed: nothing in it was masked.
        assert "input" not in gate.notice
        turn = await closed_turn(guard)
        assert turn.outcome == "blocked"
        row = await RuleMatch.objects.aget(turn=turn, rule="mask_names")
        assert row.acted

    async def test_a_masking_lens_that_failed_blocks_unless_open(self):
        name = masking_policy("test-mask-broken", broken)
        gate = await (await start(name, "Jan asked")).gate()
        assert gate.blocked and gate.message == str(Block.DEFAULT_MESSAGE)
        with override_settings(AI_SDK_OVERSIGHT={"FAIL_MODE": "open"}):
            guard = await start(name, "Jan asked")
        gate = await guard.gate()
        assert not gate.blocked and gate.input == "Jan asked"
        assert (await closed_turn(guard)).outcome == "allowed"

    async def test_an_answer_that_cannot_be_masked_is_withheld(self):
        name = masking_policy("test-no-spans-out", found_without_spans, subject=Subject.OUTPUT)
        guard = await start(name)
        finished = await guard.finish("Ask Jan.")
        assert finished.blocked and finished.output == str(Block.DEFAULT_MESSAGE)
        assert (await closed_turn(guard)).outcome == "blocked"

    async def test_a_mask_with_nothing_to_mask_is_not_recorded_as_masking(self):
        @register_policy
        class MaskOnTopic(Policy):
            name = "test-mask-on-topic"
            lenses = [  # noqa: RUF012
                JudgeLens(
                    name="topic",
                    answer=Choice,
                    labels=["hr", "other"],
                    stages=[Stage.GATE],
                    judges=[Patterns(keywords={"hr": ["salary"]})],
                ),
                PIILens(name="mail", stages=[Stage.GATE], kinds=["email"]),
            ]
            rules = [  # noqa: RUF012
                Rule(
                    "mask_hr_mail",
                    when=L("topic") == "hr",
                    then=Mask(lenses=["mail"]),
                    mode=ENFORCE,
                )
            ]

        guard = await start("test-mask-on-topic", "What about my salary?")
        gate = await guard.gate()
        assert (gate.blocked, gate.masked, gate.notice) == (False, False, None)
        turn = await closed_turn(guard)
        assert turn.outcome == "allowed"
        assert not (await RuleMatch.objects.aget(turn=turn, rule="mask_hr_mail")).acted


@pytest.mark.django_db(transaction=True)
class TestMasksThatChangeNothing:
    """Spans outside the text block; "masked" only when text changed."""

    @pytest.mark.parametrize("span", [(900, 950), (3, 3), (-1, 2)], ids=["past", "empty", "neg"])
    async def test_a_span_outside_the_question_blocks(self, span):
        def lying(text, view):
            return Answer(value=["email"], spans=[TextSpan(start=span[0], end=span[1])])

        name = policy(
            f"fx-span-{span[0]}-{span[1]}",
            [FnLens(name="pii", answer=Entities, stages=[Stage.GATE], fn=lying)],
            [Rule("m", when=L("pii").is_true, then=Mask(), mode=ENFORCE)],
        )
        guard = await start(name, "mail a@b.nl please")
        gate = await guard.gate()
        assert gate.blocked and not gate.masked
        assert (await closed_turn(guard)).outcome == "blocked"

    async def test_a_mask_that_changes_nothing_did_not_act(self):
        def masks_its_own_label(text, view):
            # Points at text that already reads as its mask.
            return Answer(value=["email"], spans=[TextSpan(start=0, end=7, label="email")])

        name = policy(
            "fx-idle-mask",
            [FnLens(name="pii", answer=Entities, stages=[Stage.GATE], fn=masks_its_own_label)],
            [Rule("m", when=L("pii").is_true, then=Mask(), mode=ENFORCE)],
        )
        guard = await start(name, "[EMAIL] is here")
        gate = await guard.gate()
        assert not gate.masked and not gate.blocked
        turn = await closed_turn(guard)
        assert turn.outcome == "allowed"
        assert not (await RuleMatch.objects.aget(turn=turn, rule="m")).acted


def test_a_span_across_question_and_answer_masks_both_sides():
    from django_ai_sdk.oversight.guard.asking import Answered
    from django_ai_sdk.oversight.guard.masks import _spans_for

    lens = JudgeLens(name="who", answer=Entities, subject=Subject.EXCHANGE)
    # "ask Jan" + "\n\n" + "Jansen here": one name across the blank line.
    found = Answered(lens, Stage.FINISH, Answer(value=["Jan Jansen"], spans=[TextSpan(start=4, end=15)]))
    question = _spans_for([found], {"who"}, Subject.INPUT, "ask Jan")
    answer = _spans_for([found], {"who"}, Subject.OUTPUT, "ask Jan")
    assert [(s.start, s.end) for s in question] == [(4, 7)]
    assert [(s.start, s.end) for s in answer] == [(0, 6)]
