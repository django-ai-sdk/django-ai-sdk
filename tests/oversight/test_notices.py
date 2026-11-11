"""What the client and the person are told about what a rule did."""

from __future__ import annotations

import json

import pytest
from django.test import override_settings
from django_ai_sdk.oversight import (
    Alert,
    Answer,
    Block,
    L,
    Notify,
    Policy,
    Queue,
    Rule,
    register_policy,
)
from django_ai_sdk.oversight.models import LensAnswer, RuleMatch
from django_ai_sdk.oversight.types import Stage, Subject, ToolCall

from tests.oversight.helpers import (
    ENFORCE,
    FnLens,
    closed_turn,
    start,
)


@register_policy
class ToldPolicy(Policy):
    name = "test-told"
    lenses = [  # noqa: RUF012
        FnLens(
            name="fraud",
            stages=[Stage.GATE],
            fn=lambda text, v: Answer(value="wire" in v.input),
        ),
        FnLens(
            name="probe",
            stages=[Stage.GATE],
            fn=lambda text, v: Answer(value="probe" in v.input),
        ),
        FnLens(
            name="quiet",
            stages=[Stage.GATE],
            fn=lambda text, v: Answer(value="quietly" in v.input),
        ),
        FnLens(
            name="big",
            subject=Subject.TOOL_CALL,
            stages=[Stage.TOOL],
            fn=lambda text, v: Answer(value=v.tool_call.name == "pay"),
        ),
    ]
    rules = [  # noqa: RUF012
        Rule(
            "fraud_to_team",
            when=L("fraud").is_true,
            then=[Queue("fraud"), Alert("critical")],
            mode=ENFORCE,
            notice="The fraud team is alerted.",
        ),
        Rule("probe_in_test", when=L("probe").is_true, then=Block(), notice="Would stop it."),
        Rule(
            "quiet_review",
            when=L("quiet").is_true,
            then=[Queue("review"), Alert("high"), Notify()],
            mode=ENFORCE,
        ),
        Rule(
            "four_eyes",
            when=L("big").is_true,
            then=[Block("Needs a second person."), Queue("four-eyes")],
            mode=ENFORCE,
            notice="A colleague approves it.",
        ),
    ]


@pytest.mark.django_db(transaction=True)
class TestWhatTheClientIsTold:
    async def test_the_client_hears_what_happened_not_which_rule(self):
        guard = await start("test-guard", "My IBAN is NL91ABNA0417164300")
        notice = (await guard.gate()).notice
        assert set(notice) == {"turn_id", "stage", "outcome", "input"}
        with override_settings(AI_SDK_OVERSIGHT={"CLIENT_DETAIL": "full"}):
            guard = await start("test-guard", "My IBAN is NL91ABNA0417164300")
            assert (await guard.gate()).notice["rules"] == ["mask_pii"]

    async def test_messages_people_see_are_translatable_and_stored_as_text(self):
        from django.utils.functional import Promise
        from django_ai_sdk.oversight import stream
        from django_ai_sdk.oversight.guard import TOO_LONG_MESSAGE

        for message in (
            Block.DEFAULT_MESSAGE,
            TOO_LONG_MESSAGE,
            stream.HELD_OUTPUT,
            stream.WITHHELD_OUTPUT,
        ):
            assert isinstance(message, Promise)
        guard = await start("test-withhold")
        finished = await guard.finish("anything")
        assert type(finished.output) is str
        turn = await closed_turn(guard)
        row = await RuleMatch.objects.aget(turn=turn, rule="withhold")
        assert json.dumps(row.actions)


@pytest.mark.django_db(transaction=True)
class TestRuleNotices:
    """A rule's notice tells the person what it did, even a queue they would never see."""

    async def test_a_queue_is_told_at_the_gate(self):
        notice = (await (await start("test-told", "wire it now")).gate()).notice
        assert notice["outcome"] == "flagged"
        assert notice["notes"] == [
            {"kind": "notice", "also": [], "text": "The fraud team is alerted.", "stage": "gate"}
        ]

    async def test_what_the_person_cannot_see_is_not_told(self):
        # A queue, an alert or a notification without a notice: the client learns nothing.
        guard = await start("test-told", "do it quietly")
        assert (await guard.gate()).notice is None
        assert (await guard.finish("Done.")).notice is None
        assert [m.rule.name for m in guard.matched if m.acted] == ["quiet_review"]
        # A developer view still shows it.
        with override_settings(AI_SDK_OVERSIGHT={"CLIENT_DETAIL": "full"}):
            notice = (await (await start("test-told", "do it quietly")).gate()).notice
        assert notice["outcome"] == "flagged" and notice["rules"] == ["quiet_review"]

    async def test_a_rule_in_test_mode_tells_nothing(self):
        assert (await (await start("test-told", "probe")).gate()).notice is None
        with override_settings(AI_SDK_OVERSIGHT={"CLIENT_DETAIL": "full"}):
            notice = (await (await start("test-told", "probe")).gate()).notice
        assert "notes" not in notice
        [match] = notice["detail"]["matches"]
        assert match["notice"] == "Would stop it."

    async def test_a_refused_tool_call_is_told_once_with_the_answer(self):
        guard = await start("test-told", "pay the invoice")
        await guard.gate()
        assert await guard.tool_call(ToolCall(name="pay")) == "Needs a second person."
        assert await guard.tool_call(ToolCall(name="pay")) == "Needs a second person."
        finished = await guard.finish("It waits for a colleague.")
        assert finished.notice["notes"] == [
            {"kind": "block", "also": [], "text": "A colleague approves it.", "stage": "tool"}
        ]

    def test_the_notice_round_trips_and_can_be_picked_and_tuned(self):
        from django_ai_sdk.oversight.policy import pick

        rule = ToldPolicy.rules[0]
        assert Rule.from_dict(rule.to_dict()).notice == "The fraud team is alerted."
        [picked] = pick(ToldPolicy.rules, "fraud_to_team", notice="Told.")
        assert (picked.notice, rule.notice) == ("Told.", "The fraud team is alerted.")

    async def test_each_tool_call_keeps_its_own_answer(self):
        guard = await start("test-told", "pay the invoice")
        await guard.gate()
        assert await guard.tool_call(ToolCall(name="pay")) == "Needs a second person."
        assert await guard.tool_call(ToolCall(name="note")) is None
        await guard.finish("Done.")
        turn = await closed_turn(guard)
        rows = [o async for o in LensAnswer.objects.filter(turn=turn, lens="big")]
        assert sorted(o.value for o in rows) == [False, True]
