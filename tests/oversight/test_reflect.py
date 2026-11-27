from __future__ import annotations

import json
import uuid
from typing import Any

import pytest
from django.test import override_settings
from django_ai_sdk.common import ChatMessage
from django_ai_sdk.oversight import Entities, L, Mask, Rule
from django_ai_sdk.oversight.guard import TurnGuard
from django_ai_sdk.oversight.models import RuleMatch
from django_ai_sdk.oversight.reflection import run_reflection
from django_ai_sdk.oversight.types import (
    Answer,
    Stage,
    Subject,
    TextSpan,
)

from tests.oversight.helpers import (
    ENFORCE,
    FnLens,
    closed_turn,
    kept,
    make_agent,
    policy,
    start,
)


@pytest.mark.django_db(transaction=True)
class TestReflect:
    async def _finished(self, name: str, answerer: Any, message_id: str = "") -> str:
        policy(
            name,
            [
                FnLens(
                    name="names",
                    answer=Entities,
                    subject=Subject.ANSWER,
                    stages=[Stage.REFLECT],
                    fn=answerer,
                )
            ],
            [Rule("mask_names", when=L("names").is_true, then=Mask(), mode=ENFORCE)],
        )
        guard = await start(name, "who?")
        guard.message_id = message_id
        await guard.finish("Ask Jan.")
        return str((await closed_turn(guard)).id)

    async def test_a_mask_that_cannot_point_at_what_it_found_masks_the_whole_answer(self):
        from django_ai_sdk.conversation.models import Message, Thread

        message_id = str(uuid.uuid4())
        await Message.objects.acreate(
            id=message_id,
            thread=await Thread.objects.acreate(),
            result=ChatMessage(id=message_id, role="assistant", content="Ask Jan.").model_dump(),
        )
        turn_id = await self._finished(
            "fx-ref-nospan", lambda lens, context: Answer(value=["Jan"]), message_id
        )
        await run_reflection(turn_id)
        assert (await RuleMatch.objects.aget(turn_id=turn_id, rule="mask_names")).acted
        message = await Message.objects.aget(id=message_id)
        assert message.result["content"] == "[WITHHELD]"

    async def test_a_judge_that_is_down_is_asked_again_and_fails_closed_at_the_last(self):
        from django_ai_sdk.oversight.models import Turn
        from django_ai_sdk.oversight.reflection import REFLECT_ATTEMPTS

        from django_ai_sdk.conversation.models import Message, Thread

        message_id = str(uuid.uuid4())
        await Message.objects.acreate(
            id=message_id,
            thread=await Thread.objects.acreate(),
            result=ChatMessage(id=message_id, role="assistant", content="Ask Jan.").model_dump(),
        )
        down = lambda lens, context: Answer(error="unavailable: ConnectionError")  # noqa: E731
        with override_settings(AI_SDK_OVERSIGHT={"FAIL_MODE": "closed"}):
            turn_id = await self._finished("fx-ref-down", down, message_id)
            for _ in range(REFLECT_ATTEMPTS - 1):
                assert await run_reflection(turn_id) == 0
                turn = await Turn.objects.aget(id=turn_id)
                assert turn.reflect_state == Turn.ReflectState.PENDING
                assert not await RuleMatch.objects.filter(turn_id=turn_id).aexists()
            await run_reflection(turn_id)
        turn = await Turn.objects.aget(id=turn_id)
        assert turn.reflect_state == Turn.ReflectState.DONE
        assert (await RuleMatch.objects.aget(turn_id=turn_id, rule="mask_names")).acted
        message = await Message.objects.aget(id=message_id)
        assert message.result["content"] == "[WITHHELD]"

    async def test_a_reason_cannot_quote_what_the_gate_masked(self):
        from django_ai_sdk.conversation.models import Message, Thread
        from django_ai_sdk.oversight.models import LensAnswer

        def who(text, context):
            at = context.input.find("Jan Jansen")
            spans = [TextSpan(start=at, end=at + 10, label="person")] if at >= 0 else []
            return Answer(value=["Jan Jansen"] if spans else [], spans=spans)

        name = policy(
            "fx-ref-quote",
            [
                FnLens(name="who", answer=Entities, stages=[Stage.GATE], fn=who),
                FnLens(
                    name="why",
                    stages=[Stage.REFLECT],
                    fn=lambda text, context: Answer(value=True, rationale=f"Quotes: {context.input}"),
                ),
            ],
            [],
        )
        question = "What does Jan Jansen earn?"
        with override_settings(AI_SDK_OVERSIGHT={"CAPTURE": "masked"}):
            guard = await start(name, question)
            guard.user_message_id = str(uuid.uuid4())
            await guard.gate()
            turn = await closed_turn(guard)
            await Message.objects.acreate(
                id=guard.user_message_id,
                thread=await Thread.objects.acreate(),
                result=ChatMessage(role="user", content=question).model_dump(),
            )
            assert (await kept(turn))["input"] == "What does [PERSON] earn?"
            await run_reflection(str(turn.id))
        why = await LensAnswer.objects.aget(turn=turn, lens="why")
        assert (await kept(turn))["texts"][str(why.id)] == {
            "rationale": "Quotes: What does [PERSON] earn?"
        }

    async def test_it_masks_the_stored_answer_the_record_keeps_masked_and_leaves_a_notice(self):
        from django_ai_sdk.conversation.models import Message, Thread
        from django_ai_sdk.protocols.vercel import VercelProtocolHandler

        answer = "Call +31 6 12345678 for it."
        with override_settings(AI_SDK_OVERSIGHT={"CAPTURE": "masked"}):
            guard = await TurnGuard.start(
                make_agent([], None, policy="test-masking"), channel="api", input="who?"
            )
            guard.message_id = str(uuid.uuid4())
            await guard.finish(answer)
            turn = await closed_turn(guard)
            stored = ChatMessage(
                id=guard.message_id,
                role="assistant",
                content=answer,
                reasoning="the number is +31 6 12345678",
            )
            await Message.objects.acreate(
                id=guard.message_id,
                thread=await Thread.objects.acreate(),
                result=stored.model_dump(),
            )
            assert (await kept(turn))["output"] == "Call [PHONE] for it."
            await run_reflection(str(turn.id))
        message = await Message.objects.aget(id=guard.message_id)
        assert message.result["content"] == "Call [PHONE] for it."
        assert message.result["reasoning"] == ""
        assert "12345678" not in json.dumps(await kept(turn))
        notice = message.result["metadata"]["oversight"][-1]
        assert (notice["stage"], notice["outcome"], notice["replaced"]) == (
            "reflect",
            "masked",
            True,
        )
        [shown] = VercelProtocolHandler().from_chat_messages([message.to_chat_message()])
        assert {"type": "data-oversight", "data": notice} in shown["parts"]


@pytest.mark.django_db(transaction=True)
async def test_a_lens_that_failed_after_an_unsure_answer_decides_nothing_at_reflect_either():
    from django_ai_sdk.oversight import Alert, Judge, JudgeLens
    from django_ai_sdk.oversight.reflection import judge_reflect_stage
    from django_ai_sdk.oversight.registry import build_policy

    from django_ai_sdk.oversight.types import TurnContext

    from tests.oversight.helpers import ENFORCE, broken, policy

    class Said(Judge):
        def __init__(self, fn: Any, name: str) -> None:
            self.fn, self.name = fn, name

        async def answer(self, lens: Any, context: Any) -> Answer:
            return self.fn(context.text(lens.subject), context)

    unsure = Said(lambda text, v: Answer(value=True, confidence=0.5), "unsure")
    lens = JudgeLens(
        name="risky",
        subject=Subject.ANSWER,
        stages=[Stage.REFLECT],
        judges=[unsure, Said(broken, "down")],
    )
    rule = Rule("r", when=L("risky").is_true, then=Alert("critical"), mode=ENFORCE)
    name = policy("test-reflect-kept", [lens], [rule])
    context = TurnContext(turn_id="t", input="q", output="a")
    answered, matched, _ = await judge_reflect_stage(
        build_policy(name), context, set(), set(), outcome="allowed"
    )
    assert answered[0].failed and matched == []
