"""What reflect does to the stored answer: masks it, leaves a notice, or leaves it alone.

How a reflection is claimed, retried and given back: test_durability.TestReflectLease.
"""

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
from django_ai_sdk.oversight.reflect import areflect
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
                    subject=Subject.OUTPUT,
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
            "fx-ref-nospan", lambda lens, view: Answer(value=["Jan"]), message_id
        )
        await areflect(turn_id)
        assert (await RuleMatch.objects.aget(turn_id=turn_id, rule="mask_names")).acted
        message = await Message.objects.aget(id=message_id)
        assert message.result["content"] == "[WITHHELD]"

    async def test_a_reason_cannot_quote_what_the_gate_masked(self):
        from django_ai_sdk.conversation.models import Message, Thread
        from django_ai_sdk.oversight.models import LensAnswer

        def who(text, view):
            at = view.input.find("Jan Jansen")
            spans = [TextSpan(start=at, end=at + 10, label="person")] if at >= 0 else []
            return Answer(value=["Jan Jansen"] if spans else [], spans=spans)

        name = policy(
            "fx-ref-quote",
            [
                FnLens(name="who", answer=Entities, stages=[Stage.GATE], fn=who),
                FnLens(
                    name="why",
                    stages=[Stage.REFLECT],
                    fn=lambda text, view: Answer(value=True, rationale=f"Quotes: {view.input}"),
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
            await areflect(str(turn.id))
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
            # The record never held the number; the thread did, and reflect fixes that copy.
            assert (await kept(turn))["output"] == "Call [PHONE] for it."
            await areflect(str(turn.id))
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
