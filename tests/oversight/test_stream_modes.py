"""auto and live: what the person sees while an answer may still be withheld."""

from __future__ import annotations

import contextlib
import json
from unittest.mock import patch

import pytest
from django.test import override_settings
from django_ai_sdk.oversight import (
    Block,
    L,
    Mask,
    PIILens,
    Policy,
    Rule,
    register_policy,
)
from django_ai_sdk.oversight.models import Turn
from django_ai_sdk.oversight.types import Stage, Subject
from haystack.components.generators.chat import MockChatGenerator

from tests.factories.db import UserFactory
from tests.oversight.helpers import (
    ENFORCE,
    FnLens,
    Storage,
    a_thread,
    call,
    chat,
    chat_turns,
    lookup_tool,
    make_agent,
    oversight_parts,
    text_of,
)


def held_policy(policy: str, withhold: bool) -> None:
    @register_policy
    class Held(Policy):
        name = policy
        lenses = [
            FnLens(
                name="withhold",
                subject=Subject.OUTPUT,
                stages=[Stage.FINISH],
                fn=lambda text, v: withhold,
            ),
            PIILens(
                name="mail_out", subject=Subject.OUTPUT, stages=[Stage.FINISH], kinds=["email"]
            ),
        ]
        rules = [
            Rule("withhold", when=L("withhold").is_true, then=Block("Withheld."), mode=ENFORCE),
            Rule("mask_mail", when=L("mail_out").is_true, then=Mask(), mode=ENFORCE),
        ]


@pytest.mark.django_db(transaction=True)
class TestAutoHoldsTheAnswer:
    """What an answer carries besides its text waits for finish too."""

    async def test_a_withheld_answer_shows_and_stores_no_tool_output(self):
        held_policy("test-held-no", withhold=True)
        tool, call = lookup_tool()
        storage = Storage()
        parts = await chat(
            make_agent([call, "Done."], storage, tools=[tool], policy="test-held-no"), "Jan?"
        )
        assert text_of(parts) == "Withheld."
        assert "NL91ABNA0417164300" not in json.dumps(parts)
        assert "NL91ABNA0417164300" not in json.dumps(storage.stored[0].tool_calls)
        # The card that showed the tool running is closed, saying why it stays empty.
        outputs = [p["output"] for p in parts if p.get("type") == "tool-output-available"]
        assert list(outputs[-1]) == ["withheld"]

    async def test_a_held_answer_shows_its_tools_running_then_what_they_found(self):
        held_policy("test-held-yes", withhold=False)
        tool, call = lookup_tool()
        parts = await chat(
            make_agent([call, "Done."], Storage(), tools=[tool], policy="test-held-yes"), "Jan?"
        )
        assert text_of(parts) == "Done."
        shown = next(
            i
            for i, p in enumerate(parts)
            if p.get("type") == "tool-output-available" and "held" in p["output"]
        )
        before, after = parts[: shown + 1], parts[shown + 1 :]
        # While it works: the tool's name and progress, none of what it was given or found.
        assert [p["toolName"] for p in before if p.get("type") == "tool-input-start"] == ["lookup"]
        assert "NL91ABNA0417164300" not in json.dumps(before) and '"Jan"' not in json.dumps(before)
        # Once it passed: the real input and output.
        assert any(
            p.get("type") == "tool-input-available" and p["input"] == {"name": "Jan"} for p in after
        )
        assert "NL91ABNA0417164300" in json.dumps(after)

    async def test_a_held_answer_shows_and_keeps_no_reasoning(self):
        from dataclasses import replace

        from haystack.dataclasses import ReasoningContent

        held_policy("test-held-mask", withhold=False)
        original = MockChatGenerator._make_chunks

        def thinking_first(self, reply):
            chunks = original(self, reply)
            thought = ReasoningContent(reasoning_text="Their address is jan@example.com, so")
            return [replace(chunks[0], content="", reasoning=thought, start=False), *chunks]

        storage = Storage()
        with patch.object(MockChatGenerator, "_make_chunks", thinking_first):
            parts = await chat(
                make_agent(["Write to jan@example.com."], storage, policy="test-held-mask"), "who?"
            )
        # Held reasoning is never shown, so never judged: it is not stored either.
        assert not [p for p in parts if p["type"] == "reasoning-delta"]
        assert "jan@example.com" not in json.dumps(parts)
        assert storage.stored[0].content == "Write to [EMAIL]."
        assert not storage.stored[0].reasoning

    async def test_a_disconnect_during_finish_stores_no_answer_but_keeps_the_notices(self):
        import asyncio

        from django_ai_sdk.views.schemas import Message

        async def slow(text, view):
            await asyncio.sleep(0.5)
            return True

        @register_policy
        class SlowFinish(Policy):
            name = "test-held-slow"
            lenses = [
                FnLens(
                    name="bad",
                    subject=Subject.OUTPUT,
                    stages=[Stage.FINISH],
                    fn=slow,
                )
            ]
            rules = [Rule("withhold", when=L("bad").is_true, then=Block("No."), mode=ENFORCE)]

        storage = Storage()
        agent = make_agent(["The secret plan is X."], storage, policy="test-held-slow")
        message = Message.model_validate(
            {"role": "user", "parts": [{"type": "text", "text": "plan?"}]}
        )
        settings = {"DEADLINE_MS": {"finish": 5000}, "CLIENT_DETAIL": "full"}
        with override_settings(AI_SDK_OVERSIGHT=settings):
            response = await agent.as_view([message])

            async def consume():
                async for _ in response.streaming_content:
                    pass

            task = asyncio.create_task(consume())
            await asyncio.sleep(0.25)  # the model is done, finish is still running
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        assert [(m.content, m.finish_reason) for m in storage.stored] == [("", "cancelled")]
        # The gate's notice reached the client before it left: a reload shows it again.
        assert [n["stage"] for n in storage.stored[0].metadata["oversight"]] == ["gate"]


@pytest.mark.usefixtures("no_reflect_queue")
@pytest.mark.django_db(transaction=True)
class TestWithheldArtifactRows:
    async def test_artifact_tool_rows_go_with_a_withheld_answer(self):
        from django_ai_sdk.artifacts.models import Artifact
        from django_ai_sdk.artifacts.schemas import SnippetArtifact

        user = await UserFactory.acreate()
        thread = await a_thread(user)
        tool = SnippetArtifact.as_tool(str(thread.id), user)
        agent = make_agent(
            [call(tool.name, code="launch code 0000"), "Here it is."],
            Storage(),
            tools=[tool],
            policy="test-withhold",
        )
        parts = await chat_turns(
            agent, [("user", "a snippet")], thread_id=str(thread.id), user=user
        )
        assert [n.get("replace") for n in oversight_parts(parts)] == ["No."]
        assert not await Artifact.objects.filter(thread_id=thread.id).aexists()


@pytest.mark.django_db(transaction=True)
class TestModelErrorWhileHeld:
    """A model that fails mid-stream under auto: what it wrote was held, so none of it leaks.

    The oversight record keeps the partial answer (masked as `CAPTURE` says): it is what
    the model said, and the turn's error says it never finished.
    """

    async def test_nothing_the_model_wrote_reaches_the_client_or_the_store(self):
        from haystack.dataclasses import StreamingChunk

        def half_then_fail(reply):
            yield StreamingChunk(content="The secret plan is ")
            raise RuntimeError("model down")

        storage = Storage()
        agent = make_agent(["unused"], storage)  # test-guard enforces finish rules: auto holds
        with patch.object(MockChatGenerator, "_make_chunks", side_effect=half_then_fail):
            parts = await chat(agent, "plan?")
        assert "secret plan" not in json.dumps(parts)
        assert all("secret plan" not in (m.content or "") for m in storage.stored)
        turn = await Turn.objects.aget()
        assert turn.error and turn.outcome == ""
