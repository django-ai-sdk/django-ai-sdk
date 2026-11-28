from __future__ import annotations

import contextlib
import json
from typing import Any
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

from django_ai_sdk.oversight.stream import HELD_OUTPUT, WITHHELD_OUTPUT
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
                subject=Subject.ANSWER,
                stages=[Stage.FINISH],
                fn=lambda text, v: withhold,
            ),
            PIILens(
                name="mail_out", subject=Subject.ANSWER, stages=[Stage.FINISH], kinds=["email"]
            ),
        ]
        rules = [
            Rule("withhold", when=L("withhold").is_true, then=Block("Withheld."), mode=ENFORCE),
            Rule("mask_mail", when=L("mail_out").is_true, then=Mask(), mode=ENFORCE),
        ]


@pytest.mark.django_db(transaction=True)
class TestAutoHoldsTheAnswer:
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
        outputs = [p["output"] for p in parts if p.get("type") == "tool-output-available"]
        assert outputs[-1] == {"result": str(WITHHELD_OUTPUT)}

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
            if p.get("type") == "tool-output-available" and p["output"] == {"result": str(HELD_OUTPUT)}
        )
        before, after = parts[: shown + 1], parts[shown + 1 :]
        assert [p["toolName"] for p in before if p.get("type") == "tool-input-start"] == ["lookup"]
        assert "NL91ABNA0417164300" not in json.dumps(before) and '"Jan"' not in json.dumps(before)
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
        assert not [p for p in parts if p["type"] == "reasoning-delta"]
        assert "jan@example.com" not in json.dumps(parts)
        assert storage.stored[0].content == "Write to [EMAIL]."
        assert not storage.stored[0].reasoning

    async def test_a_disconnect_during_finish_stores_no_answer_but_keeps_the_notices(self):
        import asyncio

        from django_ai_sdk.views.schemas import Message

        async def slow(text, context):
            await asyncio.sleep(0.5)
            return True

        @register_policy
        class SlowFinish(Policy):
            name = "test-held-slow"
            lenses = [
                FnLens(
                    name="bad",
                    subject=Subject.ANSWER,
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
            await asyncio.sleep(0.25)
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        assert [(m.content, m.finish_reason) for m in storage.stored] == [("", "cancelled")]
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
    async def test_nothing_the_model_wrote_reaches_the_client_or_the_store(self):
        from haystack.dataclasses import StreamingChunk

        def half_then_fail(reply):
            yield StreamingChunk(content="The secret plan is ")
            raise RuntimeError("model down")

        storage = Storage()
        agent = make_agent(["unused"], storage)
        with patch.object(MockChatGenerator, "_make_chunks", side_effect=half_then_fail):
            parts = await chat(agent, "plan?")
        assert "secret plan" not in json.dumps(parts)
        assert all("secret plan" not in (m.content or "") for m in storage.stored)
        turn = await Turn.objects.aget()
        assert turn.error and turn.outcome == ""


@pytest.mark.usefixtures("no_reflect_queue")
@pytest.mark.django_db(transaction=True)
class TestReceiptsOfAWithheldAnswer:
    @staticmethod
    def started(receipt: bool) -> tuple[Any, Any]:
        from django_ai_sdk import with_receipt
        from haystack.dataclasses import ChatMessage as HaystackChatMessage
        from haystack.dataclasses import ToolCall as HaystackToolCall
        from haystack.tools import Tool

        def start_workflow_run(workflow: str) -> dict:
            return {
                "run_id": "run_8f3a-01",
                "workflow": workflow,
                "status": "running",
                "workflow_id": "not an id: Jan de Vries",
                "note": "Started for Jan de Vries",
            }

        tool = Tool(
            name="start_workflow_run",
            description="Start a workflow run.",
            function=start_workflow_run,
            parameters={"type": "object", "properties": {"workflow": {"type": "string"}}},
        )
        if receipt:
            tool = with_receipt(tool, "run_id", "workflow_id", "workflow", "status")
        message = HaystackChatMessage.from_assistant(
            tool_calls=[
                HaystackToolCall(
                    tool_name=tool.name, arguments={"workflow": "intake"}, id="call_1"
                )
            ]
        )
        return tool, message

    async def test_the_stored_call_and_the_released_card_keep_only_the_receipt(self):
        tool, message = self.started(receipt=True)
        storage = Storage()
        agent = make_agent([message, "Started."], storage, tools=[tool], policy="test-withhold")
        parts = await chat(agent, "start intake")
        receipt = {"run_id": "run_8f3a-01", "workflow": "intake", "status": "running", "withheld": True}
        stored = storage.stored[-1].tool_calls
        assert [(c["arguments"], c["result"]) for c in stored] == [({}, {"result": receipt})]
        outputs = [p["output"] for p in parts if p.get("type") == "tool-output-available"]
        assert outputs[-1] == receipt
        assert "Jan" not in json.dumps(parts)

    async def test_a_tool_without_a_receipt_keeps_nothing(self):
        tool, message = self.started(receipt=False)
        storage = Storage()
        agent = make_agent([message, "Started."], storage, tools=[tool], policy="test-withhold")
        parts = await chat(agent, "start intake")
        assert [c["result"] for c in storage.stored[-1].tool_calls] == [None]
        outputs = [p["output"] for p in parts if p.get("type") == "tool-output-available"]
        assert outputs[-1] == {"result": str(WITHHELD_OUTPUT)}


class TestReceiptKeysAndValues:
    @staticmethod
    def tool(name: str, *keys: str) -> Any:
        from django_ai_sdk import with_receipt
        from haystack.tools import Tool

        tool = Tool(name=name, description=name, function=dict, parameters={"type": "object"})
        return with_receipt(tool, *keys) if keys else tool

    def test_subagents_report_their_tools_receipt_keys_at_any_depth(self):
        from types import SimpleNamespace

        from django_ai_sdk.agents.subagent import SubagentStreamFilter
        from django_ai_sdk.receipts import receipt_keys
        from haystack.tools import ComponentTool

        def delegate(name: str, *tools: Any) -> Any:
            stream_filter = SubagentStreamFilter(SimpleNamespace(tools=list(tools)), name=name)
            return ComponentTool(component=stream_filter, name=name, description=name)

        inner = delegate("ask_payroll", self.tool("start_payout", "claim_id"))
        outer = delegate("ask_care", self.tool("start", "run_id"), inner)
        tools = [self.tool("start", "workflow_id"), self.tool("lookup"), outer]
        assert receipt_keys(tools) == {
            "start": ("workflow_id", "run_id"),
            "start_payout": ("claim_id",),
        }

    @pytest.mark.parametrize(
        "value",
        ["NL91ABNA0417164300", "4111111111111111", 4111111111111111, "sk-proj-abcdefghijklmnopqrstuvwx"],
    )
    def test_a_value_that_looks_personal_or_secret_is_left_out(self, value):
        from django_ai_sdk.receipts import receipt

        kept = receipt({"run_id": "run_1", "ref": value}, ("run_id", "ref"))
        assert kept == {"run_id": "run_1", "withheld": True}
