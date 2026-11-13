"""A turn survives what goes wrong around it: errors, disconnects, crashes, lost tasks.

The turn is written when its gate ends (open) and updated at close; one that never
closes is marked interrupted by `oversight_sweep`."""

from __future__ import annotations

import asyncio
import contextlib
from datetime import timedelta
from io import StringIO
from typing import Any
from unittest.mock import patch

import pytest
from asgiref.sync import sync_to_async
from django.core.management import call_command
from django.db import DatabaseError, OperationalError
from django.test import override_settings
from django.utils import timezone
from django_ai_sdk.common import ChatMessage
from django_ai_sdk.errors import Blocked
from django_ai_sdk.oversight import (
    Answer,
    L,
    Mode,
    Notify,
    Policy,
    Rule,
    register_policy,
)
from django_ai_sdk.oversight.actions import Block, Queue
from django_ai_sdk.oversight.guard import TurnGuard
from django_ai_sdk.oversight.models import LensAnswer, ReviewItem, RuleMatch, Turn
from django_ai_sdk.oversight.notifiers import event_id
from django_ai_sdk.oversight.reflect import REFLECT_ATTEMPTS, aclaim_reflect, areflect
from django_ai_sdk.oversight.testing import dry_run
from django_ai_sdk.oversight.types import Stage, Subject
from django_tasks.backends.base import BaseTaskBackend
from haystack.components.generators.chat import MockChatGenerator

from tests.oversight.helpers import (
    FnLens,
    Storage,
    agent_with,
    chat,
    make_agent,
    race,
    start,
)

ENFORCE = Mode.ENFORCE


def slow_finish(seconds: float = 0.5) -> FnLens:
    async def slow(text: str, view: Any) -> bool:
        await asyncio.sleep(seconds)
        return False

    return FnLens(name="slow", subject=Subject.OUTPUT, stages=[Stage.FINISH], fn=slow)


@register_policy
class DurablePolicy(Policy):
    name = "test-durable"
    lenses = [  # noqa: RUF012
        FnLens(
            name="asked",
            stages=[Stage.GATE],
            fn=lambda text, view: Answer(value="refund" in view.input, rationale="it asks for one"),
        ),
        FnLens(name="banned", stages=[Stage.GATE], fn=lambda text, view: "forbidden" in view.input),
        FnLens(
            name="long",
            subject=Subject.OUTPUT,
            stages=[Stage.FINISH],
            fn=lambda text, view: len(view.output) > 20,
        ),
    ]
    rules = [  # noqa: RUF012
        Rule("refunds", when=L("asked").is_true, then=Queue("refunds"), mode=ENFORCE),
        Rule("ban", when=L("banned").is_true, then=Block("No."), mode=ENFORCE),
        Rule("long_answers", when=L("long").is_true, then=Queue("long"), mode=ENFORCE),
    ]


@register_policy
class SlowFinishPolicy(Policy):
    name = "test-durable-slow"
    lenses = [slow_finish()]  # noqa: RUF012
    rules = [  # noqa: RUF012
        Rule("never", when=L("slow").is_true, then=Block(), mode=ENFORCE),
    ]


async def close(guard: TurnGuard, **kwargs: Any) -> Turn:
    with (
        patch("django_ai_sdk.oversight.persist.enqueue_reflect"),
        patch("django_ai_sdk.oversight.persist.enqueue_notifications"),
    ):
        turn_id = await guard.close(**kwargs)
    return await Turn.objects.select_related("content").aget(id=turn_id)


async def sweep(*args: str) -> str:
    """The sweep's line about open turns."""
    out = StringIO()
    await sync_to_async(call_command)("oversight_sweep", *args, stdout=out)
    return out.getvalue().splitlines()[0]


async def age(turn_id: Any, minutes: int) -> None:
    """Make a turn look as if it started `minutes` ago."""
    await Turn.objects.filter(id=turn_id).aupdate(
        created_at=timezone.now() - timedelta(minutes=minutes)
    )


@register_policy
class ReflectingPolicy(Policy):
    name = "test-durable-reflect"
    lenses = [  # noqa: RUF012
        FnLens(
            name="later",
            subject=Subject.OUTPUT,
            stages=[Stage.REFLECT],
            fn=lambda text, view: "bad" in view.output,
        ),
    ]
    rules = [  # noqa: RUF012
        Rule("tell_later", when=L("later").is_true, then=Notify(), mode=ENFORCE),
    ]


async def reflect_pending(*args: str) -> str:
    """The sweep's line about lost reflection."""
    out = StringIO()
    await sync_to_async(call_command)("oversight_sweep", *args, stdout=out)
    return out.getvalue().splitlines()[1]


async def expire_lease(turn_id: Any) -> None:
    """What a worker killed after its claim leaves: a lease that ran out."""
    await Turn.objects.filter(id=turn_id).aupdate(
        reflect_lease_until=timezone.now() - timedelta(seconds=1)
    )


class WorkerBackend(BaseTaskBackend):
    """A backend with a worker somewhere else: tasks leave the request."""

    def enqueue(self, task: Any, args: Any, kwargs: Any) -> Any:
        raise NotImplementedError


@pytest.mark.django_db(transaction=True)
class TestWriteAhead:
    async def test_the_gate_writes_the_turn_open_with_what_it_decided(self):
        guard = await start("test-durable", "I want a refund")
        await guard.gate()
        turn = await Turn.objects.aget(id=guard.id)
        assert turn.status == Turn.Status.OPEN
        # Something acted: the row says so already. Nothing else is decided yet.
        assert turn.outcome == "flagged"
        assert turn.labels == {"asked": True, "banned": False}
        assert await RuleMatch.objects.filter(turn=turn, rule="refunds", acted=True).aexists()
        assert await ReviewItem.objects.filter(turn=turn, queue="refunds").aexists()
        assert turn.content_id is None

    async def test_an_open_turn_with_nothing_acted_has_a_blank_outcome(self):
        guard = await start("test-durable", "hello")
        await guard.gate()
        turn = await Turn.objects.aget(id=guard.id)
        assert (turn.status, turn.outcome, turn.report_outcome) == ("open", "", "")

    async def test_close_updates_the_row_and_adds_only_what_is_new(self):
        guard = await start("test-durable", "I want a refund")
        await guard.gate()
        await guard.finish("Here is a long answer about refunds.")
        turn = await close(guard)
        assert turn.status == Turn.Status.CLOSED
        assert turn.outcome == "flagged"
        assert turn.labels == {"asked": True, "banned": False, "long": True}
        assert turn.latency_ms is not None
        assert await Turn.objects.acount() == 1
        assert await LensAnswer.objects.filter(turn=turn).acount() == 3
        rules = [d.rule async for d in RuleMatch.objects.filter(turn=turn).order_by("created_at")]
        assert rules == ["refunds", "long_answers"]
        queues = {r.queue async for r in ReviewItem.objects.filter(turn=turn)}
        assert queues == {"refunds", "long"}

    @override_settings(AI_SDK_OVERSIGHT={"CAPTURE": "full"})
    async def test_what_the_gate_judges_said_goes_with_the_content_at_close(self):
        guard = await start("test-durable", "I want a refund")
        await guard.gate()
        await guard.finish("Sure.")
        turn = await close(guard)
        asked = await LensAnswer.objects.aget(turn=turn, lens="asked")
        assert turn.content.stored["texts"][str(asked.id)] == {"rationale": "it asks for one"}

    async def test_a_turn_refused_at_the_gate_is_written_there(self):
        guard = await start("test-durable", "something forbidden")
        assert (await guard.gate()).blocked
        turn = await Turn.objects.aget(id=guard.id)
        assert (turn.status, turn.outcome) == ("open", "blocked")
        turn = await close(guard)
        assert (turn.status, turn.outcome) == ("closed", "blocked")

    async def test_a_turn_refused_for_its_history_is_written_there(self):
        guard = await start("test-durable", "hello")
        messages = [
            ChatMessage(role="user", content="something forbidden"),
            ChatMessage(role="assistant", content="No."),
            ChatMessage(role="user", content="hello"),
        ]
        assert (await guard.gate_history(messages)).blocked
        turn = await Turn.objects.aget(id=guard.id)
        assert (turn.status, turn.outcome) == ("open", "blocked")

    async def test_a_write_ahead_that_fails_is_written_whole_at_close(self):
        from django_ai_sdk.oversight.guard import state

        guard = await start("test-durable", "I want a refund")
        with patch.object(state, "persist_turn", side_effect=DatabaseError("down")):
            await guard.gate()
        assert not await Turn.objects.filter(id=guard.id).aexists()
        await guard.finish("ok")
        turn = await close(guard)
        assert turn.status == Turn.Status.CLOSED
        assert await RuleMatch.objects.filter(turn=turn, rule="refunds").acount() == 1

    async def test_a_dry_run_writes_nothing(self):
        result = await dry_run("test-durable", input="I want a refund", output="ok")
        assert result.outcome == "flagged"
        assert await Turn.objects.acount() == 0


@pytest.mark.django_db(transaction=True)
class TestCrashedTurns:
    async def test_a_turn_whose_run_failed_has_a_blank_outcome(self):
        guard = await start("test-durable", "hello")
        await guard.gate()
        turn = await close(guard, error="RuntimeError")
        assert (turn.status, turn.outcome, turn.error) == ("closed", "", "RuntimeError")

    async def test_a_failed_run_still_says_what_a_rule_did(self):
        guard = await start("test-durable", "I want a refund")
        await guard.gate()
        turn = await close(guard, error="RuntimeError")
        assert turn.outcome == "flagged"

    async def test_a_run_that_raised_is_recorded_with_its_error(self):
        # Haystack skips after_run on an exception: run_agent closes the turn then.
        agent, model = agent_with("never", policy="test-durable")

        with (
            patch.object(type(model), "run", side_effect=ValueError("broken tool")),
            pytest.raises(ValueError),
        ):
            await agent.run([ChatMessage(role="user", content="hi")], channel="api")
        turn = await Turn.objects.aget()
        assert (turn.status, turn.outcome, turn.error) == ("closed", "", "ValueError")

    async def test_a_blocked_run_is_blocked_not_blank(self):
        agent, _ = agent_with("never", policy="test-durable")

        with pytest.raises(Blocked):
            await agent.run([ChatMessage(role="user", content="forbidden")], channel="api")
        turn = await Turn.objects.aget()
        assert (turn.status, turn.outcome, turn.error) == ("closed", "blocked", "")


@pytest.mark.django_db(transaction=True)
class TestStreamEnds:
    async def test_a_model_error_mid_stream_is_recorded_on_the_turn(self):
        storage = Storage()
        agent = make_agent(["unused"], storage, policy="test-durable")
        with patch.object(MockChatGenerator, "_make_chunks", side_effect=RuntimeError("down")):
            await chat(agent, "hello")
        turn = await Turn.objects.aget()
        assert turn.status == Turn.Status.CLOSED
        assert turn.error and turn.error != "disconnected"
        assert turn.outcome == ""
        assert storage.stored[-1].finish_reason == "error"

    async def test_a_finished_chat_is_closed_and_allowed(self):
        await chat(make_agent(["Fine."], Storage(), policy="test-durable"), "hello")
        turn = await Turn.objects.aget()
        assert (turn.status, turn.outcome, turn.error) == ("closed", "allowed", "")

    async def test_a_client_gone_before_finish_is_recorded_as_disconnected(self):
        from django_ai_sdk.views.schemas import Message

        agent = make_agent(["The plan is X."], Storage(), policy="test-durable-slow")
        message = Message.model_validate(
            {"role": "user", "parts": [{"type": "text", "text": "plan?"}]}
        )
        with override_settings(AI_SDK_OVERSIGHT={"DEADLINE_MS": {"finish": 5000}}):
            response = await agent.as_view([message])

            async def consume() -> None:
                async for _ in response.streaming_content:
                    pass

            task = asyncio.create_task(consume())
            await asyncio.sleep(0.25)  # the model is done, finish is still judging
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        turn = await Turn.objects.aget()
        assert (turn.status, turn.outcome, turn.error) == ("closed", "", "disconnected")

    async def test_a_response_never_read_records_and_stores_nothing(self):
        # The gate runs inside the run, so an unread response never reached the gate.
        from django_ai_sdk.views.schemas import Message

        storage = Storage()
        agent = make_agent(["Fine."], storage, policy="test-durable")
        message = Message.model_validate(
            {"role": "user", "parts": [{"type": "text", "text": "I want a refund"}]}
        )
        await agent.as_view([message])
        assert not await Turn.objects.aexists()
        assert storage.stored == []


@pytest.mark.django_db(transaction=True)
class TestSweep:
    async def open_turn(self, minutes: int, text: str = "hello") -> str:
        guard = await start("test-durable", text)
        await guard.gate()
        await age(guard.id, minutes)
        return guard.id

    async def test_a_worker_killed_mid_turn_leaves_the_gate_record_to_sweep(self):
        # The guard is dropped without close(): what a killed worker leaves.
        turn_id = await self.open_turn(45, "I want a refund")
        assert await sweep("--stale-after", "30m") == "1 turn swept"
        turn = await Turn.objects.aget(id=turn_id)
        assert turn.status == Turn.Status.INTERRUPTED
        assert await RuleMatch.objects.filter(turn=turn, rule="refunds").aexists()

    async def test_fresh_and_closed_turns_are_left_alone(self):
        fresh = await self.open_turn(5)
        guard = await start("test-durable", "hello")
        await guard.gate()
        closed = await close(guard)
        await age(closed.id, 120)
        assert await sweep("--stale-after", "30m") == "0 turns swept"
        assert (await Turn.objects.aget(id=fresh)).status == "open"
        assert (await Turn.objects.aget(id=closed.id)).status == "closed"

    async def test_dry_run_and_limit(self):
        for _ in range(3):
            await self.open_turn(60)
        assert await sweep("--dry-run") == "3 turns would be swept"
        assert await Turn.objects.filter(status="open").acount() == 3
        assert await sweep("--limit", "2") == "2 turns swept"
        assert await Turn.objects.filter(status="open").acount() == 1

    async def test_a_turn_that_closes_after_the_sweep_is_closed(self):
        guard = await start("test-durable", "hello")
        await guard.gate()
        await age(guard.id, 60)
        await sweep()
        await guard.finish("Fine.")
        turn = await close(guard)
        assert (turn.status, turn.outcome) == ("closed", "allowed")

    async def test_a_bad_age_is_a_command_error(self):
        from django.core.management.base import CommandError

        with pytest.raises(CommandError):
            await sweep("--stale-after", "soon")
        with pytest.raises(CommandError):
            await sweep("--reflect-after", "soon")


class TestParseAge:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("90s", timedelta(seconds=90)),
            ("30m", timedelta(minutes=30)),
            ("2h", timedelta(hours=2)),
            ("1d", timedelta(days=1)),
            ("15", timedelta(minutes=15)),
        ],
    )
    def test_ages(self, text: str, expected: timedelta) -> None:
        from django_ai_sdk.management.sweep import parse_age

        assert parse_age(text) == expected

    def test_anything_else_is_refused(self) -> None:
        from django_ai_sdk.management.sweep import parse_age

        with pytest.raises(ValueError):
            parse_age("-5m")


@pytest.mark.django_db(transaction=True)
class TestReflectLease:
    async def closed(self, output: str = "a bad answer") -> str:
        guard = await start("test-durable-reflect", "hello")
        await guard.gate()
        await guard.finish(output)
        turn = await close(guard)
        assert turn.reflect_state == Turn.ReflectState.PENDING
        return str(turn.id)

    async def test_a_reflected_turn_is_done_once(self):
        turn_id = await self.closed()
        with patch("django_ai_sdk.oversight.persist.enqueue_notifications"):
            assert await areflect(turn_id) == 1
            # A duplicate delivery finds it done.
            assert await areflect(turn_id) == 0
        turn = await Turn.objects.aget(id=turn_id)
        assert (turn.reflect_state, turn.reflect_attempts) == ("done", 1)
        assert turn.reflected_at is not None and turn.reflect_lease_until is None
        assert await LensAnswer.objects.filter(turn_id=turn_id, lens="later").acount() == 1

    async def test_two_deliveries_at_once_reflect_it_once(self):
        turn_id = await self.closed()
        with patch("django_ai_sdk.oversight.persist.enqueue_notifications"):
            results = await asyncio.gather(areflect(turn_id), areflect(turn_id))
        assert sorted(results) == [0, 1]
        assert await RuleMatch.objects.filter(turn_id=turn_id, rule="tell_later").acount() == 1

    async def test_a_claim_held_by_a_live_worker_is_not_taken(self):
        turn_id = await self.closed()
        assert await aclaim_reflect(turn_id) is not None
        assert await areflect(turn_id) == 0
        assert (await Turn.objects.aget(id=turn_id)).reflect_state == "running"

    async def test_a_lease_that_ran_out_is_claimed_again(self):
        turn_id = await self.closed()
        assert await aclaim_reflect(turn_id) is not None  # its worker dies here
        await expire_lease(turn_id)
        with patch("django_ai_sdk.oversight.persist.enqueue_notifications"):
            assert await areflect(turn_id) == 1
        turn = await Turn.objects.aget(id=turn_id)
        assert (turn.reflect_state, turn.reflect_attempts) == ("done", 2)

    async def test_a_run_that_lost_its_lease_writes_nothing(self):
        from django_ai_sdk.oversight.reflect import _commit_reflection, _judge

        turn_id = await self.closed()
        stale_lease = await aclaim_reflect(turn_id)
        turn = await Turn.objects.select_related("content").aget(id=turn_id)
        judged = await _judge(turn)
        assert judged is not None
        # Too slow: the lease ran out, and another run reflected the turn.
        await expire_lease(turn_id)
        with patch("django_ai_sdk.oversight.persist.enqueue_notifications"):
            assert await areflect(turn_id) == 1
        kept = await sync_to_async(_commit_reflection)(turn, *judged, lease=stale_lease)
        assert kept is False
        assert await RuleMatch.objects.filter(turn_id=turn_id, rule="tell_later").acount() == 1

    async def test_a_run_that_fails_gives_its_claim_back(self):
        turn_id = await self.closed("fine")
        with patch("django_ai_sdk.oversight.reflect.save_answers", side_effect=RuntimeError("db")):
            with pytest.raises(RuntimeError):
                await areflect(turn_id)
        assert (await Turn.objects.aget(id=turn_id)).reflected_at is None
        assert await areflect(turn_id) == 1

    async def test_an_after_effect_that_fails_writes_nothing_and_is_retried(self):
        turn_id = await self.closed()
        with (
            patch("django_ai_sdk.oversight.reflect._after_effects", side_effect=RuntimeError("x")),
            pytest.raises(RuntimeError),
        ):
            await areflect(turn_id)
        # Its answers and matches went with it: the reflection is all or nothing.
        assert (await Turn.objects.aget(id=turn_id)).reflect_state == "pending"
        assert not await RuleMatch.objects.filter(turn_id=turn_id, rule="tell_later").aexists()
        with patch("django_ai_sdk.oversight.persist.enqueue_notifications"):
            assert await areflect(turn_id) == 1
        assert await RuleMatch.objects.filter(turn_id=turn_id, rule="tell_later").aexists()
        assert (await Turn.objects.aget(id=turn_id)).reflect_state == "done"

    async def test_a_policy_that_did_not_load_gives_the_claim_back(self):
        turn_id = await self.closed("fine")
        with patch("django_ai_sdk.oversight.resolve.aget_policy", return_value=None):
            assert await areflect(turn_id) == 0
        assert (await Turn.objects.aget(id=turn_id)).reflect_state == "pending"
        assert await areflect(turn_id) == 1

    async def test_a_turn_that_cannot_be_read_gives_the_claim_back(self):
        turn_id = await self.closed("fine")
        with (
            patch.object(Turn.objects, "select_related", side_effect=OperationalError("db")),
            pytest.raises(OperationalError),
        ):
            await areflect(turn_id)
        assert (await Turn.objects.aget(id=turn_id)).reflect_state == "pending"

    async def test_a_turn_that_keeps_failing_is_marked_failed(self):
        turn_id = await self.closed()
        with patch("django_ai_sdk.oversight.reflect.save_answers", side_effect=RuntimeError("db")):
            for _ in range(REFLECT_ATTEMPTS):
                with pytest.raises(RuntimeError):
                    await areflect(turn_id)
        turn = await Turn.objects.aget(id=turn_id)
        assert (turn.reflect_state, turn.reflect_attempts) == ("failed", REFLECT_ATTEMPTS)
        assert turn.error == "reflect failed"
        assert await areflect(turn_id) == 0

    async def test_a_turn_without_reflect_lenses_is_never_claimed(self):
        guard = await start("test-durable", "hello")
        await guard.gate()
        await guard.finish("Fine.")
        turn = await close(guard)
        assert turn.reflect_state == ""
        assert await areflect(str(turn.id)) == 0


@pytest.mark.django_db(transaction=True)
class TestReflectPending:
    async def lost(self) -> str:
        """A closed turn whose reflect task was never queued."""
        guard = await start("test-durable-reflect", "hello")
        await guard.gate()
        await guard.finish("a bad answer")
        return str((await close(guard)).id)

    async def test_a_lost_enqueue_is_queued_again(self):
        turn_id = await self.lost()
        assert await reflect_pending() == "0 turns queued"
        await age(turn_id, 30)
        with patch("django_ai_sdk.oversight.persist.enqueue_notifications"):
            assert await reflect_pending("--reflect-after", "10m") == "1 turn queued"
        # The test settings run tasks at once.
        assert (await Turn.objects.aget(id=turn_id)).reflect_state == "done"

    async def test_a_dead_workers_turn_is_queued_again(self):
        turn_id = await self.lost()
        await aclaim_reflect(turn_id)
        assert await reflect_pending() == "0 turns queued"
        await expire_lease(turn_id)
        with patch("django_ai_sdk.oversight.persist.enqueue_notifications"):
            assert await reflect_pending() == "1 turn queued"
        assert (await Turn.objects.aget(id=turn_id)).reflect_state == "done"

    async def test_a_dead_workers_last_attempt_is_marked_failed(self):
        turn_id = await self.lost()
        await Turn.objects.filter(id=turn_id).aupdate(reflect_attempts=REFLECT_ATTEMPTS - 1)
        await aclaim_reflect(turn_id)
        await expire_lease(turn_id)
        assert await reflect_pending() == "0 turns queued"
        turn = await Turn.objects.aget(id=turn_id)
        # The same as when reflect itself gives up.
        assert (turn.reflect_state, turn.error) == ("failed", "reflect failed")

    async def test_dry_run_queues_nothing(self):
        turn_id = await self.lost()
        await age(turn_id, 30)
        assert await reflect_pending("--dry-run") == "1 turn would be queued"
        assert (await Turn.objects.aget(id=turn_id)).reflect_state == "pending"


@pytest.mark.django_db(transaction=True)
class TestStableEventIds:
    async def test_a_turns_notification_has_the_same_id_each_time(self):
        guard = await start("test-durable-reflect", "hello")
        await guard.gate()
        await guard.finish("a bad answer")
        turn_id = str((await close(guard)).id)
        sent: list[dict[str, Any]] = []
        with patch(
            "django_ai_sdk.oversight.persist.enqueue_notifications",
            side_effect=lambda events: sent.extend(e for e, _ in events),
        ):
            await areflect(turn_id)
        assert [e["id"] for e in sent] == [event_id(turn_id, "tell_later", "reflect")]

    async def test_the_gates_notifications_repeat_their_ids(self):
        @register_policy
        class Noisy(Policy):
            name = "test-durable-noisy"
            lenses = [  # noqa: RUF012
                FnLens(name="always", stages=[Stage.GATE], fn=lambda text, view: True)
            ]
            rules = [  # noqa: RUF012
                Rule("tell", when=L("always").is_true, then=Notify(), mode=ENFORCE)
            ]

        guard = await start("test-durable-noisy", "hello")
        with patch("django_ai_sdk.oversight.persist.enqueue_notifications"):
            await guard.gate()
        first, again = guard.notifications(guard.matched), guard.notifications(guard.matched)
        assert [e["id"] for e, _ in first] == [e["id"] for e, _ in again]
        assert first[0][0]["id"] == event_id(guard.id, "tell", "gate")

    def test_the_webhook_sends_the_event_id(self):
        from django_ai_sdk.oversight.notifiers import WebhookNotifier

        hook = WebhookNotifier("https://example.test/hook", secret="s")
        event = {"id": event_id("t", "r", "gate"), "type": "oversight.match"}
        _, headers = hook._request(event)
        _, repeated = hook._request(event)
        assert headers["webhook-id"] == repeated["webhook-id"] == event["id"]

    def test_notify_sends_to_the_notifiers_it_names(self):
        from django_ai_sdk.oversight.notifiers import (
            LogNotifier,
            WebhookNotifier,
            configured_notifiers,
        )

        log, hook = LogNotifier(), WebhookNotifier("https://example.test/hook")
        assert (log.name, hook.name) == ("log", "webhook")
        with override_settings(AI_SDK_OVERSIGHT={"NOTIFIERS": [log, hook]}):
            assert configured_notifiers(Notify(to=["log"]).to) == [log]
            assert configured_notifiers(Notify().to) == [log, hook]
        assert Notify(to=["log"]).to_dict() == {
            "kind": "notify",
            "to": ["log"],
            "severity": "warning",
        }


class TestTaskBackendCheck:
    def ids(self) -> list[str]:
        from django_ai_sdk.oversight.checks import check_oversight_deploy

        return [m.id for m in check_oversight_deploy() if m.id == "ai_sdk.oversight.W014"]

    def test_tasks_run_inline_warns(self):
        # The test settings use the immediate backend.
        assert self.ids() == ["ai_sdk.oversight.W014"]

    def test_tasks_never_run_warns(self):
        dummy = {"default": {"BACKEND": "django_tasks.backends.dummy.DummyBackend"}}
        with override_settings(TASKS=dummy):
            assert self.ids() == ["ai_sdk.oversight.W014"]

    def test_a_worker_backend_says_nothing(self):
        worker = {"default": {"BACKEND": f"{__name__}.WorkerBackend"}}
        with override_settings(TASKS=worker):
            assert self.ids() == []

    def test_off_says_nothing(self):
        with override_settings(AI_SDK_OVERSIGHT={"ENABLED": False}):
            assert self.ids() == []


@pytest.mark.django_db(transaction=True)
@pytest.mark.usefixtures("postgres_only")
class TestRacesOnPostgres:
    """Claims race on separate connections; SQLite cannot show it (`DATABASE_URL`)."""

    def closed(self) -> str:
        from asgiref.sync import async_to_sync

        async def make() -> str:
            guard = await start("test-durable-reflect", "hello")
            await guard.gate()
            await guard.finish("a bad answer")
            return str((await close(guard)).id)

        return async_to_sync(make)()

    def test_one_of_many_claims_wins(self):
        from asgiref.sync import async_to_sync

        turn_id = self.closed()
        leases = race(*[lambda: async_to_sync(aclaim_reflect)(turn_id)] * 8)
        assert sum(lease is not None for lease in leases) == 1
        turn = Turn.objects.get(id=turn_id)
        assert (turn.reflect_state, turn.reflect_attempts) == ("running", 1)

    def test_many_deliveries_reflect_once(self):
        from asgiref.sync import async_to_sync

        turn_id = self.closed()
        with patch("django_ai_sdk.oversight.persist.enqueue_notifications"):
            added = race(*[lambda: async_to_sync(areflect)(turn_id)] * 6)
        assert sorted(added) == [0, 0, 0, 0, 0, 1]
        assert RuleMatch.objects.filter(turn_id=turn_id, rule="tell_later").count() == 1
        assert Turn.objects.get(id=turn_id).reflect_state == "done"

    def test_a_sweep_and_a_close_at_once_end_closed(self):
        from asgiref.sync import async_to_sync

        async def opened() -> TurnGuard:
            guard = await start("test-durable", "hello")
            await guard.gate()
            await guard.finish("Fine.")
            return guard

        guard = async_to_sync(opened)()
        Turn.objects.filter(id=guard.id).update(created_at=timezone.now() - timedelta(hours=1))

        race(lambda: async_to_sync(sweep)(), lambda: async_to_sync(close)(guard))
        # The sweep marks only a turn still open: whichever comes first, it ends closed.
        assert Turn.objects.get(id=guard.id).status == "closed"
        assert Turn.objects.filter(id=guard.id).count() == 1
