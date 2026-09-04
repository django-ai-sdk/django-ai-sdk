from __future__ import annotations

import uuid

import pytest
from asgiref.sync import sync_to_async
from django_ai_sdk.common import ChatMessage
from django_ai_sdk.conversation.models import Message, Thread
from django_ai_sdk.permissions import PermissionDenied
from django_ai_sdk.tracing import bind
from django_ai_sdk.tracing.schemas import TokenUsage, TraceOut
from django_ai_sdk.tracing.services import TraceService
from django_ai_sdk.tracing.tracer import DefaultTracer
from haystack.dataclasses import ChatMessage as HaystackChatMessage
from tests.mocks.permissions import thread_permissions
from tests.mocks.storage import setup_thread_adapter

ALLOW_ALL = "django_ai_sdk.permissions.AllowAll"
DENY_ALL = "django_ai_sdk.permissions.DenyAll"


def _thread_with_messages():
    thread = Thread.objects.create()
    chat_message = ChatMessage(role="assistant", content="hi", id=str(uuid.uuid4()))
    first = Message.objects.create(thread=thread, result=chat_message.model_dump())
    second = Message.objects.create(thread=thread, result=chat_message.model_dump())
    return thread, first, second


def _agent_run(thread, message, usage):
    """Trace one agent run: a rollup span over one step per usage pair."""
    tracer = DefaultTracer()
    with bind(thread_id=thread.id, message_id=message.id):
        with tracer.trace("haystack.agent.run") as agent_span:
            for prompt, completion in usage:
                with tracer.trace("haystack.agent.step", parent_span=agent_span) as step:
                    with tracer.trace("haystack.agent.step.llm", parent_span=step) as llm:
                        reply = HaystackChatMessage.from_assistant(
                            "hi",
                            meta={
                                "model": "gpt-x",
                                "usage": {
                                    "prompt_tokens": prompt,
                                    "completion_tokens": completion,
                                    "total_tokens": prompt + completion,
                                },
                            },
                        )
                        llm.set_content_tag("haystack.agent.step.llm.output", {"replies": [reply]})
            agent_span.set_content_tag(
                "haystack.agent.output",
                {
                    "token_usage": {
                        "prompt_tokens": sum(u[0] for u in usage),
                        "completion_tokens": sum(u[1] for u in usage),
                        "total_tokens": sum(u[0] + u[1] for u in usage),
                    }
                },
            )


@pytest.fixture
def traced_thread(mock_storage_adapter_registry, mock_user):
    """A thread with two traced runs, resolvable by ThreadService."""
    with thread_permissions(ALLOW_ALL):
        setup_thread_adapter(mock_storage_adapter_registry, user_id=mock_user.pk)
        thread, first, second = _thread_with_messages()
        _agent_run(thread, first, [(10, 5), (20, 8)])
        _agent_run(thread, second, [(7, 3)])
        yield thread, first, second


@pytest.mark.django_db(transaction=True)
class TestThreadTraces:
    async def test_returns_every_span_newest_first(self, traced_thread, mock_user):
        thread, _, _ = traced_thread

        traces = await TraceService.thread_traces(thread.id, user=mock_user)

        # One rollup + 2 spans per step, per run: (1 + 2*2) + (1 + 2*1)
        assert len(traces) == 8
        assert all(isinstance(t, TraceOut) for t in traces)
        assert [t.started_at for t in traces] == sorted(
            (t.started_at for t in traces), reverse=True
        )

    async def test_paginates(self, traced_thread, mock_user):
        thread, _, _ = traced_thread

        page = await TraceService.thread_traces(thread.id, user=mock_user, limit=4, offset=2)
        everything = await TraceService.thread_traces(thread.id, user=mock_user)

        assert [t.id for t in page] == [t.id for t in everything[2:6]]

    async def test_message_id_narrows_to_one_run(self, traced_thread, mock_user):
        thread, _, second = traced_thread

        traces = await TraceService.thread_traces(thread.id, user=mock_user, message_id=second.id)

        assert len(traces) == 3
        assert {t.message_id for t in traces} == {second.id}

    async def test_operation_name_narrows_to_one_operation(self, traced_thread, mock_user):
        thread, _, _ = traced_thread

        traces = await TraceService.thread_traces(
            thread.id, user=mock_user, operation_name="haystack.agent.run"
        )

        assert len(traces) == 2
        assert {t.operation_name for t in traces} == {"haystack.agent.run"}

    async def test_unknown_thread_raises_value_error(
        self, mock_storage_adapter_registry, mock_user
    ):
        # No adapter resolves the id, which is how a missing thread looks.
        mock_storage_adapter_registry.get_all_adapters.return_value = []

        with thread_permissions(ALLOW_ALL), pytest.raises(ValueError, match="Thread not found"):
            await TraceService.thread_traces(uuid.uuid4(), user=mock_user)


@pytest.mark.django_db(transaction=True)
class TestMessageTraces:
    async def test_resolves_the_thread_from_the_message(self, traced_thread, mock_user):
        _, first, _ = traced_thread

        traces = await TraceService.message_traces(first.id, user=mock_user)

        assert len(traces) == 5
        assert {t.message_id for t in traces} == {first.id}

    async def test_matches_the_thread_scoped_filter(self, traced_thread, mock_user):
        thread, first, _ = traced_thread

        by_message = await TraceService.message_traces(first.id, user=mock_user)
        by_filter = await TraceService.thread_traces(thread.id, user=mock_user, message_id=first.id)

        assert [t.id for t in by_message] == [t.id for t in by_filter]

    async def test_unknown_message_raises_value_error(self, traced_thread, mock_user):
        with pytest.raises(ValueError, match="Message not found"):
            await TraceService.message_traces(uuid.uuid4(), user=mock_user)


@pytest.mark.django_db(transaction=True)
class TestTokenUsage:
    async def test_thread_total_does_not_double_count_the_rollup(self, traced_thread, mock_user):
        thread, _, _ = traced_thread

        usage = await TraceService.thread_token_usage(thread.id, user=mock_user)

        assert usage == TokenUsage(prompt_tokens=37, completion_tokens=16, total_tokens=53)

    async def test_message_total_isolates_one_run(self, traced_thread, mock_user):
        _, _, second = traced_thread

        usage = await TraceService.message_token_usage(second.id, user=mock_user)

        assert usage == TokenUsage(prompt_tokens=7, completion_tokens=3, total_tokens=10)


@pytest.mark.django_db(transaction=True)
class TestPermissions:
    async def test_every_method_is_gated_by_the_thread(self, traced_thread, mock_user):
        thread, first, _ = traced_thread

        with thread_permissions(DENY_ALL):
            with pytest.raises(PermissionDenied):
                await TraceService.thread_traces(thread.id, user=mock_user)
            with pytest.raises(PermissionDenied):
                await TraceService.message_traces(first.id, user=mock_user)
            with pytest.raises(PermissionDenied):
                await TraceService.thread_token_usage(thread.id, user=mock_user)
            with pytest.raises(PermissionDenied):
                await TraceService.message_token_usage(first.id, user=mock_user)


def _subagent_run(thread, message, agent_id, name, usage):
    """Trace a coordinator delegating one sub-task to a named subagent."""
    from django_ai_sdk.agents.subagent import SUBAGENT_ID_TAG, SUBAGENT_NAME_TAG

    tracer = DefaultTracer()
    with bind(thread_id=thread.id, message_id=message.id):
        with tracer.trace("haystack.agent.run") as coordinator:
            with tracer.trace(
                "django_ai_sdk.subagent.run",
                tags={SUBAGENT_NAME_TAG: name, SUBAGENT_ID_TAG: agent_id},
                parent_span=coordinator,
            ) as wrapper:
                with tracer.trace("haystack.agent.run", parent_span=wrapper) as inner:
                    for prompt, completion in usage:
                        with tracer.trace("haystack.agent.step.llm", parent_span=inner) as llm:
                            reply = HaystackChatMessage.from_assistant(
                                "report",
                                meta={
                                    "model": "gpt-x",
                                    "usage": {
                                        "prompt_tokens": prompt,
                                        "completion_tokens": completion,
                                        "total_tokens": prompt + completion,
                                    },
                                },
                            )
                            llm.set_content_tag(
                                "haystack.agent.step.llm.output", {"replies": [reply]}
                            )


PLANNER_ID = "11111111-1111-1111-1111-111111111111"
CHECKER_ID = "22222222-2222-2222-2222-222222222222"


@pytest.mark.django_db(transaction=True)
class TestSubagentUsage:
    """thread_token_usage() carries the per-subagent breakdown — one call for
    both the total and the crew's cost, no separate round trip."""

    async def test_breakdown_keys_by_id_not_name(self, mock_user):
        """You never have to know a subagent's id — it comes back as the key.
        Keyed by id, not name: two different classes sharing a display name
        must not merge their totals (agent_name is carried for display only).
        """
        thread, first, second = await sync_to_async(_thread_with_messages)()
        await sync_to_async(_subagent_run)(
            thread, first, PLANNER_ID, "Research Planner", [(10, 5), (20, 7)]
        )
        await sync_to_async(_subagent_run)(thread, second, CHECKER_ID, "Fact Checker", [(3, 1)])

        with thread_permissions(ALLOW_ALL):
            usage = await TraceService.thread_token_usage(thread.id, user=mock_user)

        # The outer total still covers every run in the thread…
        assert usage.total_tokens == 46
        # …and each subagent's own cost is broken out under its id.
        assert usage.by_subagent == {
            PLANNER_ID: TokenUsage(
                agent_name="Research Planner",
                prompt_tokens=30,
                completion_tokens=12,
                total_tokens=42,
            ),
            CHECKER_ID: TokenUsage(
                agent_name="Fact Checker", prompt_tokens=3, completion_tokens=1, total_tokens=4
            ),
        }

    async def test_empty_without_subagents(self, traced_thread, mock_user):
        thread, _, _ = traced_thread

        with thread_permissions(ALLOW_ALL):
            usage = await TraceService.thread_token_usage(thread.id, user=mock_user)

        assert usage.by_subagent == {}

    async def test_message_scope_is_also_broken_down(self, mock_user):
        thread, first, _ = await sync_to_async(_thread_with_messages)()
        await sync_to_async(_subagent_run)(thread, first, PLANNER_ID, "Research Planner", [(10, 5)])

        with thread_permissions(ALLOW_ALL):
            usage = await TraceService.message_token_usage(first.id, user=mock_user)

        assert usage.by_subagent == {
            PLANNER_ID: TokenUsage(
                agent_name="Research Planner", prompt_tokens=10, completion_tokens=5, total_tokens=15
            )
        }
