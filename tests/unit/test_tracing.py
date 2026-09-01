from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace

import pytest
from asgiref.sync import sync_to_async
from django_ai_sdk.adapters.base import Stream
from django_ai_sdk.common import ChatMessage
from django_ai_sdk.conversation.models import Message, Thread
from django_ai_sdk.tracing import aflush, bind, current
from django_ai_sdk.tracing.models import Trace
from django_ai_sdk.tracing.tracer import DefaultTracer
from haystack import Pipeline
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.core.component import component
from haystack.dataclasses import ChatMessage as HaystackChatMessage
from haystack.tracing.tracer import tracer as haystack_tracer
from haystack.utils import Secret


@component
class StubGenerator:
    """Minimal component mimicking a generator's output shape — no LLM calls."""

    def __init__(self, model="stub-model"):
        self.model = model
        self.streaming_callback = None

    @component.output_types(replies=list[HaystackChatMessage], meta=list[dict])
    def run(self, messages: list[HaystackChatMessage]):
        reply = HaystackChatMessage.from_assistant("stub reply")
        meta = [
            {
                "model": self.model,
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
        ]
        return {"replies": [reply], "meta": meta}


def _build_pipeline():
    pipeline = Pipeline()
    pipeline.add_component("stub", StubGenerator())
    return pipeline


def _make_thread_and_message():
    thread = Thread.objects.create()
    chat_message = ChatMessage(role="assistant", content="hi", id=str(uuid.uuid4()))
    message = Message.objects.create(thread=thread, result=chat_message.model_dump())
    return thread, message


async def _amake_thread_and_message():
    return await sync_to_async(_make_thread_and_message)()


@pytest.fixture
def orm_tracer():
    """Enable the DB tracer, restoring Haystack's previous tracer afterwards."""
    from haystack import tracing

    previous = tracing.tracer.actual_tracer
    tracer = DefaultTracer()
    tracing.enable_tracing(tracer)
    yield tracer
    tracing.enable_tracing(previous)


class TestSyncPipeline:
    @pytest.mark.django_db
    def test_creates_span_rows_with_hierarchy(self, orm_tracer):
        result = _build_pipeline().run({"messages": [HaystackChatMessage.from_user("hi")]})

        assert result["stub"]["replies"]
        root = Trace.objects.get(operation_name="haystack.pipeline.run")
        stub = Trace.objects.get(operation_name="haystack.component.run")
        assert root.parent_id is None
        assert stub.parent_id == root.id
        for row in (root, stub):
            assert row.ended_at is not None
            assert row.duration_ms is not None
            assert row.duration_ms >= 0
        assert stub.tags["haystack.component.name"] == "stub"

    @pytest.mark.django_db
    def test_whole_tree_is_written_in_one_query(self, orm_tracer, django_assert_num_queries):
        tracer = DefaultTracer()
        with django_assert_num_queries(1):
            with tracer.trace("root") as root:
                with tracer.trace("child", parent_span=root) as child:
                    with tracer.trace("grandchild", parent_span=child):
                        pass

        assert Trace.objects.count() == 3

    @pytest.mark.django_db
    def test_nested_spans_use_current_span_as_parent(self, orm_tracer):
        # Haystack's Agent parents itself on tracer.current_span() rather than
        # passing parent_span explicitly.
        tracer = DefaultTracer()
        with tracer.trace("root"):
            with tracer.trace("implicit-child"):
                pass

        root = Trace.objects.get(operation_name="root")
        child = Trace.objects.get(operation_name="implicit-child")
        assert child.parent_id == root.id


class TestAsyncPipeline:
    # Writes are scheduled on the loop, escaping pytest-django's per-test
    # transaction wrap, so these tests need real transactions.
    @pytest.mark.django_db(transaction=True)
    async def test_run_async_persists_without_blocking(self, orm_tracer):
        result = await _build_pipeline().run_async(
            {"messages": [HaystackChatMessage.from_user("hi")]}
        )
        await orm_tracer.aflush()

        assert result["stub"]["replies"]
        root = await Trace.objects.aget(operation_name="haystack.pipeline.run")
        stub = await Trace.objects.aget(operation_name="haystack.component.run")
        assert stub.parent_id == root.id
        assert root.duration_ms is not None
        assert stub.duration_ms is not None

    @pytest.mark.django_db(transaction=True)
    async def test_worker_thread_child_inherits_from_parent_span(self, orm_tracer):
        thread, message = await _amake_thread_and_message()
        tracer = DefaultTracer()

        def make_child():
            with tracer.trace("worker-child", parent_span=root):
                pass

        with bind(thread_id=thread.id, message_id=message.id):
            with tracer.trace("root") as root:
                await asyncio.to_thread(make_child)
        await aflush()

        child = await Trace.objects.aget(operation_name="worker-child")
        root_row = await Trace.objects.aget(operation_name="root")
        assert child.parent_id == root_row.id
        # Inherited from the parent span object, not from the contextvar.
        assert child.thread_id == thread.id
        assert child.message_id == message.id


class TestCorrelation:
    @pytest.mark.django_db
    def test_bind_stamps_all_spans(self, orm_tracer):
        thread, message = _make_thread_and_message()
        tracer = DefaultTracer()
        with bind(thread_id=str(thread.id), message_id=str(message.id)):
            with tracer.trace("root") as root:
                with tracer.trace("child", parent_span=root):
                    pass

        root_row = Trace.objects.get(operation_name="root")
        child = Trace.objects.get(operation_name="child")
        for row in (root_row, child):
            assert row.thread_id == thread.id
            assert row.message_id == message.id
        assert child.parent_id == root_row.id

    @pytest.mark.django_db
    def test_unbound_spans_have_null_correlation(self, orm_tracer):
        tracer = DefaultTracer()
        with tracer.trace("root"):
            pass

        row = Trace.objects.get(operation_name="root")
        assert row.thread_id is None
        assert row.message_id is None

    @pytest.mark.django_db
    def test_bind_is_restored_on_exit(self, orm_tracer):
        thread, message = _make_thread_and_message()
        tracer = DefaultTracer()
        with bind(thread_id=str(thread.id), message_id=str(message.id)):
            assert current() is not None
        assert current() is None

        with tracer.trace("after"):
            pass
        assert Trace.objects.get(operation_name="after").thread_id is None

    @pytest.mark.django_db
    def test_invalid_ids_are_skipped(self, orm_tracer):
        tracer = DefaultTracer()
        with bind(thread_id="not-a-uuid", message_id=None):
            with tracer.trace("root"):
                pass

        row = Trace.objects.get(operation_name="root")
        assert row.thread_id is None
        assert row.message_id is None

    @pytest.mark.django_db(transaction=True)
    async def test_bind_propagates_into_async_pipeline(self, orm_tracer):
        thread, message = await _amake_thread_and_message()
        with bind(thread_id=str(thread.id), message_id=str(message.id)):
            await _build_pipeline().run_async({"messages": [HaystackChatMessage.from_user("hi")]})
        await aflush()

        spans = [span async for span in Trace.objects.filter(message_id=message.id)]
        assert {span.operation_name for span in spans} == {
            "haystack.pipeline.run",
            "haystack.component.run",
        }
        assert all(span.thread_id == thread.id for span in spans)

    @pytest.mark.django_db
    def test_thread_scoped_queries(self, orm_tracer):
        thread_a, message_a = _make_thread_and_message()
        thread_b, message_b = _make_thread_and_message()
        tracer = DefaultTracer()
        for thread, message in [(thread_a, message_a), (thread_b, message_b)]:
            with bind(thread_id=thread.id, message_id=message.id):
                with tracer.trace("root") as root:
                    with tracer.trace("child", parent_span=root):
                        pass

        assert Trace.objects.filter(thread_id=thread_a.id).count() == 2
        root_of_a = Trace.objects.filter(message_id=message_a.id, parent__isnull=True).get()
        assert root_of_a.operation_name == "root"
        assert Trace.objects.filter(thread_id=thread_b.id, message_id=message_b.id).count() == 2

    @pytest.mark.django_db
    def test_thread_deletion_cascades_to_traces(self, orm_tracer):
        thread, message = _make_thread_and_message()
        tracer = DefaultTracer()
        with bind(thread_id=thread.id, message_id=message.id):
            with tracer.trace("root"):
                pass

        thread.delete()
        assert not Trace.objects.filter(operation_name="root").exists()


class TestStreamAdapter:
    """The streaming adapter's tracing touchpoints."""

    def test_openai_generator_gets_include_usage(self):
        # OpenAI sends no usage chunk on a streamed response unless asked, which
        # would leave every token column null.
        generator = OpenAIChatGenerator(api_key=Secret.from_token("test"))
        pipeline = Pipeline()
        pipeline.add_component("gen", generator)
        Stream(pipeline=pipeline, generator=generator, store=False)

        assert generator.generation_kwargs["stream_options"] == {"include_usage": True}

    def test_non_openai_generator_is_left_alone(self):
        # stream_options is OpenAI-specific; other generators would reject it.
        pipeline = _build_pipeline()
        generator = pipeline.get_component("stub")
        Stream(pipeline=pipeline, generator=generator, store=False)

        assert not hasattr(generator, "generation_kwargs")

    @pytest.mark.django_db(transaction=True)
    async def test_stream_stamps_spans_with_its_thread_and_message(self, orm_tracer):
        thread, _ = await _amake_thread_and_message()
        pipeline = _build_pipeline()
        stream = Stream(
            pipeline=pipeline,
            generator=pipeline.get_component("stub"),
            store=False,
            storage_adapter=SimpleNamespace(thread_id=str(thread.id)),
        )

        message_id = None
        async for event in stream.stream([ChatMessage(role="user", content="hi")]):
            message_id = getattr(event, "message_id", None) or message_id
        await aflush()

        assert message_id is not None
        spans = [span async for span in Trace.objects.all()]
        assert spans
        assert all(str(span.message_id) == message_id for span in spans)
        assert all(span.thread_id == thread.id for span in spans)


class TestUsageHarvesting:
    """Token usage must land whether or not content tracing stores the payload."""

    @staticmethod
    def _llm_output(model="gpt-x", prompt=11, completion=7, total=18):
        reply = HaystackChatMessage.from_assistant(
            "hi",
            meta={
                "model": model,
                "usage": {
                    "prompt_tokens": prompt,
                    "completion_tokens": completion,
                    "total_tokens": total,
                },
            },
        )
        return {"replies": [reply]}

    @pytest.mark.django_db
    def test_agent_step_llm_usage_without_content_tracing(self, orm_tracer):
        tracer = DefaultTracer()
        with tracer.trace("haystack.agent.step.llm") as span:
            span.set_content_tag("haystack.agent.step.llm.output", self._llm_output())

        row = Trace.objects.get(operation_name="haystack.agent.step.llm")
        assert row.model_name == "gpt-x"
        assert (row.prompt_tokens, row.completion_tokens, row.total_tokens) == (11, 7, 18)
        # The payload itself stays out of the DB while content tracing is off.
        assert row.tags == {}

    @pytest.mark.django_db
    def test_agent_step_llm_payload_stored_with_content_tracing(self, orm_tracer, monkeypatch):
        monkeypatch.setattr(haystack_tracer, "is_content_tracing_enabled", True)
        tracer = DefaultTracer()
        with tracer.trace("haystack.agent.step.llm") as span:
            span.set_content_tag("haystack.agent.step.llm.output", self._llm_output())

        row = Trace.objects.get(operation_name="haystack.agent.step.llm")
        assert row.total_tokens == 18
        assert "haystack.agent.step.llm.output" in row.tags

    @pytest.mark.django_db
    def test_agent_run_total_from_token_usage(self, orm_tracer):
        tracer = DefaultTracer()
        with tracer.trace("haystack.agent.run") as span:
            span.set_content_tag(
                "haystack.agent.output",
                {
                    "token_usage": {
                        "prompt_tokens": 30,
                        "completion_tokens": 12,
                        "total_tokens": 42,
                    },
                    "last_message": HaystackChatMessage.from_assistant(
                        "done", meta={"model": "gpt-x"}
                    ),
                },
            )

        row = Trace.objects.get(operation_name="haystack.agent.run")
        assert row.model_name == "gpt-x"
        assert row.total_tokens == 42

    @pytest.mark.django_db
    def test_component_output_usage_without_content_tracing(self, orm_tracer):
        _build_pipeline().run({"messages": [HaystackChatMessage.from_user("hi")]})

        stub = Trace.objects.get(operation_name="haystack.component.run")
        assert stub.model_name == "stub-model"
        assert (stub.prompt_tokens, stub.completion_tokens, stub.total_tokens) == (10, 5, 15)
        # The payload is gated even though the counts are not.
        assert "haystack.component.output" not in stub.tags

    @pytest.mark.django_db
    def test_component_output_payload_stored_with_content_tracing(self, orm_tracer, monkeypatch):
        monkeypatch.setattr(haystack_tracer, "is_content_tracing_enabled", True)
        _build_pipeline().run({"messages": [HaystackChatMessage.from_user("hi")]})

        stub = Trace.objects.get(operation_name="haystack.component.run")
        assert stub.total_tokens == 15
        assert "haystack.component.output" in stub.tags

    @pytest.mark.django_db
    def test_usage_extraction_tolerates_missing_meta(self, orm_tracer):
        tracer = DefaultTracer()
        with tracer.trace("span") as span:
            span.set_content_tag("haystack.component.output", {"replies": []})

        row = Trace.objects.get(operation_name="span")
        assert row.model_name == ""
        assert row.total_tokens is None

    @pytest.mark.django_db
    def test_usage_extraction_tolerates_dict_meta(self, orm_tracer):
        tracer = DefaultTracer()
        with tracer.trace("span") as span:
            span.set_content_tag(
                "haystack.component.output",
                {
                    "meta": {
                        "model": "m",
                        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
                    }
                },
            )

        row = Trace.objects.get(operation_name="span")
        assert row.model_name == "m"
        assert row.total_tokens == 3

    @pytest.mark.django_db
    def test_usage_extraction_tolerates_non_dict_payload(self, orm_tracer):
        tracer = DefaultTracer()
        with tracer.trace("span") as span:
            span.set_content_tag("haystack.agent.step.tool.output", "plain string")

        row = Trace.objects.get(operation_name="span")
        assert row.total_tokens is None


class TestTokenUsage:
    """Totals must not double-count the agent's own rollup span."""

    @staticmethod
    def _agent_run(thread, message, usage):
        """Trace a realistic agent run: a rollup span over one step per usage."""
        tracer = DefaultTracer()
        with bind(thread_id=thread.id, message_id=message.id):
            with tracer.trace("haystack.agent.run") as agent_span:
                total = {
                    "prompt_tokens": sum(u[0] for u in usage),
                    "completion_tokens": sum(u[1] for u in usage),
                    "total_tokens": sum(u[0] + u[1] for u in usage),
                }
                for prompt, completion in usage:
                    with tracer.trace("haystack.agent.step", parent_span=agent_span) as step:
                        with tracer.trace("haystack.agent.step.llm", parent_span=step) as llm:
                            llm.set_content_tag(
                                "haystack.agent.step.llm.output",
                                TestUsageHarvesting._llm_output(
                                    prompt=prompt, completion=completion, total=prompt + completion
                                ),
                            )
                # The Agent aggregates its steps onto its own span.
                agent_span.set_content_tag("haystack.agent.output", {"token_usage": total})

    @pytest.mark.django_db
    def test_thread_total_ignores_the_rollup(self, orm_tracer):
        thread, message_a = _make_thread_and_message()
        message_b = Message.objects.create(thread=thread, result={})
        self._agent_run(thread, message_a, [(10, 5), (20, 8)])
        self._agent_run(thread, message_b, [(7, 3)])

        # The rollups hold the same numbers, so a naive Sum would double these.
        assert Trace.objects.for_thread(thread.id).token_usage() == {
            "prompt_tokens": 37,
            "completion_tokens": 16,
            "total_tokens": 53,
        }

    @pytest.mark.django_db
    def test_message_total_isolates_one_run(self, orm_tracer):
        thread, message_a = _make_thread_and_message()
        message_b = Message.objects.create(thread=thread, result={})
        self._agent_run(thread, message_a, [(10, 5), (20, 8)])
        self._agent_run(thread, message_b, [(7, 3)])

        assert Trace.objects.for_message(message_b.id).token_usage() == {
            "prompt_tokens": 7,
            "completion_tokens": 3,
            "total_tokens": 10,
        }

    @pytest.mark.django_db
    def test_rollup_row_still_carries_the_run_total(self, orm_tracer):
        thread, message = _make_thread_and_message()
        self._agent_run(thread, message, [(10, 5), (20, 8)])

        rollup = Trace.objects.get(operation_name="haystack.agent.run")
        assert rollup.total_tokens == 43
        assert Trace.objects.for_message(message.id).token_usage()["total_tokens"] == 43

    @pytest.mark.django_db
    def test_no_traces_totals_to_zero(self, orm_tracer):
        thread, _ = _make_thread_and_message()

        assert Trace.objects.for_thread(thread.id).token_usage() == {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    @pytest.mark.django_db(transaction=True)
    async def test_async_total_matches_the_sync_one(self, orm_tracer):
        thread, message = await _amake_thread_and_message()
        await sync_to_async(self._agent_run)(thread, message, [(10, 5), (20, 8)])
        await aflush()

        assert await Trace.objects.for_thread(thread.id).atoken_usage() == {
            "prompt_tokens": 30,
            "completion_tokens": 13,
            "total_tokens": 43,
        }


class TestTagHandling:
    @pytest.mark.django_db
    def test_excluded_tags_are_dropped(self, orm_tracer, settings):
        settings.AI_SDK_TRACING_EXCLUDED_TAGS = ["haystack.agent.tools"]
        tracer = DefaultTracer()
        with tracer.trace("haystack.agent.run") as span:
            span.set_tags({"haystack.agent.tools": ["huge"], "haystack.agent.max_steps": 100})

        row = Trace.objects.get(operation_name="haystack.agent.run")
        assert "haystack.agent.tools" not in row.tags
        assert row.tags["haystack.agent.max_steps"] == 100

    @pytest.mark.django_db
    def test_excluding_a_content_tag_keeps_its_token_usage(
        self, orm_tracer, settings, monkeypatch
    ):
        # Usage is harvested before the exclusion guard, so the counts survive
        # even when the payload carrying them is dropped.
        monkeypatch.setattr(haystack_tracer, "is_content_tracing_enabled", True)
        settings.AI_SDK_TRACING_EXCLUDED_TAGS = ["haystack.agent.step.llm.output"]
        tracer = DefaultTracer()
        with tracer.trace("haystack.agent.step.llm") as span:
            span.set_content_tag(
                "haystack.agent.step.llm.output", TestUsageHarvesting._llm_output()
            )

        row = Trace.objects.get(operation_name="haystack.agent.step.llm")
        assert row.tags == {}
        assert (row.model_name, row.total_tokens) == ("gpt-x", 18)

    @pytest.mark.django_db
    def test_coerced_tags_are_stored(self, orm_tracer):
        tracer = DefaultTracer()
        with tracer.trace("span") as span:
            span.set_tag("plain", {"nested": [1, 2]})

        row = Trace.objects.get(operation_name="span")
        assert "nested" in row.tags["plain"]
