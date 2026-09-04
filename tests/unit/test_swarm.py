from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from django_ai_sdk import Agent
from django_ai_sdk.adapters.base import Stream
from django_ai_sdk.agents import (
    LogToolCallsHook,
    ToolAgent,
    ToolAgentConfig,
    ToolCallBudgetHook,
)
from django_ai_sdk.agents.subagent import (
    FINAL_PROMPT,
    SUBAGENT_META_KEY,
    SubagentStreamFilter,
    _tool_chunks_only,
    build_subagent,
    subagent_response,
    subagent_tool_name,
)
from django_ai_sdk.agents.tool_agent import SKIPPED_META_KEY
from django_ai_sdk.common import ChatMessage as SDKChatMessage
from django_ai_sdk.common import MessageChunk, StreamWriter
from django_ai_sdk.permissions import DenyAll, PermissionDenied
from django_ai_sdk.protocols.vercel import VercelProtocolHandler
from django_ai_sdk.storage.memory import MemoryStorageAdapter
from haystack.components.generators.chat import MockChatGenerator
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.dataclasses import StreamingChunk as HaystackStreamingChunk
from haystack.tools import Tool

MOCK_MODEL = "gpt-4o-mini"


def _weather_tool() -> Tool:
    def get_weather(city: str) -> str:
        return f"{city}: sunny"

    return Tool(
        name="get_weather",
        description="Get the weather for a city",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
        function=get_weather,
    )


def _tool_call_msg(tool_name: str, arguments: dict, call_id: str = "call_1") -> ChatMessage:
    return ChatMessage.from_assistant(
        tool_calls=[ToolCall(tool_name=tool_name, arguments=arguments, id=call_id)]
    )


def _completed_task(result: dict) -> asyncio.Task[Any]:
    """Return an already-completed asyncio Task yielding ``result``."""

    async def _done() -> dict:
        return result

    return asyncio.ensure_future(_done())


class SubAgent(Agent):
    name = "Sub Agent"
    description = "A generic subagent for tests."
    model = MOCK_MODEL
    protocol = VercelProtocolHandler
    storage_adapter = MemoryStorageAdapter

    async def get_pipeline_adapter(self, thread_id=None, user=None):
        pass


class DeniedSubAgent(SubAgent):
    name = "Denied Sub Agent"
    permissions = [DenyAll]


class CoordinatorAgent(Agent):
    name = "Coordinator Agent"
    description = "Coordinates a subagent crew."
    model = MOCK_MODEL
    protocol = VercelProtocolHandler
    storage_adapter = MemoryStorageAdapter
    agents = []

    async def get_pipeline_adapter(self, thread_id=None, user=None):
        pass


CoordinatorAgent.agents = [
    SubAgent,
    DeniedSubAgent,
    SubAgent,
    CoordinatorAgent,
]


class TestToolAgentBuild:
    def test_pipeline_builds_with_new_config_fields(self):
        gen = MockChatGenerator(responses=["hi"])
        config = ToolAgentConfig(
            model=MOCK_MODEL,
            system_prompt="be concise",
            tools=[],
            max_agent_steps=8,
            max_tool_calls=2,
            tool_concurrency_limit=3,
        )
        agent = ToolAgent(config=config, generator=gen)
        pipeline = agent.pipeline()
        haystack_agent = pipeline.get_component("agent")
        assert haystack_agent.max_agent_steps == 8
        assert haystack_agent.tool_concurrency_limit == 3
        assert haystack_agent.hooks.get("before_tool")

    def test_stream_subagent_tools_defaults_to_true(self):
        agent = ToolAgent.build_agent(MockChatGenerator(responses=["hi"]), [], "be concise")
        assert agent.tool_streaming_callback_passthrough is True

    def test_stream_subagent_tools_config_is_threaded(self):
        config = ToolAgentConfig(
            model=MOCK_MODEL,
            system_prompt="s",
            tools=[],
            stream_subagent_tools=False,
        )
        agent = (
            ToolAgent(config=config, generator=MockChatGenerator(responses=["hi"]))
            .pipeline()
            .get_component("agent")
        )
        assert agent.tool_streaming_callback_passthrough is False

    def test_budget_hook_is_not_added_when_unset(self):
        gen = MockChatGenerator(responses=["hi"])
        config = ToolAgentConfig(model=MOCK_MODEL, system_prompt="s", tools=[])
        pipeline = ToolAgent(config=config, generator=gen).pipeline()
        assert not pipeline.get_component("agent").hooks.get("before_tool")

    def test_custom_hooks_are_merged_with_budget_hook(self):
        gen = MockChatGenerator(responses=["hi"])
        config = ToolAgentConfig(
            model=MOCK_MODEL,
            system_prompt="s",
            tools=[],
            max_tool_calls=3,
            hooks={"before_tool": [LogToolCallsHook()]},
        )
        pipeline = ToolAgent(config=config, generator=gen).pipeline()
        hooks = pipeline.get_component("agent").hooks["before_tool"]
        assert len(hooks) == 2
        assert any(isinstance(h, LogToolCallsHook) for h in hooks)
        assert any(type(h).__name__ == "ToolCallBudgetHook" for h in hooks)

    def test_build_agent_exposes_user_prompt_and_required_variables(self):
        agent = ToolAgent.build_agent(
            MockChatGenerator(responses=["ok"]),
            [],
            "system",
            user_prompt="Task: {{task}}",
            required_variables=["task"],
        )
        assert agent.user_prompt == "Task: {{task}}"
        assert agent.required_variables == ["task"]


class TestToolCallBudgetHook:
    def test_hard_cap_on_tool_calls(self):
        tool = _weather_tool()
        agent = ToolAgent.build_agent(
            MockChatGenerator(
                responses=[
                    _tool_call_msg("get_weather", {"city": "a"}, "c1"),
                    _tool_call_msg("get_weather", {"city": "b"}, "c2"),
                    ChatMessage.from_assistant("Done, sunny everywhere."),
                ]
            ),
            [tool],
            "be concise",
            max_agent_steps=10,
            hooks={"before_tool": [ToolCallBudgetHook(max_tool_calls=1)]},
        )
        result = agent.run(messages=[ChatMessage.from_user("weather?")])
        assert result["tool_call_counts"]["get_weather"] == 1

    def test_no_cap_when_under_budget(self):
        tool = _weather_tool()
        agent = ToolAgent.build_agent(
            MockChatGenerator(
                responses=[
                    _tool_call_msg("get_weather", {"city": "a"}, "c1"),
                    _tool_call_msg("get_weather", {"city": "b"}, "c2"),
                    ChatMessage.from_assistant("Done."),
                ]
            ),
            [tool],
            "be concise",
            max_agent_steps=10,
            hooks={"before_tool": [ToolCallBudgetHook(max_tool_calls=5)]},
        )
        result = agent.run(messages=[ChatMessage.from_user("weather?")])
        assert result["tool_call_counts"]["get_weather"] == 2

    def test_only_runs_under_before_tool(self):
        assert ToolCallBudgetHook(1).allowed_hook_points == ["before_tool"]


class TestSubagentBuilders:
    def test_tool_name_from_display_name(self):
        assert subagent_tool_name(SubAgent) == "sub_agent"

    def test_tool_name_falls_back_to_class_name(self):
        class NoName(Agent):
            model = MOCK_MODEL

            async def get_pipeline_adapter(self, thread_id=None, user=None):
                pass

        assert subagent_tool_name(NoName) == "no_name"

    def test_report_handler(self):
        assert subagent_response([ChatMessage.from_assistant("Aye!")]) == "Aye!"
        assert "could not be completed" in subagent_response(None)

    def test_default_hooks_caps_tool_calls_when_configured(self):
        """The exact hook assembly build_subagent relies on — extracted so
        Agent.run(tools=True) can reuse it instead of a shadow set of limits."""
        from django_ai_sdk.agents.tool_agent import default_hooks

        class Capped(SubAgent):
            max_tool_calls = 2

        hooks = default_hooks(Capped())
        assert any(type(h).__name__ == "LogToolCallsHook" for h in hooks["before_tool"])
        budget = [h for h in hooks["before_tool"] if type(h).__name__ == "ToolCallBudgetHook"]
        assert len(budget) == 1
        assert budget[0].max_tool_calls == 2

    def test_default_hooks_uncapped_when_max_tool_calls_is_none(self):
        from django_ai_sdk.agents.tool_agent import default_hooks

        class Uncapped(SubAgent):
            max_tool_calls = None

        hooks = default_hooks(Uncapped())
        assert all(type(h).__name__ != "ToolCallBudgetHook" for h in hooks["before_tool"])

    @pytest.mark.asyncio
    async def test_build_subagent(self):
        with patch(
            "django_ai_sdk.agent.Agent.get_llm",
            return_value=MockChatGenerator(responses=["ok"]),
        ):
            built = await build_subagent(SubAgent, thread_id="t1", user=None)
        assert built is not None
        agent, agent_id = built
        assert agent.required_variables == ["task"]
        assert "{{task}}" in agent.user_prompt
        assert agent_id == SubAgent().agent_id

    @pytest.mark.asyncio
    async def test_build_subagent_defaults_budgets(self):
        with patch(
            "django_ai_sdk.agent.Agent.get_llm",
            return_value=MockChatGenerator(responses=["ok"]),
        ):
            agent, _ = await build_subagent(SubAgent, thread_id="t1", user=None)
        assert agent.max_agent_steps == 6
        budget = [h for h in agent.hooks["before_tool"] if type(h).__name__ == "ToolCallBudgetHook"]
        assert len(budget) == 1
        assert budget[0].max_tool_calls == 6

    @pytest.mark.asyncio
    async def test_build_subagent_honors_custom_budgets(self):
        class BudgetSubagent(SubAgent):
            max_agent_steps = 3
            max_tool_calls = 2

        with patch(
            "django_ai_sdk.agent.Agent.get_llm",
            return_value=MockChatGenerator(responses=["ok"]),
        ):
            agent, _ = await build_subagent(BudgetSubagent, thread_id="t1", user=None)
        assert agent.max_agent_steps == 3
        budget = [h for h in agent.hooks["before_tool"] if type(h).__name__ == "ToolCallBudgetHook"]
        assert len(budget) == 1
        assert budget[0].max_tool_calls == 2

    @pytest.mark.asyncio
    async def test_build_subagent_no_tool_cap_when_none(self):
        class UncappedSubagent(SubAgent):
            max_tool_calls = None

        with patch(
            "django_ai_sdk.agent.Agent.get_llm",
            return_value=MockChatGenerator(responses=["ok"]),
        ):
            agent, _ = await build_subagent(UncappedSubagent, thread_id="t1", user=None)
        assert agent.max_agent_steps == 6
        assert all(type(h).__name__ != "ToolCallBudgetHook" for h in agent.hooks["before_tool"])

    @pytest.mark.asyncio
    async def test_build_subagent_detects_cycles(self):
        class CycleAgentB(Agent):
            name = "Cycle B"
            description = "B"
            model = MOCK_MODEL
            protocol = VercelProtocolHandler
            storage_adapter = MemoryStorageAdapter
            agents = []  # filled after both classes exist

            async def get_pipeline_adapter(self, thread_id=None, user=None):
                pass

        class CycleAgentA(Agent):
            name = "Cycle A"
            description = "A"
            model = MOCK_MODEL
            protocol = VercelProtocolHandler
            storage_adapter = MemoryStorageAdapter
            agents = [CycleAgentB]

            async def get_pipeline_adapter(self, thread_id=None, user=None):
                pass

        CycleAgentB.agents = [CycleAgentA]

        with patch(
            "django_ai_sdk.agent.Agent.get_llm",
            return_value=MockChatGenerator(responses=["ok"]),
        ):
            built = await build_subagent(CycleAgentA, thread_id="t1", user=None)
        # The A -> B -> A back-edge must be skipped without infinite recursion.
        assert built is not None


class TestSubagentStreaming:
    def _chunks(self):
        from haystack.dataclasses import StreamingChunk, ToolCall, ToolCallDelta, ToolCallResult

        text = StreamingChunk(content="report text", index=0)
        tool_call = StreamingChunk(
            content="",
            index=0,
            tool_calls=[
                ToolCallDelta(index=0, tool_name="search_web", arguments='{"query": "Leukeleu"}')
            ],
        )
        tool_result = StreamingChunk(
            content="",
            index=0,
            tool_call_result=ToolCallResult(
                result="two results",
                origin=ToolCall(tool_name="search_web", arguments={"query": "Leukeleu"}, id="c1"),
                error=False,
            ),
        )
        return text, tool_call, tool_result

    @pytest.mark.asyncio
    async def test_tool_chunks_only_drops_text_forwards_tools(self):
        text, tool_call, tool_result = self._chunks()
        received = []

        async def sink(chunk):
            received.append(chunk)

        filtered = _tool_chunks_only(sink)
        await filtered(text)
        await filtered(tool_call)
        await filtered(tool_result)

        assert len(received) == 2
        assert received[0].tool_calls[0].tool_name == "search_web"
        assert received[1].tool_call_result.origin.tool_name == "search_web"

    def test_tool_chunks_only_returns_none_for_no_callback(self):
        assert _tool_chunks_only(None) is None

    @pytest.mark.asyncio
    async def test_stream_filter_forwards_task_and_filters_callback(self):
        inner = AsyncMock()
        inner.run_async = AsyncMock(
            return_value={"messages": [], "last_message": ChatMessage.from_assistant("report")}
        )
        wrapper = SubagentStreamFilter(inner)

        received = []

        async def sink(chunk):
            received.append(chunk)

        result = await wrapper.run_async(
            messages=[ChatMessage.from_user("hi")],
            task="Research Leukeleu",
            streaming_callback=sink,
        )
        assert result["last_message"].text == "report"
        kwargs = inner.run_async.call_args.kwargs
        assert kwargs["task"] == "Research Leukeleu"
        assert kwargs["streaming_callback"] is not None

        text, tool_call, _ = self._chunks()
        await kwargs["streaming_callback"](text)
        await kwargs["streaming_callback"](tool_call)
        assert len(received) == 1
        assert received[0].tool_calls[0].tool_name == "search_web"

    def test_component_tool_over_filter_exposes_streaming_callback_socket(self):
        from haystack.tools.component_tool import ComponentTool

        inner = ToolAgent.build_agent(
            MockChatGenerator(responses=["ok"]),
            [],
            "system",
            user_prompt="Task: {{task}}",
            required_variables=["task"],
        )
        tool = ComponentTool(
            component=SubagentStreamFilter(inner),
            name="research_planner",
            description="Researches a topic.",
            parameters={
                "type": "object",
                "properties": {"task": {"type": "string", "description": "task"}},
                "required": ["task"],
            },
            outputs_to_string={"source": "messages", "handler": subagent_response},
            inputs_from_state={"messages": "messages"},
        )
        sockets = tool._component.__haystack_input__._sockets_dict
        assert "streaming_callback" in sockets
        assert "task" in sockets
        assert "messages" in sockets

    @pytest.mark.asyncio
    async def test_tool_chunks_are_persisted_chronologically_by_get_events(self):
        from django_ai_sdk.adapters.base import _SENTINEL, Stream
        from django_ai_sdk.common import StreamWriter
        from haystack import Pipeline
        from haystack.dataclasses import StreamingChunk, ToolCall, ToolCallDelta, ToolCallResult

        # The delegation starts first, its subagent's tool completes first, and
        # the delegation output returns last. Stored order must follow start
        # order, not completion order.
        planner_start = StreamingChunk(
            content="",
            index=0,
            tool_calls=[
                ToolCallDelta(
                    index=0, tool_name="research_planner", arguments='{"task": "x"}', id="p1"
                )
            ],
        )
        sub_start = StreamingChunk(
            content="",
            index=0,
            tool_calls=[
                ToolCallDelta(
                    index=0, tool_name="search_web", arguments='{"query": "Leukeleu"}', id="s1"
                )
            ],
        )
        sub_result = StreamingChunk(
            content="",
            index=0,
            tool_call_result=ToolCallResult(
                result="two results",
                origin=ToolCall(tool_name="search_web", arguments={"query": "Leukeleu"}, id="s1"),
                error=False,
            ),
        )
        planner_result = StreamingChunk(
            content="",
            index=0,
            tool_call_result=ToolCallResult(
                result="report",
                origin=ToolCall(tool_name="research_planner", arguments={"task": "x"}, id="p1"),
                error=False,
            ),
        )

        queue: asyncio.Queue = asyncio.Queue()
        queue.put_nowait(planner_start)
        queue.put_nowait(sub_start)
        queue.put_nowait(sub_result)
        queue.put_nowait(planner_result)
        queue.put_nowait(_SENTINEL)

        stream = Stream(pipeline=Pipeline(), generator=AsyncMock())
        stream_writer = StreamWriter(message_id="m1", storage_callback=None)
        events = [e async for e in stream.get_events(queue, stream_writer)]

        event_types = [type(e).__name__ for e in events]
        assert event_types == [
            "ToolCallStartEvent",
            "ToolInputCompleteEvent",
            "ToolCallStartEvent",
            "ToolInputCompleteEvent",
            "ToolOutputEvent",
            "ToolOutputEvent",
        ]
        tools = stream_writer.message.tool_calls
        assert [t["name"] for t in tools] == ["research_planner", "search_web"]
        assert tools[0]["result"]["result"] == "report"
        assert tools[1]["arguments"] == {"query": "Leukeleu"}
        assert tools[1]["result"]["result"] == "two results"

        # Live persistence records the tool_call ids so get_pipeline_result
        # won't store the same tool activity a second time.
        assert "p1" in stream._persisted_tool_ids and "p1" in stream._persisted_tool_output_ids
        assert "s1" in stream._persisted_tool_ids and "s1" in stream._persisted_tool_output_ids

    @pytest.mark.asyncio
    async def test_get_pipeline_result_skips_already_persisted_tool_ids(self):
        from django_ai_sdk.adapters.base import _SENTINEL, Stream
        from django_ai_sdk.common import StreamWriter
        from haystack import Pipeline
        from haystack.dataclasses import StreamingChunk, ToolCall, ToolCallDelta, ToolCallResult

        tool_call = StreamingChunk(
            content="",
            index=0,
            tool_calls=[
                ToolCallDelta(
                    index=0, tool_name="search_web", arguments='{"query": "Leukeleu"}', id="s1"
                )
            ],
        )
        tool_result = StreamingChunk(
            content="",
            index=0,
            tool_call_result=ToolCallResult(
                result="two results",
                origin=ToolCall(tool_name="search_web", arguments={"query": "Leukeleu"}, id="s1"),
                error=False,
            ),
        )

        queue: asyncio.Queue = asyncio.Queue()
        queue.put_nowait(tool_call)
        queue.put_nowait(tool_result)
        queue.put_nowait(_SENTINEL)

        stream = Stream(pipeline=Pipeline(), generator=AsyncMock())
        stream_writer = StreamWriter(message_id="m1", storage_callback=None)
        async for _ in stream.get_events(queue, stream_writer):
            pass

        before = list(stream_writer.message.tool_calls)
        assert [t["name"] for t in before] == ["search_web"]

        # The generator returns the same tool activity in its final message;
        # the backstop must not double-store it.
        message = ChatMessage.from_dict(
            {
                "role": "assistant",
                "content": [
                    {
                        "tool_call": {
                            "tool_name": "search_web",
                            "arguments": {"query": "Leukeleu"},
                            "id": "s1",
                        }
                    },
                    {
                        "tool_call_result": {
                            "result": "two results",
                            "origin": {
                                "tool_name": "search_web",
                                "arguments": {"query": "Leukeleu"},
                                "id": "s1",
                            },
                            "error": False,
                        }
                    },
                ],
            }
        )
        await stream.get_pipeline_result(_completed_task({"messages": [message]}), stream_writer)
        assert stream_writer.message.tool_calls == before

    @pytest.mark.asyncio
    async def test_get_pipeline_result_fills_missing_output_for_live_started_call(self):
        from django_ai_sdk.adapters.base import _SENTINEL, Stream
        from django_ai_sdk.common import StreamWriter
        from haystack import Pipeline
        from haystack.dataclasses import StreamingChunk, ToolCallDelta

        # The start streams live but its output never does (e.g. a tool whose
        # result is not streamed through the callback).
        tool_call = StreamingChunk(
            content="",
            index=0,
            tool_calls=[
                ToolCallDelta(
                    index=0, tool_name="research_planner", arguments='{"task": "x"}', id="p1"
                )
            ],
        )

        queue: asyncio.Queue = asyncio.Queue()
        queue.put_nowait(tool_call)
        queue.put_nowait(_SENTINEL)

        stream = Stream(pipeline=Pipeline(), generator=AsyncMock())
        stream_writer = StreamWriter(message_id="m1", storage_callback=None)
        async for _ in stream.get_events(queue, stream_writer):
            pass

        message = ChatMessage.from_dict(
            {
                "role": "assistant",
                "content": [
                    {
                        "tool_call": {
                            "tool_name": "research_planner",
                            "arguments": {"task": "x"},
                            "id": "p1",
                        }
                    },
                    {
                        "tool_call_result": {
                            "result": "report",
                            "origin": {
                                "tool_name": "research_planner",
                                "arguments": {"task": "x"},
                                "id": "p1",
                            },
                            "error": False,
                        }
                    },
                ],
            }
        )
        await stream.get_pipeline_result(_completed_task({"messages": [message]}), stream_writer)

        tools = stream_writer.message.tool_calls
        assert [t["name"] for t in tools] == ["research_planner"]
        assert tools[0]["result"]["result"] == "report"
        assert len(tools) == 1

    @pytest.mark.asyncio
    async def test_get_pipeline_result_persists_ids_not_streamed_live(self):
        from django_ai_sdk.adapters.base import _SENTINEL, Stream
        from django_ai_sdk.common import StreamWriter
        from haystack import Pipeline

        queue: asyncio.Queue = asyncio.Queue()
        queue.put_nowait(_SENTINEL)

        stream = Stream(pipeline=Pipeline(), generator=AsyncMock())
        stream_writer = StreamWriter(message_id="m1", storage_callback=None)
        async for _ in stream.get_events(queue, stream_writer):
            pass

        message = ChatMessage.from_dict(
            {
                "role": "assistant",
                "content": [
                    {
                        "tool_call": {
                            "tool_name": "boat_expert",
                            "arguments": {"task": "x"},
                            "id": "c1",
                        }
                    },
                    {
                        "tool_call_result": {
                            "result": "keel lore",
                            "origin": {
                                "tool_name": "boat_expert",
                                "arguments": {"task": "x"},
                                "id": "c1",
                            },
                            "error": False,
                        }
                    },
                ],
            }
        )
        await stream.get_pipeline_result(_completed_task({"messages": [message]}), stream_writer)
        names = [t["name"] for t in stream_writer.message.tool_calls]
        assert names == ["boat_expert"]

    def test_sync_run_accepts_sync_streaming_callback(self):
        inner = ToolAgent.build_agent(
            MockChatGenerator(responses=["ok"]),
            [],
            "system",
            user_prompt="Task: {{task}}",
            required_variables=["task"],
        )
        wrapper = SubagentStreamFilter(inner)

        received = []

        def sync_sink(chunk):
            received.append(chunk)

        # Sync run() with a sync callback must not raise "The init callback
        # cannot be a coroutine" (async-only filter would have failed here).
        result = wrapper.run(
            messages=[ChatMessage.from_user("hi")],
            task="Research Leukeleu",
            streaming_callback=sync_sink,
        )
        assert result["last_message"].text == "ok"
        assert received == []


class TestCoordinatorGetAgentTools:
    @pytest.mark.asyncio
    async def test_builds_component_tools_with_unique_names(self):
        coordinator = CoordinatorAgent()
        # VIEW_AGENT is gated on an authenticated user (AgentDefaultPermission),
        # so an anonymous caller gets every subagent filtered out.
        user = MagicMock(is_authenticated=True)
        with patch(
            "django_ai_sdk.agent.Agent.get_llm",
            return_value=MockChatGenerator(responses=["ok"]),
        ):
            tools = await coordinator.get_agent_tools(thread_id="t1", user=user)

        names = [t.name for t in tools]
        assert len(names) == len(set(names))
        # DeniedSubAgent filtered out; self-reference + dupes skipped.
        assert "denied_sub_agent" not in names
        assert "sub_agent" in names
        assert "coordinator_agent" not in names
        assert len(tools) == 1

    @pytest.mark.asyncio
    async def test_permission_denied_subagent_skipped(self):
        coordinator = CoordinatorAgent()
        with patch(
            "django_ai_sdk.agent.Agent.get_llm",
            return_value=MockChatGenerator(responses=["ok"]),
        ):
            with patch(
                "django_ai_sdk.agents.services.AgentService.has_perms",
                new_callable=AsyncMock,
                side_effect=PermissionDenied,
            ):
                tools = await coordinator.get_agent_tools(thread_id="t1", user=None)
        assert tools == []

    def test_system_prompt_includes_roster(self):
        coordinator = CoordinatorAgent()
        prompt = coordinator.get_system_prompt()
        assert "Sub Agent" in prompt
        assert "Denied Sub Agent" in prompt

    @pytest.mark.asyncio
    async def test_name_collision_is_resolved_by_agent_id_not_a_counter(self):
        """Two different classes sharing a display name must not get the
        old `_2`/`_3` counter suffix — agent_id guarantees no repeat collision,
        a positional counter is just "the next free slot"."""

        class SubAgentTwin(SubAgent):
            pass

        class Coordinator(CoordinatorAgent):
            agents = [SubAgent, SubAgentTwin]

        coordinator = Coordinator()
        user = MagicMock(is_authenticated=True)
        with patch(
            "django_ai_sdk.agent.Agent.get_llm",
            return_value=MockChatGenerator(responses=["ok"]),
        ):
            tools = await coordinator.get_agent_tools(thread_id="t1", user=user)

        names = [t.name for t in tools]
        assert names[0] == "sub_agent"
        expected_suffix = SubAgentTwin().agent_id.replace("-", "")[:6]
        assert names[1] == f"sub_agent_{expected_suffix}"

    def test_system_prompt_without_agents(self):
        prompt = SubAgent().get_system_prompt()
        assert "Available subagents" not in prompt


class TestSwarmEndToEnd:
    @pytest.mark.asyncio
    async def test_coordinator_delegates_to_subagent(self):
        from haystack.tools.component_tool import ComponentTool

        subagent = ToolAgent.build_agent(
            MockChatGenerator(responses=[ChatMessage.from_assistant("Aye! Keel lore.")]),
            [],
            "You are the boat expert.",
            user_prompt="Task: {{task}}",
            required_variables=["task"],
        )
        boat_tool = ComponentTool(
            component=subagent,
            name="boat_expert",
            description="Pirate boat lore",
            parameters={
                "type": "object",
                "properties": {"task": {"type": "string", "description": "task"}},
                "required": ["task"],
            },
            outputs_to_string={"source": "messages", "handler": subagent_response},
            inputs_from_state={"messages": "messages"},
        )

        coordinator = ToolAgent.build_agent(
            MockChatGenerator(
                responses=[
                    _tool_call_msg("boat_expert", {"task": "Explain keels"}, "c1"),
                    ChatMessage.from_assistant("Yarr, expert says: Aye! Keel lore."),
                ]
            ),
            [boat_tool],
            "You are the captain. Delegate then summarize.",
        )
        result = await coordinator.run_async(messages=[ChatMessage.from_user("Keels?")])

        assert result["tool_call_counts"]["boat_expert"] == 1
        delegated = [m for m in result["messages"] if m.tool_call_results]
        assert delegated, "coordinator should have received the subagent output"
        # The report, plus the source list the handoff always appends.
        handed_back = delegated[0].tool_call_results[0].result
        assert handed_back.startswith("Aye! Keel lore.")
        assert "boat_expert(task='Explain keels')" in handed_back

    @pytest.mark.asyncio
    async def test_coordinator_streams_subagent_tool_calls(self):
        from haystack.tools.component_tool import ComponentTool

        def search_web(query: str) -> str:
            return f"results for {query}"

        search_tool = Tool(
            name="search_web",
            description="Search the web.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            function=search_web,
        )

        subagent = ToolAgent.build_agent(
            MockChatGenerator(
                responses=[
                    _tool_call_msg("search_web", {"query": "Leukeleu"}, "s1"),
                    ChatMessage.from_assistant("DISTINCT_SUBAGENT_TEXT"),
                ]
            ),
            [search_tool],
            "You are the researcher.",
            user_prompt="Task: {{task}}",
            required_variables=["task"],
        )
        planner_tool = ComponentTool(
            component=SubagentStreamFilter(subagent),
            name="research_planner",
            description="Researches a topic.",
            parameters={
                "type": "object",
                "properties": {"task": {"type": "string", "description": "task"}},
                "required": ["task"],
            },
            outputs_to_string={"source": "messages", "handler": subagent_response},
            inputs_from_state={"messages": "messages"},
        )
        coordinator = ToolAgent.build_agent(
            MockChatGenerator(
                responses=[
                    _tool_call_msg("research_planner", {"task": "Research Leukeleu"}, "c1"),
                    ChatMessage.from_assistant("Coordinator summary."),
                ]
            ),
            [planner_tool],
            "You are the coordinator.",
            stream_subagent_tools=True,
        )

        received: list = []

        async def sink(chunk):
            received.append(chunk)

        result = await coordinator.run_async(
            messages=[ChatMessage.from_user("Find info about Leukeleu")],
            streaming_callback=sink,
        )

        call_names = [
            tc.tool_name for chunk in received if chunk.tool_calls for tc in chunk.tool_calls
        ]
        assert "research_planner" in call_names, call_names
        assert "search_web" in call_names, call_names
        result_names = [
            chunk.tool_call_result.origin.tool_name for chunk in received if chunk.tool_call_result
        ]
        assert "search_web" in result_names, result_names
        streamed_text = "".join(c.content for c in received if c.content)
        assert "Coordinator summary." in streamed_text
        assert "DISTINCT_SUBAGENT_TEXT" not in streamed_text
        assert result["tool_call_counts"]["research_planner"] == 1

    def test_report_never_returns_none(self):
        assert subagent_response([ChatMessage.from_assistant("Aye! Keel lore.")]) == (
            "Aye! Keel lore."
        )
        assert "could not be completed" in subagent_response(None)

        # A run that ends on a bare tool call gathered nothing to hand back.
        tool_only = ChatMessage.from_assistant(
            tool_calls=[ToolCall(tool_name="noop", arguments={"x": 1}, id="c1")]
        )
        fallback = subagent_response([tool_only])
        assert isinstance(fallback, str) and fallback
        assert "Sources consulted" in fallback


def _stream_with_handoffs(*handoff_tools: str) -> Stream:
    """A Stream whose coordinator can delegate to the given subagent tools."""
    from haystack import Pipeline
    from haystack.tools.component_tool import ComponentTool

    tools = [
        ComponentTool(
            component=SubagentStreamFilter(
                ToolAgent.build_agent(MockChatGenerator(responses=["ok"]), [], "sub"),
                name=name.replace("_", " ").title(),
            ),
            name=name,
            description=f"Delegate to {name}",
            parameters={
                "type": "object",
                "properties": {"task": {"type": "string"}},
                "required": ["task"],
            },
        )
        for name in handoff_tools
    ]
    agent = ToolAgent.build_agent(MockChatGenerator(responses=["ok"]), tools, "coordinator")
    pipeline = Pipeline()
    pipeline.add_component("agent", agent)
    return Stream(pipeline=pipeline, generator=agent.chat_generator, store=False)


def _handoff_message(result: str, content: str = "Here is the summary.") -> SDKChatMessage:
    return SDKChatMessage(
        role="assistant",
        content=content,
        tool_calls=[
            {
                "id": "call_1",
                "name": "research_planner",
                "arguments": {"task": "Research Leukeleu"},
                "result": {"result": result, "origin": {"id": "call_1"}, "error": False},
                "handoff": "Research Planner",
            }
        ],
    )


class TestHandoffAttribution:
    def test_stream_discovers_handoff_tools_from_pipeline(self):
        stream = _stream_with_handoffs("research_planner")
        assert stream._handoff_tools == {"research_planner": "Research Planner"}

    def test_attribution_marks_handoff_and_subagent(self):
        stream = _stream_with_handoffs("research_planner")
        assert stream.get_attribution("research_planner") == {"handoff": "Research Planner"}
        assert stream.get_attribution("search_web", "Research Planner") == {
            "agent": "Research Planner"
        }
        assert stream.get_attribution("get_weather") == {}

    def test_hallucinated_self_named_call_is_not_a_handoff(self):
        """A subagent cannot delegate to itself (agent.py skips its own class,
        build_subagent guards cycles), so a subagent hallucinating a tool call
        named after the coordinator's delegation tool must never be labeled as
        the agent handing off to itself — it stays a plain subagent tool call.
        """
        stream = _stream_with_handoffs("research_planner")

        assert stream.get_attribution("research_planner", "Research Planner") == {
            "agent": "Research Planner"
        }

    def test_streamed_chunk_carries_subagent_tag(self):
        from haystack.dataclasses import StreamingChunk, ToolCallDelta

        stream = _stream_with_handoffs("research_planner")
        chunk = StreamingChunk(
            content="",
            index=0,
            meta={SUBAGENT_META_KEY: "Research Planner"},
            tool_calls=[ToolCallDelta(index=0, tool_name="search_web", id="call_9")],
        )
        start = stream.get_streaming_tool_chunks(chunk)[0]
        assert start.metadata == {"agent": "Research Planner"}

    def test_tool_chunks_only_tags_forwarded_chunks(self):
        from haystack.dataclasses import StreamingChunk, ToolCallDelta

        received = []
        forward = _tool_chunks_only(received.append, name="Research Planner", async_sink=False)
        forward(
            StreamingChunk(
                content="",
                index=0,
                tool_calls=[ToolCallDelta(index=0, tool_name="search_web", id="call_9")],
            )
        )
        # Text-only chunks stay dropped.
        forward(StreamingChunk(content="raw subagent text"))

        assert len(received) == 1
        assert received[0].meta[SUBAGENT_META_KEY] == "Research Planner"

    def test_stream_writer_stores_attribution(self):
        writer = StreamWriter(message_id="m1")
        writer.add_chunk(
            MessageChunk(
                type="tool_call_start",
                content={"tool_call_id": "call_9", "tool_name": "search_web"},
                metadata={"agent": "Research Planner"},
            )
        )
        assert writer.message.tool_calls[0]["agent"] == "Research Planner"
        assert "handoff" not in writer.message.tool_calls[0]


class TestHandoffHistoryReplay:
    def test_handoff_replays_as_assistant_tool_pair(self):
        stream = _stream_with_handoffs("research_planner")
        history = [
            SDKChatMessage(role="user", content="Research Leukeleu"),
            _handoff_message("Full report: Leukeleu is a Dutch agency."),
            SDKChatMessage(role="user", content="Which city?"),
        ]

        replayed = stream.get_messages(history)

        assert [m.role.value for m in replayed] == ["user", "assistant", "tool", "user"]
        assert replayed[1].tool_calls[0].tool_name == "research_planner"
        # The report itself is back in context, not just the coordinator's summary.
        assert "Dutch agency" in replayed[2].tool_call_results[0].result
        assert replayed[2].tool_call_results[0].origin.id == "call_1"

    def test_long_result_is_truncated(self, settings):
        settings.AI_SDK_HISTORY_TOOL_OUTPUT_LIMIT = 50
        stream = _stream_with_handoffs("research_planner")

        replayed = stream.get_messages([_handoff_message("x" * 500)])

        result = replayed[1].tool_call_results[0].result
        assert result.startswith("x" * 50)
        assert "truncated, 450 chars omitted" in result

    def test_truncation_disabled_by_zero_limit(self, settings):
        settings.AI_SDK_HISTORY_TOOL_OUTPUT_LIMIT = 0
        stream = _stream_with_handoffs("research_planner")

        replayed = stream.get_messages([_handoff_message("x" * 500)])

        assert replayed[1].tool_call_results[0].result == "x" * 500

    def test_subagent_tool_calls_are_not_replayed(self):
        """`agent` marks a subagent's own tool; only `handoff` replays."""
        stream = _stream_with_handoffs("research_planner")
        message = SDKChatMessage(
            role="assistant",
            content="Summary.",
            tool_calls=[
                {
                    "id": "call_2",
                    "name": "search_web",
                    "arguments": {"query": "x"},
                    "result": {"result": "hits"},
                    "agent": "Research Planner",
                }
            ],
        )

        replayed = stream.get_messages([message])

        assert [m.role.value for m in replayed] == ["assistant"]
        assert not replayed[0].tool_calls

    def test_messages_without_tool_calls_are_unchanged(self):
        """History stored before this change has no handoff key — no KeyError."""
        stream = _stream_with_handoffs("research_planner")
        history = [
            SDKChatMessage(role="user", content="hi"),
            SDKChatMessage(role="assistant", content="hello"),
        ]

        replayed = stream.get_messages(history)

        assert [m.text for m in replayed] == ["hi", "hello"]

    def test_merge_does_not_span_a_handoff(self):
        stream = _stream_with_handoffs("research_planner")
        stream.merge_messages = True
        history = [
            SDKChatMessage(role="user", content="one"),
            SDKChatMessage(role="user", content="two"),
            _handoff_message("report"),
            SDKChatMessage(role="assistant", content="after"),
        ]

        replayed = stream.get_messages(history)

        assert [m.role.value for m in replayed] == ["user", "assistant", "tool", "assistant"]
        # The two user turns merged; the handoff pair stayed intact between them.
        assert replayed[0].text == "one\n\ntwo"

    def test_incomplete_handoff_replays_empty_result(self):
        """A delegation that never returned stores result=None."""
        stream = _stream_with_handoffs("research_planner")
        message = _handoff_message("unused")
        message.tool_calls[0]["result"] = None

        replayed = stream.get_messages([message])

        assert replayed[1].tool_call_results[0].result == ""


def _research_messages(*, report=None, skipped=False):
    """A subagent conversation that gathered real material."""
    search = ToolCall(tool_name="search_web", arguments={"query": "bordeaux"}, id="c1")
    fetch = ToolCall(tool_name="fetch_page", arguments={"url": "https://ap.example"}, id="c2")
    messages = [
        ChatMessage.from_user("Research Bordeaux wildfires"),
        ChatMessage.from_assistant(tool_calls=[search]),
        ChatMessage.from_tool("5 results: Reuters, AP, France24", origin=search),
        ChatMessage.from_assistant(tool_calls=[fetch]),
        ChatMessage.from_tool("France and Spain battle wildfires", origin=fetch),
    ]
    if skipped:
        dropped = ToolCall(tool_name="fetch_page", arguments={"url": "https://f24"}, id="c3")
        messages += [
            ChatMessage.from_assistant(tool_calls=[dropped]),
            ChatMessage.from_tool(
                "Tool budget reached (3 calls); this call was not run.",
                origin=dropped,
                error=True,
                meta={SKIPPED_META_KEY: True},
            ),
        ]
    if report:
        messages.append(ChatMessage.from_assistant(report))
    return messages


class TestSubagentReport:
    def test_report_carries_the_work_when_the_budget_cut_the_run_short(self):
        """The reported bug: budget fires, steps run out, research is discarded."""
        output = subagent_response(_research_messages(skipped=True))

        assert "cut short" in output
        assert "France and Spain battle wildfires" in output
        assert "5 results: Reuters, AP, France24" in output
        # The budget stub is not gathered material.
        assert "Tool budget reached" not in output

    def test_successful_run_returns_report_plus_sources(self):
        output = subagent_response(_research_messages(report="Bordeaux: 220,000 evacuated."))

        assert output.startswith("Bordeaux: 220,000 evacuated.")
        assert "Sources consulted:" in output
        assert "search_web(query='bordeaux')" in output
        assert "fetch_page(url='https://ap.example')" in output
        # No digest when there is a real report.
        assert "cut short" not in output
        assert "France and Spain battle wildfires" not in output

    def test_digest_budget_is_shared_across_results(self, settings):
        settings.AI_SDK_SUBAGENT_DIGEST_LIMIT = 20
        messages = _research_messages()

        output = subagent_response(messages)

        # Both bodies together are capped, not each one separately.
        assert "…" in output
        assert len(output) < 400

    def test_digest_limit_of_zero_disables_truncation(self, settings):
        settings.AI_SDK_SUBAGENT_DIGEST_LIMIT = 0

        output = subagent_response(_research_messages())

        assert "France and Spain battle wildfires" in output


class TestToolCallBudgetHook:
    def _state(self, messages, counts):
        from haystack.components.agents.state.state import State

        state = State(schema={"tool_call_counts": {"type": dict}})
        state.set("messages", messages)
        state.set("tool_call_counts", counts)
        return state

    def test_pending_call_is_answered_not_erased(self):
        """The turn must stay well-formed: tool_calls followed by tool results."""
        call = ToolCall(tool_name="fetch_page", arguments={"url": "u"}, id="c9")
        state = self._state(
            [ChatMessage.from_user("go"), ChatMessage.from_assistant(tool_calls=[call])],
            {"fetch_page": 3},
        )

        ToolCallBudgetHook(max_tool_calls=3).run(state)

        messages = state.data["messages"]
        assert messages[-2].tool_calls == [call]
        assert messages[-1].tool_call_results[0].origin.id == "c9"
        assert messages[-1].meta[SKIPPED_META_KEY] is True
        # Marker survives parse_tool_output as {"skipped": ...} for the client.
        from django_ai_sdk.adapters.base import parse_tool_output

        parsed = parse_tool_output(messages[-1].tool_call_results[0].to_dict())
        assert "skipped" in parsed["result"]
        # No tool_calls on the last message, so nothing executes this step.
        assert not messages[-1].tool_calls

    def test_under_budget_is_untouched(self):
        call = ToolCall(tool_name="fetch_page", arguments={"url": "u"}, id="c9")
        state = self._state([ChatMessage.from_assistant(tool_calls=[call])], {"fetch_page": 1})

        ToolCallBudgetHook(max_tool_calls=3).run(state)

        assert len(state.data["messages"]) == 1


class TestSubagentSynthesis:
    def _filter(self, result, generator):
        agent = MagicMock()
        agent.run.return_value = result
        agent.chat_generator = generator
        return SubagentStreamFilter(agent, name="Research Planner")

    def test_synthesis_runs_when_no_report_was_written(self):
        generator = MagicMock()
        generator.run.return_value = {
            "replies": [ChatMessage.from_assistant("Bordeaux: 220,000 evacuated.")]
        }
        result = {"messages": _research_messages(skipped=True)}

        out = self._filter(result, generator).run(messages=[], task="research")

        assert out["last_message"].text == "Bordeaux: 220,000 evacuated."
        # Called with the gathered conversation and no tools.
        sent = generator.run.call_args.kwargs["messages"]
        assert "tools" not in generator.run.call_args.kwargs
        assert sent[-1].text == FINAL_PROMPT
        assert subagent_response(out["messages"]).startswith("Bordeaux: 220,000 evacuated.")

    def test_synthesis_is_skipped_when_a_report_exists(self):
        generator = MagicMock()
        result = {"messages": _research_messages(report="All good.")}

        self._filter(result, generator).run(messages=[], task="research")

        generator.run.assert_not_called()

    def test_failed_synthesis_falls_through_to_the_digest(self):
        generator = MagicMock()
        generator.run.side_effect = RuntimeError("provider down")
        result = {"messages": _research_messages(skipped=True)}

        out = self._filter(result, generator).run(messages=[], task="research")

        # The run still returns, and the work is still recoverable.
        digest = subagent_response(out["messages"])
        assert "France and Spain battle wildfires" in digest


class TestUnstreamedResults:
    def test_hook_written_result_is_streamed_once(self):
        generator = MagicMock()
        generator.run.return_value = {"replies": [ChatMessage.from_assistant("done")]}
        agent = MagicMock()
        agent.chat_generator = generator
        messages = _research_messages(skipped=True)
        agent.run.return_value = {"messages": messages}
        received = []

        SubagentStreamFilter(agent, name="Research Planner").run(
            messages=[], task="research", streaming_callback=received.append
        )

        # Nothing streamed during the run, so every result is replayed after it.
        ids = [c.tool_call_result.origin.id for c in received]
        assert ids == ["c1", "c2", "c3"]
        assert all(c.meta[SUBAGENT_META_KEY] == "Research Planner" for c in received)

    def test_already_streamed_result_is_not_duplicated(self):
        from haystack.dataclasses import ToolCallResult

        generator = MagicMock()
        generator.run.return_value = {"replies": [ChatMessage.from_assistant("done")]}
        messages = _research_messages(report="done")
        streamed = ToolCallResult(
            result="5 results: Reuters, AP, France24",
            origin=messages[1].tool_calls[0],
            error=False,
        )

        def fake_run(messages=None, streaming_callback=None, **kwargs):
            streaming_callback(
                HaystackStreamingChunk(content="", index=0, tool_call_result=streamed)
            )
            return {"messages": _research_messages(report="done")}

        agent = MagicMock()
        agent.chat_generator = generator
        agent.run.side_effect = fake_run
        received = []

        SubagentStreamFilter(agent, name="Research Planner").run(
            messages=[], task="research", streaming_callback=received.append
        )

        ids = [c.tool_call_result.origin.id for c in received]
        assert ids.count("c1") == 1
