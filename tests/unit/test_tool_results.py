"""What the model gets back from tools: failures as they are, oversized results cut."""

from __future__ import annotations

from haystack.dataclasses import ChatMessage, ToolCall
from haystack.tools import Tool

from django_ai_sdk.agents import ToolAgent, ToolAgentConfig
from django_ai_sdk.agents.tool_agent import TRUNCATED_META_KEY, ToolOutputLimitHook, default_hooks


class ScriptedGenerator:
    """Replies from a script, keeping what each call was sent."""

    def __init__(self, replies: list[ChatMessage]) -> None:
        self.replies = list(replies)
        self.sent: list[list[ChatMessage]] = []

    def run(self, messages: list[ChatMessage], tools=None, **kwargs):
        self.sent.append(list(messages))
        return {"replies": [self.replies.pop(0)]}


def call(name: str, id: str = "c1") -> ChatMessage:
    return ChatMessage.from_assistant(tool_calls=[ToolCall(tool_name=name, arguments={}, id=id)])


def tool(name: str, function) -> Tool:
    return Tool(
        name=name,
        description=name,
        parameters={"type": "object", "properties": {}},
        function=function,
    )


def tool_results(result: dict) -> list:
    return [m.tool_call_result for m in result["messages"] if m.tool_call_result is not None]


class TestToolErrors:
    def test_the_model_sees_why_a_tool_failed(self):
        """So it can correct itself, e.g. fix its arguments or pick an existing tool."""

        def broken():
            raise ValueError("city must include a country code")

        generator = ScriptedGenerator([call("weather"), ChatMessage.from_assistant("Retrying.")])
        agent = ToolAgent.build_agent(generator, [tool("weather", broken)], "be brief")

        [failed] = tool_results(agent.run(messages=[ChatMessage.from_user("hi")]))

        assert failed.error
        assert "city must include a country code" in failed.result


class TestToolOutputLimit:
    def run(self, generator, function, max_chars=1_000):
        agent = ToolAgent.build_agent(
            generator,
            [tool("wiki", function)],
            "be brief",
            hooks={"after_tool": [ToolOutputLimitHook(max_chars)]},
        )
        return agent.run(messages=[ChatMessage.from_user("read it all")])

    def test_the_model_gets_the_start_and_a_note_up_front(self):
        generator = ScriptedGenerator([call("wiki"), ChatMessage.from_assistant("Summary.")])

        [cut] = tool_results(self.run(generator, lambda: "x" * 5_000))

        note, body = cut.result.split("\n\n", 1)
        assert note.startswith("[This tool result is 5,000 characters; only the first 1,000")
        assert body == "x" * 1_000
        assert generator.sent[-1][-1].tool_call_result.result == cut.result

    def test_a_result_within_the_limit_is_left_alone(self):
        generator = ScriptedGenerator([call("wiki"), ChatMessage.from_assistant("Summary.")])

        [result] = tool_results(self.run(generator, lambda: "x" * 1_000))

        assert result.result == "x" * 1_000

    def test_a_result_is_cut_once(self):
        """A later step's hook run leaves the earlier cut, and its note, as they were."""
        generator = ScriptedGenerator(
            [call("wiki", "c1"), call("wiki", "c2"), ChatMessage.from_assistant("Summary.")]
        )

        result = self.run(generator, lambda: "x" * 5_000)

        cut_messages = [m for m in result["messages"] if m.tool_call_result is not None]
        assert len(cut_messages) == 2
        for message in cut_messages:
            assert message.meta[TRUNCATED_META_KEY] == {"original_chars": 5_000, "kept_chars": 1_000}
            assert message.tool_call_result.result.count("[This tool result is") == 1


class TestToolOutputLimitWiring:
    def config(self, **kwargs) -> ToolAgentConfig:
        return ToolAgentConfig(model="m", system_prompt="be brief", **kwargs)

    def limit_hooks(self, config: ToolAgentConfig) -> list:
        hooks = ToolAgent(config, generator=None)._hooks() or {}
        return [h for h in hooks.get("after_tool", []) if isinstance(h, ToolOutputLimitHook)]

    def test_on_by_default(self):
        [hook] = self.limit_hooks(self.config())
        assert hook.max_chars == 100_000

    def test_the_setting_changes_it(self, settings):
        settings.AI_SDK_TOOL_OUTPUT_LIMIT = 20_000
        [hook] = self.limit_hooks(self.config())
        assert hook.max_chars == 20_000

    def test_an_agent_overrides_the_setting(self, settings):
        settings.AI_SDK_TOOL_OUTPUT_LIMIT = 20_000
        [hook] = self.limit_hooks(self.config(max_tool_output_chars=500_000))
        assert hook.max_chars == 500_000

    def test_zero_turns_it_off(self):
        assert self.limit_hooks(self.config(max_tool_output_chars=0)) == []

    def test_subagents_and_run_get_it_too(self):
        class Agent:
            max_tool_calls = None

        [hook] = default_hooks(Agent())["after_tool"]
        assert isinstance(hook, ToolOutputLimitHook)
