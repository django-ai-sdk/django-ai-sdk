from __future__ import annotations

import ast
import dataclasses
import json
from typing import Any

from haystack import Pipeline
from haystack.components.agents import Agent as HaystackAgent
from haystack.components.agents.state.state import State, replace_values
from haystack.dataclasses import ChatMessage, ToolCallResult
from pydantic import BaseModel

from django_ai_sdk.logger import get_logger
from django_ai_sdk.utils import resolve_setting

logger = get_logger(__name__)

SKIPPED_META_KEY = "django_ai_sdk.skipped"
TRUNCATED_META_KEY = "django_ai_sdk.truncated"
BLOCKED_META_KEY = "django_ai_sdk.refused"
STREAMED_META_KEY = "django_ai_sdk.streamed"
MASKED_META_KEY = "django_ai_sdk.masked"


def pending_tool_calls(state: State) -> list[ChatMessage] | None:
    messages = state.data.get("messages") or []
    return messages if messages and messages[-1].tool_calls else None


def answer_calls_with(
    state: State, messages: list[ChatMessage], stubs: list[tuple[dict[str, Any], dict[str, Any]]]
) -> list[ChatMessage]:
    answers = [
        ChatMessage.from_tool(
            json.dumps(payload),
            origin=dataclasses.replace(call, arguments={}) if meta.get(BLOCKED_META_KEY) else call,
            error=True,
            meta=meta,
        )
        for call, (payload, meta) in zip(messages[-1].tool_calls, stubs, strict=True)
    ]
    state.set("messages", [*messages, *answers], handler_override=replace_values)
    return answers


def decode_tool_result(value: Any) -> Any:
    """A tool's result as it returned it: Haystack writes a dict as JSON, older results as reprs."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except ValueError:
        pass
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def returned_error(value: Any) -> bool:
    return isinstance(value, dict) and bool(value.get("error"))


def tool_failed(result: ToolCallResult) -> bool:
    return bool(result.error) or returned_error(decode_tool_result(result.result))


def tool_calls_made(state: State) -> int:
    return sum((state.data.get("tool_call_counts") or {}).values())


def tool_results(
    messages: list[ChatMessage], unmarked: str
) -> list[tuple[int, ChatMessage, ToolCallResult]]:
    return [
        (index, message, result)
        for index, message in enumerate(messages)
        if (result := message.tool_call_result) is not None and unmarked not in message.meta
    ]


def replace_messages(
    state: State, messages: list[ChatMessage], changed: dict[int, ChatMessage]
) -> None:
    if changed:
        updated = [changed.get(index, message) for index, message in enumerate(messages)]
        state.set("messages", updated, handler_override=replace_values)


class ToolCallBudgetHook:
    allowed_hook_points = ["before_tool"]

    def __init__(self, max_tool_calls: int, message: str | None = None) -> None:
        self.max_tool_calls = max_tool_calls
        self.message = message or (
            f"Tool budget reached ({max_tool_calls} calls); this call was not run. "
            "Do not call any more tools; answer directly with what you have now."
        )

    def run(self, state: State) -> None:
        if tool_calls_made(state) < self.max_tool_calls:
            return
        messages = pending_tool_calls(state)
        if messages is None:
            return
        stub = ({"skipped": self.message}, {SKIPPED_META_KEY: True})
        answer_calls_with(state, messages, [stub] * len(messages[-1].tool_calls))


class ToolCallBudgetSpentHook:
    allowed_hook_points = ["before_llm"]

    def __init__(self, max_tool_calls: int) -> None:
        self.max_tool_calls = max_tool_calls

    def run(self, state: State) -> None:
        if tool_calls_made(state) >= self.max_tool_calls and (tools := state.data.get("tools")):
            tools[:] = []  # in place: the model and the ToolInvoker share this list


class LogToolCallsHook:
    """Log every tool call an agent makes: its name at INFO, its arguments at DEBUG."""

    allowed_hook_points = ["before_tool"]

    def __init__(self, level: str = "info") -> None:
        self._log = getattr(logger, level, logger.info)

    def run(self, state: State) -> None:
        messages = state.data.get("messages") or []
        if not messages or not messages[-1].tool_calls:
            return
        for tool_call in messages[-1].tool_calls:
            self._log("Tool call: {}", tool_call.tool_name)
            logger.debug("Tool call {} args={}", tool_call.tool_name, tool_call.arguments)


class ToolOutputLimitHook:
    allowed_hook_points = ["after_tool"]

    _NOTE = (
        "[This tool result is {original:,} characters; only the first {kept:,} are shown. "
        "Answer from this part and say briefly that it is partial; a narrower request "
        "can get the rest.]\n\n"
    )

    def __init__(self, max_chars: int) -> None:
        self.max_chars = max_chars

    def run(self, state: State) -> None:
        messages = state.data.get("messages") or []
        changed: dict[int, ChatMessage] = {}
        for index, message, result in tool_results(messages, TRUNCATED_META_KEY):
            if not isinstance(result.result, str) or len(result.result) <= self.max_chars:
                continue
            original = len(result.result)
            logger.warning(
                "Cutting tool result {} to {} of {} characters",
                result.origin.tool_name,
                self.max_chars,
                original,
            )
            changed[index] = ChatMessage.from_tool(
                self._NOTE.format(kept=self.max_chars, original=original)
                + result.result[: self.max_chars],
                origin=result.origin,
                error=result.error,
                meta={
                    **message.meta,
                    TRUNCATED_META_KEY: {"original_chars": original, "kept_chars": self.max_chars},
                },
            )
        replace_messages(state, messages, changed)


def tool_output_limit(max_chars: int | None = None) -> int | None:
    limit = (
        max_chars if max_chars is not None else resolve_setting("AI_SDK_TOOL_OUTPUT_LIMIT", 100_000)
    )
    return limit if limit > 0 else None


REQUIRE_TOOLS_META_KEY = "django_ai_sdk.require_tools"


class RequireToolsHook:
    allowed_hook_points = ["on_exit"]

    def __init__(self, required_tools: list[str] | None = None, max_nudges: int = 2) -> None:
        self.required_tools = list(required_tools or [])
        self.max_nudges = max_nudges

    def run(self, state: State) -> None:
        context = state.data.get("hook_context") or {}
        required = {*self.required_tools, *(context.get("required_tools") or [])}
        if not required:
            return
        available = {getattr(t, "name", None) for t in state.data.get("tools") or []}
        counts = state.data.get("tool_call_counts") or {}
        missing = sorted(n for n in required if n in available and not counts.get(n))
        if not missing:
            return
        messages = state.data.get("messages") or []
        if sum(1 for m in messages if m.meta.get(REQUIRE_TOOLS_META_KEY)) >= self.max_nudges:
            logger.warning(f"Required tools never called: {missing}")
            return
        nudge = ChatMessage.from_system(
            f"Before answering you must call: {', '.join(missing)}. Call it now, then answer.",
            meta={REQUIRE_TOOLS_META_KEY: True},
        )
        state.set("messages", [nudge])
        state.set("continue_run", True)


def default_hooks(agent: Any) -> dict[str, list[Any]]:
    return _tool_hooks(
        {"before_tool": [LogToolCallsHook()]},
        max_tool_calls=agent.max_tool_calls,
        max_output_chars=None,
        required_tools=getattr(agent, "required_tools", None),
    )


def _tool_hooks(
    base: dict[str, list[Any]] | None,
    *,
    max_tool_calls: int | None,
    max_output_chars: int | None,
    required_tools: list[str] | None,
) -> dict[str, list[Any]]:
    hooks = {point: list(items) for point, items in (base or {}).items()}
    replaced: tuple[type, ...] = ()
    if max_tool_calls is not None:
        replaced += (ToolCallBudgetHook, ToolCallBudgetSpentHook)
    if max_output_chars is not None:
        replaced += (ToolOutputLimitHook,)
    before = [h for h in hooks.get("before_tool", []) if not isinstance(h, replaced)]
    after = [h for h in hooks.get("after_tool", []) if not isinstance(h, replaced)]
    before_llm = [h for h in hooks.get("before_llm", []) if not isinstance(h, replaced)]
    if max_tool_calls is not None:
        before.append(ToolCallBudgetHook(max_tool_calls))
        before_llm.append(ToolCallBudgetSpentHook(max_tool_calls))
    if (
        not any(isinstance(h, ToolOutputLimitHook) for h in after)
        and (limit := tool_output_limit(max_output_chars)) is not None
    ):
        after.append(ToolOutputLimitHook(limit))
    hooks["before_tool"] = before
    hooks["after_tool"] = after
    if before_llm:
        hooks["before_llm"] = before_llm
    on_exit = hooks.setdefault("on_exit", [])
    if not any(isinstance(h, RequireToolsHook) for h in on_exit):
        on_exit.append(RequireToolsHook(required_tools))
    return hooks


class ToolAgentConfig(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    model: str
    system_prompt: str
    tools: list[Any] = []

    # Loop exits once the generator replies with a plain text message.
    exit_conditions: list[str] = ["text"]

    max_agent_steps: int = 100

    # Optional hard cap on tool calls per run. None disables the budget hook.
    max_tool_calls: int | None = None

    # Characters kept of each tool result. None uses AI_SDK_TOOL_OUTPUT_LIMIT; 0 disables.
    max_tool_output_chars: int | None = None

    tool_concurrency_limit: int = 4

    state_schema: dict[str, Any] | None = None

    hooks: dict[str, list[Any]] | None = None

    required_tools: list[str] = []

    stream_subagent_tools: bool = True


class ToolAgent:
    def __init__(
        self,
        config: ToolAgentConfig,
        generator: Any,
    ) -> None:
        self.config = config
        self.generator = generator
        logger.debug(f"ToolAgent initialized with {len(config.tools)} tools")

    @staticmethod
    def build_agent(
        generator: Any,
        tools: list[Any],
        system_prompt: str,
        *,
        user_prompt: str | None = None,
        required_variables: list[str] | None = None,
        exit_conditions: list[str] | None = None,
        max_agent_steps: int | None = None,
        tool_concurrency_limit: int | None = None,
        state_schema: dict[str, Any] | None = None,
        hooks: dict[str, list[Any]] | None = None,
        stream_subagent_tools: bool = True,
    ) -> HaystackAgent:
        """Only non-None loop knobs are passed, keeping Haystack's defaults."""
        kwargs: dict[str, Any] = {
            "chat_generator": generator,
            "tools": tools,
            "system_prompt": system_prompt,
        }
        if user_prompt is not None:
            kwargs["user_prompt"] = user_prompt
        if required_variables is not None:
            kwargs["required_variables"] = required_variables
        if exit_conditions is not None:
            kwargs["exit_conditions"] = exit_conditions
        if max_agent_steps is not None:
            kwargs["max_agent_steps"] = max_agent_steps
        if tool_concurrency_limit is not None:
            kwargs["tool_concurrency_limit"] = tool_concurrency_limit
        if state_schema is not None:
            kwargs["state_schema"] = state_schema
        if hooks is not None:
            kwargs["hooks"] = hooks
        kwargs["tool_streaming_callback_passthrough"] = stream_subagent_tools
        return HaystackAgent(**kwargs)

    def _hooks(self) -> dict[str, list[Any]]:
        config = self.config
        return _tool_hooks(
            config.hooks,
            max_tool_calls=config.max_tool_calls,
            max_output_chars=config.max_tool_output_chars,
            required_tools=config.required_tools,
        )

    def pipeline(self) -> Pipeline:
        logger.debug("Building ToolAgent pipeline")

        agent = self.build_agent(
            self.generator,
            self.config.tools,
            self.config.system_prompt,
            exit_conditions=self.config.exit_conditions,
            max_agent_steps=self.config.max_agent_steps,
            tool_concurrency_limit=self.config.tool_concurrency_limit,
            state_schema=self.config.state_schema,
            hooks=self._hooks(),
            stream_subagent_tools=self.config.stream_subagent_tools,
        )

        pipeline = Pipeline()
        pipeline.add_component("agent", agent)

        logger.debug("ToolAgent pipeline built successfully")
        return pipeline
