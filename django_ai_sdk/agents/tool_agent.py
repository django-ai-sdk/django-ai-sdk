from __future__ import annotations

import json
from typing import Any

from haystack import Pipeline
from haystack.components.agents import Agent as HaystackAgent
from haystack.components.agents.state.state import State, replace_values
from haystack.dataclasses import ChatMessage
from pydantic import BaseModel

from django_ai_sdk.logger import get_logger
from django_ai_sdk.utils import resolve_setting

logger = get_logger(__name__)

# message meta flag on a tool result the budget hook
SKIPPED_META_KEY = "django_ai_sdk.skipped"
# message meta on a cut tool result: {"original_chars", "kept_chars"}
TRUNCATED_META_KEY = "django_ai_sdk.truncated"


class ToolCallBudgetHook:
    """Hard cap on how many tool calls an agent may make in a single run."""

    allowed_hook_points = ["before_tool"]

    def __init__(self, max_tool_calls: int, message: str | None = None) -> None:
        self.max_tool_calls = max_tool_calls
        self.message = message or (
            f"Tool budget reached ({max_tool_calls} calls); this call was not run. "
            "Do not call any more tools; answer directly with what you have now."
        )

    def run(self, state: State) -> None:
        counts = state.data.get("tool_call_counts") or {}
        if sum(counts.values()) < self.max_tool_calls:
            return
        messages = state.data.get("messages") or []
        if not messages or not messages[-1].tool_calls:
            return
        # answer the pending calls with a stub result
        payload = json.dumps({"skipped": self.message})
        skipped = [
            ChatMessage.from_tool(
                payload,
                origin=call,
                error=True,
                meta={SKIPPED_META_KEY: True},
            )
            for call in messages[-1].tool_calls
        ]
        state.set("messages", [*messages, *skipped], handler_override=replace_values)


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
            # Arguments can carry PII or tokens: keep them out of the default log.
            logger.debug("Tool call {} args={}", tool_call.tool_name, tool_call.arguments)


class ToolOutputLimitHook:
    """Cut tool results longer than `max_chars`, with a note to the model.

    Each result is cut once, when it arrives; later steps see it as it was left.
    """

    allowed_hook_points = ["after_tool"]

    # Ahead of the result: a note at the end of a long result is easily missed.
    _NOTE = (
        "[This tool result is {original:,} characters; only the first {kept:,} are shown. "
        "Answer from this part and say briefly that it is partial; a narrower request "
        "can get the rest.]\n\n"
    )

    def __init__(self, max_chars: int) -> None:
        self.max_chars = max_chars

    def run(self, state: State) -> None:
        messages = list(state.data.get("messages") or [])
        changed = False
        for index, message in enumerate(messages):
            result = message.tool_call_result
            if (
                result is None
                or not isinstance(result.result, str)
                or len(result.result) <= self.max_chars
                or TRUNCATED_META_KEY in message.meta
            ):
                continue
            original = len(result.result)
            logger.warning(
                "Cutting tool result {} to {} of {} characters",
                result.origin.tool_name,
                self.max_chars,
                original,
            )
            messages[index] = ChatMessage.from_tool(
                self._NOTE.format(kept=self.max_chars, original=original)
                + result.result[: self.max_chars],
                origin=result.origin,
                error=result.error,
                meta={
                    **message.meta,
                    TRUNCATED_META_KEY: {"original_chars": original, "kept_chars": self.max_chars},
                },
            )
            changed = True
        if changed:
            state.set("messages", messages, handler_override=replace_values)


def tool_output_limit(max_chars: int | None = None) -> int | None:
    """`max_chars`, else ``AI_SDK_TOOL_OUTPUT_LIMIT``; None when 0 or below."""
    limit = (
        max_chars if max_chars is not None else resolve_setting("AI_SDK_TOOL_OUTPUT_LIMIT", 100_000)
    )
    return limit if limit > 0 else None


def default_hooks(agent: Any) -> dict[str, list[Any]]:
    """Standard hooks for a tool-capable agent."""
    hooks: dict[str, list[Any]] = {"before_tool": [LogToolCallsHook()]}
    if agent.max_tool_calls is not None:
        hooks["before_tool"].append(ToolCallBudgetHook(agent.max_tool_calls))
    if (limit := tool_output_limit()) is not None:
        hooks["after_tool"] = [ToolOutputLimitHook(limit)]
    return hooks


class ToolAgentConfig(BaseModel):
    """Configuration for ToolAgent."""

    model_config = {"arbitrary_types_allowed": True}

    model: str
    system_prompt: str
    tools: list[Any] = []

    # Loop exits once the generator replies with a plain text message.
    exit_conditions: list[str] = ["text"]

    # Hard cap on loop iterations per run.
    max_agent_steps: int = 100

    # Optional hard cap on tool calls per run. None disables the budget hook.
    max_tool_calls: int | None = None

    # Characters kept of each tool result. None uses AI_SDK_TOOL_OUTPUT_LIMIT; 0 disables.
    max_tool_output_chars: int | None = None

    # Maximum number of tools invoked in parallel per step.
    tool_concurrency_limit: int = 4

    # Optional state schema passed through to the Haystack Agent.
    state_schema: dict[str, Any] | None = None

    # Optional hooks passed through to the Haystack Agent, keyed by hook point.
    hooks: dict[str, list[Any]] | None = None

    # Forward the agent's streaming callback into tools that accept one
    # subagent wrappers, so sub-agent tool calls stream to the client through
    # the coordinator's event queue.
    stream_subagent_tools: bool = True


class ToolAgent:
    """ToolAgent that wraps a Haystack Agent with tools."""

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
        """Build and return a configured Haystack Agent.

        Shared construction path for both streamed agents (via ToolAgent.pipeline())
        and subagent wrappers. Only non-None loop knobs are
        passed, so defaults match the Haystack Agent's own defaults.

        ``user_prompt`` + ``required_variables`` are used for subagents: they
        make the Agent's ``run()`` accept a ``task`` parameter,
        which is how the coordinator's tool call passes the
        delegated sub-task down.
        """
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

    def _hooks(self) -> dict[str, list[Any]] | None:
        hooks = {point: list(items) for point, items in (self.config.hooks or {}).items()}
        if self.config.max_tool_calls is not None:
            hooks.setdefault("before_tool", []).append(
                ToolCallBudgetHook(self.config.max_tool_calls)
            )
        if (limit := tool_output_limit(self.config.max_tool_output_chars)) is not None:
            hooks.setdefault("after_tool", []).append(ToolOutputLimitHook(limit))
        return hooks or None

    def pipeline(self) -> Pipeline:
        """Build and return the fully connected pipeline."""
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
