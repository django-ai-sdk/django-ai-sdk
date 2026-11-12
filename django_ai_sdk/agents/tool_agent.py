from __future__ import annotations

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

# Message meta keys the SDK sets, all under "django_ai_sdk.".
# On a stub result for a call the budget hook skipped.
SKIPPED_META_KEY = "django_ai_sdk.skipped"
# On a cut tool result: {"original_chars", "kept_chars"}.
TRUNCATED_META_KEY = "django_ai_sdk.truncated"
# On a stub result for a call oversight refused (True) or skipped with it (False).
REFUSED_META_KEY = "django_ai_sdk.refused"
# On a tool result oversight has checked for masking.
MASKED_META_KEY = "django_ai_sdk.masked"


def pending_tool_calls(state: State) -> list[ChatMessage] | None:
    """Return the run's messages when the last one calls tools, else None."""
    messages = state.data.get("messages") or []
    return messages if messages and messages[-1].tool_calls else None


def answer_calls_with(
    state: State, messages: list[ChatMessage], stubs: list[tuple[dict[str, Any], dict[str, Any]]]
) -> list[ChatMessage]:
    """Answer the last message's tool calls with error stubs, so they never run.

    `stubs` holds a (payload, meta) pair per call. Returns the stub messages.
    """
    answers = [
        ChatMessage.from_tool(json.dumps(payload), origin=call, error=True, meta=meta)
        for call, (payload, meta) in zip(messages[-1].tool_calls, stubs, strict=True)
    ]
    state.set("messages", [*messages, *answers], handler_override=replace_values)
    return answers


def tool_results(
    messages: list[ChatMessage], unmarked: str
) -> list[tuple[int, ChatMessage, ToolCallResult]]:
    """Return the tool results without the `unmarked` meta key, with their index."""
    return [
        (index, message, result)
        for index, message in enumerate(messages)
        if (result := message.tool_call_result) is not None and unmarked not in message.meta
    ]


def replace_messages(
    state: State, messages: list[ChatMessage], changed: dict[int, ChatMessage]
) -> None:
    """Replace the messages at the indexes in `changed`. Nothing happens when it is empty."""
    if changed:
        updated = [changed.get(index, message) for index, message in enumerate(messages)]
        state.set("messages", updated, handler_override=replace_values)


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
        messages = pending_tool_calls(state)
        if messages is None:
            return
        stub = ({"skipped": self.message}, {SKIPPED_META_KEY: True})
        answer_calls_with(state, messages, [stub] * len(messages[-1].tool_calls))


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
    """`max_chars`, else ``AI_SDK_TOOL_OUTPUT_LIMIT``; None when 0 or below."""
    limit = (
        max_chars if max_chars is not None else resolve_setting("AI_SDK_TOOL_OUTPUT_LIMIT", 100_000)
    )
    return limit if limit > 0 else None


# message meta flag on the nudge the require-tools hook adds
REQUIRE_TOOLS_META_KEY = "django_ai_sdk.require_tools"


class RequireToolsHook:
    """Keep the agent running until every required tool has been called.

    Required = `required_tools` plus `hook_context["required_tools"]` of the
    run. Tools not in the current toolset are ignored, and the model is nudged
    at most `max_nudges` times per run, so a stubborn model can't loop.
    """

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
    """Standard hooks for a tool-capable agent."""
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
    """Add the SDK's tool hooks to `base`: call budget, output limit, required tools.

    `base` may already hold them (from `default_hooks`), so each kind is added
    once and a limit passed here wins over one already in `base`. Oversight adds
    its own hooks when the agent runs (see `oversight.runs.attach`).
    """
    hooks = {point: list(items) for point, items in (base or {}).items()}
    replaced: tuple[type, ...] = ()
    if max_tool_calls is not None:
        replaced += (ToolCallBudgetHook,)
    if max_output_chars is not None:
        replaced += (ToolOutputLimitHook,)
    before = [h for h in hooks.get("before_tool", []) if not isinstance(h, replaced)]
    after = [h for h in hooks.get("after_tool", []) if not isinstance(h, replaced)]
    if max_tool_calls is not None:
        before.append(ToolCallBudgetHook(max_tool_calls))
    if (
        not any(isinstance(h, ToolOutputLimitHook) for h in after)
        and (limit := tool_output_limit(max_output_chars)) is not None
    ):
        after.append(ToolOutputLimitHook(limit))
    hooks["before_tool"] = before
    hooks["after_tool"] = after
    on_exit = hooks.setdefault("on_exit", [])
    if not any(isinstance(h, RequireToolsHook) for h in on_exit):
        on_exit.append(RequireToolsHook(required_tools))
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

    # Tools the agent must call before it may finish (see RequireToolsHook).
    required_tools: list[str] = []

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

    def _hooks(self) -> dict[str, list[Any]]:
        config = self.config
        return _tool_hooks(
            config.hooks,
            max_tool_calls=config.max_tool_calls,
            max_output_chars=config.max_tool_output_chars,
            required_tools=config.required_tools,
        )

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
