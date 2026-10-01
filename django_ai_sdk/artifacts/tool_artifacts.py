"""Show an artifact after a tool ran, without the model having to call it.

Declared on an Agent by tool name, so it also works for tools the project didn't
write (package or hidden tools like ``ask_image``)::

    class MyAgent(Agent):
        tool_artifacts = {ASK_IMAGE_TOOL: ASK_IMAGE_ARTIFACT}

After the tool returns, the artifact is stored and streamed as its own
``artifact_*`` tool part, exactly as if the model had called the artifact tool: the
frontend renders it with the component it already has, and it is saved with the
message, so a reloaded thread shows it too. The model only gets the tool's own result.

Needs a streaming run (the artifact is a stream part); in ``Agent.run()`` the tool
simply runs without one.
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import itertools
import json
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, cast

from haystack.dataclasses import StreamingChunk, ToolCall, ToolCallDelta, ToolCallResult
from haystack.tools import Tool
from pydantic import ValidationError

from django_ai_sdk.logger import get_logger

if TYPE_CHECKING:
    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser

    from django_ai_sdk.artifacts.schemas import ArtifactSchema

logger = get_logger(__name__)

#: ``(tool arguments, tool result, thread id) -> artifact data``, or None for no
#: artifact this time. May be async.
ArtifactBuilder = Callable[
    [dict[str, Any], Any, str], "dict[str, Any] | None | Awaitable[dict[str, Any] | None]"
]

# Streaming chunks need an index; the model's own tool calls use small ones.
_indexes = itertools.count(10_000)


@dataclasses.dataclass(frozen=True)
class ToolArtifact:
    """Which artifact to show after a tool, and how to build its data from the call."""

    artifact: type[ArtifactSchema]
    build: ArtifactBuilder


def with_tool_artifact(
    tool: Tool,
    spec: ToolArtifact,
    *,
    thread_id: str,
    user: AbstractBaseUser | AnonymousUser | None = None,
) -> Tool:
    """A copy of `tool` that streams `spec`'s artifact after it returns.

    The tool's own result is never changed: a builder returning None, or data the
    artifact rejects, just means no artifact (logged).
    """
    async_fn, sync_fn = tool.async_function, tool.function
    original = async_fn or sync_fn
    if original is None:  # a Tool always has one; keeps the types honest
        raise ValueError(f"Tool {tool.name!r} has no function to wrap")
    forwards_callback = "streaming_callback" in inspect.signature(original).parameters

    async def run(streaming_callback: Any = None, **kwargs: Any) -> Any:
        if forwards_callback:
            kwargs["streaming_callback"] = streaming_callback
        if async_fn is not None:
            result = await async_fn(**kwargs)
        else:
            result = await asyncio.to_thread(original, **kwargs)
        if streaming_callback is not None and thread_id:
            arguments = {k: v for k, v in kwargs.items() if k != "streaming_callback"}
            await _stream_artifact(spec, arguments, result, thread_id, user, streaming_callback)
        return result

    return dataclasses.replace(tool, function=None, async_function=run)


async def _stream_artifact(
    spec: ToolArtifact,
    arguments: dict[str, Any],
    result: Any,
    thread_id: str,
    user: AbstractBaseUser | AnonymousUser | None,
    streaming_callback: Callable[[StreamingChunk], Any],
) -> None:
    name = spec.artifact.tool_name()
    try:
        data = spec.build(arguments, result, thread_id)
        if inspect.isawaitable(data):
            data = cast("dict[str, Any] | None", await data)
        if data is None:
            return
        payload = await spec.artifact.store(data, thread_id, user)
    except (ValidationError, ValueError) as e:
        logger.warning("No {} after tool call: {}", name, e)
        return

    call = ToolCall(id=f"{name}-{payload['artifact_id']}", tool_name=name, arguments=data)
    index = next(_indexes)
    for chunk in (
        StreamingChunk(
            content="",
            index=index,
            tool_calls=[
                ToolCallDelta(index=index, id=call.id, tool_name=name, arguments=json.dumps(data))
            ],
        ),
        StreamingChunk(
            content="",
            index=index,
            tool_call_result=ToolCallResult(result=json.dumps(payload), origin=call, error=False),
        ),
    ):
        sent = streaming_callback(chunk)
        if inspect.isawaitable(sent):
            await sent


def apply_tool_artifacts(
    tools: list[Any],
    tool_artifacts: dict[str, ToolArtifact],
    *,
    thread_id: str,
    user: AbstractBaseUser | AnonymousUser | None = None,
) -> list[Any]:
    """`tools` with every plain Tool named in `tool_artifacts` wrapped."""
    if not tool_artifacts:
        return tools
    return [
        with_tool_artifact(tool, tool_artifacts[tool.name], thread_id=thread_id, user=user)
        if type(tool) is Tool and tool.name in tool_artifacts
        else tool
        for tool in tools
    ]
