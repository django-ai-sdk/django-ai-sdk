"""Show an artifact after a tool ran, without the model having to call it.

Declared on an Agent by tool name, so it also works for tools the project didn't
write (package, MCP or hidden tools like ``ask_image``)::

    class MyAgent(Agent):
        tool_artifacts = {ASK_IMAGE_TOOL: ASK_IMAGE_ARTIFACT}

When the stream sees that tool's result, the artifact is built, stored and streamed
as its own ``artifact_*`` tool part: the frontend renders it with the component it
already has, and it is saved with the message, so a reloaded thread shows it too.
The model only gets the tool's own result.

Streamed runs only (the artifact is a stream part); in ``Agent.run()`` the tool
simply runs without one.
"""

from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, cast

from pydantic import ValidationError

from django_ai_sdk.logger import get_logger

if TYPE_CHECKING:
    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser

    from django_ai_sdk.artifacts.schemas import ArtifactSchema

logger = get_logger(__name__)

#: ``(tool arguments, tool result, thread id) -> artifact data``, or None for no
#: artifact this time. May be async. The result is what the tool returned, as JSON
#: data (a list or dict comes back as one).
ArtifactBuilder = Callable[
    [dict[str, Any], Any, str], "dict[str, Any] | None | Awaitable[dict[str, Any] | None]"
]


@dataclasses.dataclass(frozen=True)
class ToolArtifact:
    """Which artifact to show after a tool, and how to build its data from the call."""

    artifact: type[ArtifactSchema]
    build: ArtifactBuilder


async def build_artifact(
    spec: ToolArtifact,
    arguments: dict[str, Any],
    result: Any,
    thread_id: str,
    user: AbstractBaseUser | AnonymousUser | None = None,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Build and store `spec`'s artifact for one tool call: `(data, payload)`, or None.

    None when the builder returns None, or when the artifact rejects the data
    (logged): the tool's own result is never affected.
    """
    try:
        data = spec.build(arguments, result, thread_id)
        if inspect.isawaitable(data):
            data = cast("dict[str, Any] | None", await data)
        if data is None:
            return None
        return data, await spec.artifact.store(data, thread_id, user)
    except (ValidationError, ValueError) as e:
        logger.warning("No {} after tool call: {}", spec.artifact.tool_name(), e)
        return None
