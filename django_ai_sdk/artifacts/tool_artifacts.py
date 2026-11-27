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
#: artifact this time. May be async.
ArtifactBuilder = Callable[
    [dict[str, Any], Any, str], "dict[str, Any] | None | Awaitable[dict[str, Any] | None]"
]


@dataclasses.dataclass(frozen=True)
class ToolArtifact:
    artifact: type[ArtifactSchema]
    build: ArtifactBuilder


async def build_artifact(
    spec: ToolArtifact,
    arguments: dict[str, Any],
    result: Any,
    thread_id: str,
    user: AbstractBaseUser | AnonymousUser | None = None,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
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
