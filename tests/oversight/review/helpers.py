"""Readers with the oversight model permissions, and closed turns to read."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from django.test import override_settings

from django_ai_sdk.oversight.guard import TurnGuard
from django_ai_sdk.oversight.resolve import invalidate
from tests.factories.db import UserFactory, agrant
from tests.oversight.helpers import make_agent

METRICS = "django_ai_sdk_oversight.view_turn"
CONTENT = "django_ai_sdk_oversight.view_turncontent"
REVIEW = "django_ai_sdk_oversight.change_reviewitem"


async def reader(*perms: str, **traits: Any) -> Any:
    """A user holding `perms` (and `staff=True`/`superuser=True` when asked)."""
    user = await UserFactory.acreate(**traits)
    return await agrant(user, *perms)


async def a_turn(
    text: str = "What about my salary?",
    output: str = "Ask jan@example.com.",
    *,
    user: Any = None,
    capture: str = "full",
    policy: str = "test-guard",
    channel: str = "api",
    thread_id: str = "",
) -> str:
    """A closed test-guard turn: "salary" queues it for hr and labels it topic=hr."""
    with override_settings(AI_SDK_OVERSIGHT={"CAPTURE": capture}):
        invalidate()
        guard = await TurnGuard.start(
            make_agent([], None, policy=policy),
            channel=channel,
            input=text,
            user=user,
            thread_id=thread_id or None,
        )
        await guard.gate()
        await guard.finish(output)
        with (
            patch("django_ai_sdk.oversight.persist.enqueue_reflect"),
            patch("django_ai_sdk.oversight.persist.enqueue_notifications"),
        ):
            turn_id = await guard.close()
    invalidate()
    assert turn_id is not None
    return turn_id
