from __future__ import annotations

from typing import Any
from unittest.mock import patch

from django.test import override_settings

from django_ai_sdk.oversight.guard import TurnGuard
from django_ai_sdk.oversight.registry import invalidate_policy
from tests.factories.db import UserFactory, agrant
from tests.oversight.helpers import make_agent

METRICS = "django_ai_sdk_oversight.view_turn"
CONTENT = "django_ai_sdk_oversight.view_turncontent"
REVIEW = "django_ai_sdk_oversight.change_reviewitem"


async def reader(*perms: str, **traits: Any) -> Any:
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
    with override_settings(AI_SDK_OVERSIGHT={"CAPTURE": capture}):
        invalidate_policy()
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
            patch("django_ai_sdk.oversight.tasks.enqueue_reflect"),
            patch("django_ai_sdk.oversight.tasks.enqueue_notifications"),
        ):
            turn_id = await guard.close()
    invalidate_policy()
    assert turn_id is not None
    return turn_id
