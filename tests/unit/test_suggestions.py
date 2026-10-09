"""Follow-up suggestions come from AI_SDK_TASK_MODEL when set, else the chat agent."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from django_ai_sdk.adapters.base import Run
from django_ai_sdk.adapters.suggestions import DefaultSuggestionGenerator, FollowUpSuggestions
from django_ai_sdk.common import ChatMessage

MESSAGES = [ChatMessage(role="user", content="What is XAF?")]
REPLY = FollowUpSuggestions(follow_ups=["Is postal code mandatory?"])


@pytest.mark.asyncio
async def test_the_chat_agent_suggests_by_default() -> None:
    agent = MagicMock(run=AsyncMock(return_value=REPLY))

    suggestions = await DefaultSuggestionGenerator(agent=agent).generate(MESSAGES, "An audit file.")

    assert suggestions == ["Is postal code mandatory?"]
    assert agent.run.call_args.kwargs["response_format"] is FollowUpSuggestions


@pytest.mark.asyncio
async def test_the_task_model_suggests_when_set(settings: Any) -> None:
    settings.AI_SDK_TASK_MODEL = "openai/gpt-oss-120b"
    agent = MagicMock(run=AsyncMock())

    with patch.object(Run, "run", AsyncMock(return_value=REPLY)) as task_run:
        suggestions = await DefaultSuggestionGenerator(agent=agent).generate(
            MESSAGES, "An audit file."
        )

    assert suggestions == ["Is postal code mandatory?"]
    assert task_run.call_args.kwargs["response_format"] is FollowUpSuggestions
    agent.run.assert_not_called()
