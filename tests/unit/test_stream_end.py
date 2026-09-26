"""The client waits for the end marker: an unexpected error must still send it."""

from unittest.mock import MagicMock, patch

import pytest
from haystack import Pipeline

from django_ai_sdk.adapters.base import Stream


@pytest.mark.asyncio
async def test_a_critical_error_still_ends_the_stream():
    stream = Stream(pipeline=Pipeline(), generator=MagicMock(spec=[]))

    with patch.object(Stream, "get_task", side_effect=RuntimeError("boom")):
        events = [e async for e in stream.stream([])]

    assert [type(e).__name__ for e in events][-2:] == ["ErrorEvent", "StreamEndEvent"]
