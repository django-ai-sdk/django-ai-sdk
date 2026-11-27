from __future__ import annotations

import pytest

from django_ai_sdk.protocols.openai import OpenAIProtocolHandler
from django_ai_sdk.protocols.vercel import VercelProtocolHandler
from django_ai_sdk.responses import stream_response

VERCEL_HEADER = "x-vercel-ai-ui-message-stream"


async def never_built():
    raise AssertionError


@pytest.mark.asyncio
async def test_only_the_vercel_protocol_sends_the_vercel_header():
    vercel = await stream_response(never_built, [], VercelProtocolHandler())
    openai = await stream_response(never_built, [], OpenAIProtocolHandler())

    assert vercel[VERCEL_HEADER] == "v1"
    assert VERCEL_HEADER not in openai
