"""Loguru formats with `{}` and takes tracebacks via `opt(exception=...)`, not `exc_info`."""

import asyncio

import pytest
from loguru import logger


@pytest.fixture
def logged():
    records: list[str] = []
    sink = logger.add(lambda m: records.append(str(m)), level="DEBUG", format="{message}")
    yield records
    logger.remove(sink)


@pytest.mark.asyncio
async def test_a_failed_trace_write_logs_its_traceback(logged):
    from django_ai_sdk.tracing.context import _log_write_failure

    async def boom():
        raise RuntimeError("db down")

    task = asyncio.ensure_future(boom())
    await asyncio.gather(task, return_exceptions=True)

    _log_write_failure(task)

    [record] = logged
    assert "Trace write failed: db down" in record
    assert "Traceback" in record and "RuntimeError: db down" in record


@pytest.mark.django_db
@pytest.mark.asyncio
async def test_a_failed_agent_lookup_names_the_agent(logged):
    from django_ai_sdk.agents.services import AgentService

    with pytest.raises(ValueError):
        await AgentService.get("no-such-agent")

    assert any("RuntimeAgent lookup failed for no-such-agent:" in r for r in logged)


def test_tool_call_arguments_stay_out_of_the_info_log():
    """Arguments can carry PII or tokens: INFO names the tool, DEBUG adds the arguments."""
    from unittest.mock import MagicMock

    from haystack.dataclasses import ChatMessage, ToolCall

    from django_ai_sdk.agents.tool_agent import LogToolCallsHook

    records: list[tuple[str, str]] = []
    sink = logger.add(
        lambda m: records.append((m.record["level"].name, m.record["message"])), level="DEBUG"
    )
    call = ToolCall(tool_name="send_email", arguments={"to": "a@b.c", "token": "s3cret"})
    state = MagicMock(data={"messages": [ChatMessage.from_assistant(tool_calls=[call])]})
    try:
        LogToolCallsHook().run(state)
    finally:
        logger.remove(sink)

    info = [msg for level, msg in records if level == "INFO"]
    debug = [msg for level, msg in records if level == "DEBUG"]
    assert info == ["Tool call: send_email"]
    assert any("s3cret" in msg for msg in debug)
