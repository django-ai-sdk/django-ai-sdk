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
