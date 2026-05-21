"""Helper for handing off queries to other agents (swarm pattern)."""

import asyncio
from typing import Any, Literal

from pydantic import BaseModel

from django_ai_sdk.assistants.services import AssistantService
from django_ai_sdk.common import ChatMessage
from django_ai_sdk.logger import get_logger

logger = get_logger(__name__)


class HandoffResult(BaseModel):
    """Structured result from handing off to another agent."""

    specialist: str
    query: str
    status: Literal["success", "error"]
    answer: str | None = None
    error: str | None = None


def invoke_assistant_for_query(
    assistant_id: str,
    query: str,
    *,
    specialist: str = "",
) -> HandoffResult:
    """
    Run an assistant's full pipeline on a single query (no thread context).

    Works synchronously by creating/managing its own event loop.
    Safe to call from thread pools (e.g., Haystack tool execution).
    """
    logger.debug(f"Invoking assistant {assistant_id} with query: {query[:100]}...")

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    try:
        result = loop.run_until_complete(
            _invoke_assistant_for_query_async(assistant_id, query)
        )
        return HandoffResult(
            specialist=specialist,
            query=query,
            status="success",
            answer=result["answer"],
            error=None,
        )
    except Exception as e:
        logger.error(f"Handoff failed for assistant {assistant_id}: {e}", exc_info=True)
        return HandoffResult(
            specialist=specialist,
            query=query,
            status="error",
            answer=None,
            error=str(e),
        )


async def _invoke_assistant_for_query_async(
    assistant_id: str,
    query: str,
) -> dict[str, Any]:
    """Internal async implementation. Streams assistant response."""
    assistant = AssistantService.from_registry(assistant_id)
    adapter = await assistant.get_pipeline_adapter(thread_id=None)
    messages = [ChatMessage(role="user", content=query)]
    answer = ""

    async for event in adapter.stream(messages):
        if event.event_type == "text_chunk":
            answer += event.content

    return {"answer": answer}
