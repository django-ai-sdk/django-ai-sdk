from __future__ import annotations

from typing import TYPE_CHECKING, cast

from django.conf import settings
from django.http import StreamingHttpResponse

from django_ai_sdk.logger import get_logger
from django_ai_sdk.protocols.utils import format_sse

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Coroutine

    from django_ai_sdk.adapters.protocols import Streamable
    from django_ai_sdk.common import ChatMessage
    from django_ai_sdk.protocols.base import BaseProtocolHandler

logger = get_logger(__name__)


async def _ensure_adapter(
    adapter: Streamable | Callable[[], Coroutine[None, None, Streamable]],
    messages: list[ChatMessage],
    protocol_handler: BaseProtocolHandler,
) -> AsyncGenerator[bytes, None]:
    yield format_sse({"type": "data-warmup", "data": {"status": "start"}, "transient": True})

    try:
        if callable(adapter):
            factory = cast("Callable[[], Coroutine[None, None, Streamable]], adapter")
            adapter = await factory()
    except Exception:
        logger.error("Adapter initialization failed", exc_info=True)
        yield format_sse({"type": "data-warmup", "data": {"status": "failed"}, "transient": True})
        yield format_sse("[DONE]")
        return

    yield format_sse({"type": "data-warmup", "data": {"status": "ready"}, "transient": True})
    async for chunk in protocol_handler.sse(adapter, messages):
        yield chunk


async def stream_response(
    adapter: Streamable | Callable[[], Coroutine[None, None, Streamable]],
    messages: list[ChatMessage],
    protocol_handler: BaseProtocolHandler,
    extra_headers: dict[str, str] | None = None,
) -> StreamingHttpResponse:
    """
    Generic streaming chat view that works with any pipeline adapter and protocol handler.

    Args:
        adapter: Pipeline adapter instance or async factory function
        messages: List of chat messages to process
        protocol_handler: Protocol handler instance for formatting output
        extra_headers: Optional additional headers to include in response

    Returns:
        StreamingHttpResponse with SSE-formatted AI responses
    """
    logger.debug(
        f"Stream response initiated: adapter={type(adapter).__name__ if not callable(adapter) else 'factory'}, messages={len(messages)}, protocol={type(protocol_handler).__name__}"
    )

    sse_stream = _ensure_adapter(adapter, messages, protocol_handler)

    # Build streaming HTTP response
    response = StreamingHttpResponse(  # type: ignore[arg-type]
        sse_stream,
        content_type="text/event-stream",
    )

    # Default SSE headers
    response["Cache-Control"] = "no-cache"
    cors_origin = getattr(settings, "AI_SDK_STREAM_CORS_ORIGIN", None)
    if cors_origin:
        response["Access-Control-Allow-Origin"] = cors_origin
    response["Access-Control-Allow-Headers"] = "Cache-Control"

    # TODO: Vercel AI UI message stream version, needs to be optional
    response["x-vercel-ai-ui-message-stream"] = "v1"

    # Add any extra headers, this might be useful for CORS or other custom headers
    if extra_headers:
        logger.debug(f"Adding {len(extra_headers)} extra headers")
        for key, value in extra_headers.items():
            response[key] = value

    logger.debug("Stream response configured and ready")
    return response
