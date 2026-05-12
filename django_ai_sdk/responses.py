from collections.abc import Awaitable, Callable

from django.http import StreamingHttpResponse

from django_ai_sdk.adapters.base import BasePipelineAdapter
from django_ai_sdk.common import ChatMessage
from django_ai_sdk.logger import get_logger
from django_ai_sdk.protocols.base import BaseProtocolHandler

logger = get_logger(__name__)


async def stream_response(
    adapter: BasePipelineAdapter | Callable[[], BasePipelineAdapter],
    messages: list[ChatMessage],
    protocol_handler: BaseProtocolHandler,
    extra_headers: dict[str, str] | None = None,
    on_complete: Callable[[], Awaitable[None]] | None = None,
) -> StreamingHttpResponse:
    """
    Generic streaming chat view that works with any pipeline adapter and protocol handler.

    Args:
        adapter: Pipeline adapter instance or factory function that returns an adapter
        messages: List of chat messages to process
        protocol_handler: Protocol handler instance for formatting output
        extra_headers: Optional additional headers to include in response

    Returns:
        StreamingHttpResponse with SSE-formatted AI responses
    """
    logger.debug(
        f"Stream response initiated: adapter={type(adapter).__name__ if not callable(adapter) else 'factory'}, messages={len(messages)}, protocol={type(protocol_handler).__name__}"
    )

    adapter = adapter() if callable(adapter) else adapter
    logger.debug(f"Adapter resolved: {type(adapter).__name__}")

    logger.debug("Creating streaming response with SSE headers")
    # Note: sse() is an async generator method, calling it returns a coroutine
    # that resolves to an async generator. StreamingHttpResponse handles this.
    sse_stream = protocol_handler.sse(adapter, messages)

    if on_complete is not None:
        async def _with_on_complete():
            async for chunk in sse_stream:
                yield chunk
            try:
                await on_complete()
            except Exception as completion_error:
                logger.error(f"stream on_complete failed: {completion_error}")

        body = _with_on_complete()
    else:
        body = sse_stream

    response = StreamingHttpResponse(  # type: ignore[arg-type]
        body,
        content_type="text/event-stream",
    )

    # Default SSE headers
    response["Cache-Control"] = "no-cache"
    response["Access-Control-Allow-Origin"] = "*"
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
