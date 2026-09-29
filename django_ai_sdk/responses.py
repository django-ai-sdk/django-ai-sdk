from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, cast

from django.http import StreamingHttpResponse

from django_ai_sdk.common import StreamWriter
from django_ai_sdk.errors import describe_error
from django_ai_sdk.events import MessageStartEvent, StreamEndEvent
from django_ai_sdk.logger import get_logger
from django_ai_sdk.protocols.utils import format_sse
from django_ai_sdk.utils import resolve_setting

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Coroutine

    from django_ai_sdk.adapters.interfaces import Streamable
    from django_ai_sdk.adapters.suggestions import SuggestionGenerator
    from django_ai_sdk.common import ChatMessage
    from django_ai_sdk.events import StreamEvent
    from django_ai_sdk.protocols.base import BaseProtocolHandler
    from django_ai_sdk.storage.base import BaseStorageAdapter

logger = get_logger(__name__)


class _FailedStream:
    """Streams a single error, so a failed adapter build is formatted by the protocol handler."""

    model: str | None = None
    instructions: str | None = None
    suggestion_generator: SuggestionGenerator | None = None

    def __init__(self, event: StreamEvent, message_id: str) -> None:
        self.event = event
        self.message_id = message_id

    async def stream(self, messages: list[ChatMessage]) -> AsyncGenerator[StreamEvent, None]:
        yield MessageStartEvent(message_id=self.message_id)
        yield self.event
        yield StreamEndEvent()


async def _ensure_adapter(
    adapter: Streamable | Callable[[], Coroutine[None, None, Streamable]],
    messages: list[ChatMessage],
    protocol_handler: BaseProtocolHandler,
    storage_adapter: BaseStorageAdapter | None = None,
) -> AsyncGenerator[bytes, None]:
    yield format_sse({"type": "data-warmup", "data": {"status": "start"}, "transient": True})

    try:
        if callable(adapter):
            factory = cast("Callable[[], Coroutine[None, None, Streamable]]", adapter)
            adapter = await factory()
    except Exception as exc:
        # Imported here: adapters.base imports agents, which imports this module.
        from django_ai_sdk.adapters.base import get_error_chunk, get_error_event

        info = describe_error(exc)
        logger.opt(exception=exc).error(
            "Adapter initialization failed [{}] {}: {}: {}",
            info.ref,
            info.code,
            type(exc).__name__,
            exc,
        )
        message_id = str(uuid.uuid4())
        if storage_adapter:
            # No pipeline ran to store the reply, so a reload would show no error.
            writer = StreamWriter(
                message_id=message_id, storage_callback=storage_adapter.storage_callback
            )
            writer.add_chunk(get_error_chunk(exc, info))
            await writer.finalize("error")
        yield format_sse({"type": "data-warmup", "data": {"status": "failed"}, "transient": True})
        async for chunk in protocol_handler.sse(
            _FailedStream(get_error_event(info), message_id), messages
        ):
            yield chunk
        return

    yield format_sse({"type": "data-warmup", "data": {"status": "ready"}, "transient": True})
    async for chunk in protocol_handler.sse(adapter, messages):
        yield chunk


async def stream_response(
    adapter: Streamable | Callable[[], Coroutine[None, None, Streamable]],
    messages: list[ChatMessage],
    protocol_handler: BaseProtocolHandler,
    extra_headers: dict[str, str] | None = None,
    storage_adapter: BaseStorageAdapter | None = None,
) -> StreamingHttpResponse:
    """
    Generic streaming chat view that works with any pipeline adapter and protocol handler.

    Args:
        adapter: Pipeline adapter instance or async factory function
        messages: List of chat messages to process
        protocol_handler: Protocol handler instance for formatting output
        extra_headers: Optional additional headers to include in response
        storage_adapter: Stores an errored reply if the adapter factory itself fails

    Returns:
        StreamingHttpResponse with SSE-formatted AI responses
    """
    logger.debug(
        f"Stream response initiated: adapter={type(adapter).__name__ if not callable(adapter) else 'factory'}, messages={len(messages)}, protocol={type(protocol_handler).__name__}"
    )

    sse_stream = _ensure_adapter(adapter, messages, protocol_handler, storage_adapter)

    # Build streaming HTTP response
    response = StreamingHttpResponse(  # type: ignore[arg-type]
        sse_stream,
        content_type="text/event-stream",
    )

    # Default SSE headers
    response["Cache-Control"] = "no-cache"
    cors_origin = resolve_setting("AI_SDK_STREAM_CORS_ORIGIN")
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
