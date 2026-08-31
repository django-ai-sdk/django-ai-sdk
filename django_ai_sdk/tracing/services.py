from __future__ import annotations

from typing import TYPE_CHECKING

from asgiref.sync import async_to_sync

from django_ai_sdk.conversation.models import Message
from django_ai_sdk.logger import get_logger
from django_ai_sdk.storage.services import ThreadService
from django_ai_sdk.tracing.models import Trace
from django_ai_sdk.tracing.schemas import TokenUsage, TraceOut

if TYPE_CHECKING:
    import uuid

    from django_ai_sdk.tracing.managers import TraceQuerySet
    from django_ai_sdk.types import UserType

logger = get_logger(__name__)


class TraceService:
    """Service for reading spans recorded"""

    @classmethod
    async def thread_traces(
        cls,
        thread_id: str | uuid.UUID,
        *,
        user: UserType,
        message_id: str | uuid.UUID | None = None,
        operation_name: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[TraceOut]:
        """
        Spans recorded for one thread, newest first.

        Args:
            thread_id: Thread whose spans to return
            user: User for permission checking
            message_id: Narrow to a single message's run
            operation_name: Narrow to one Haystack operation
            limit: Maximum spans to return (default 100)
            offset: Number of spans to skip

        Returns:
            List of TraceOut

        Raises:
            PermissionDenied: If user has no VIEW_THREAD permission for the thread
            ValueError: If the thread does not exist
        """
        await cls.get_permissions(thread_id, user=user)
        qs = Trace.objects.for_thread(thread_id)
        if message_id:
            qs = qs.for_message(message_id)
        return await cls._get_page(qs, operation_name, limit, offset)

    @classmethod
    async def message_traces(
        cls,
        message_id: str | uuid.UUID,
        *,
        user: UserType,
        operation_name: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[TraceOut]:
        """
        Spans recorded for one message's run, newest first.

        The owning thread is resolved from the message, so a caller holding only
        a message id does not need to know its thread.

        Args:
            message_id: Message whose spans to return
            user: User for permission checking
            operation_name: Narrow to one Haystack operation
            limit: Maximum spans to return (default 100)
            offset: Number of spans to skip

        Returns:
            List of TraceOut

        Raises:
            PermissionDenied: If user has no VIEW_THREAD permission for the thread
            ValueError: If the message does not exist
        """
        await cls.get_permissions(await cls._thread_message(message_id), user=user)
        qs = Trace.objects.for_message(message_id)
        return await cls._get_page(qs, operation_name, limit, offset)

    @classmethod
    async def thread_token_usage(cls, thread_id: str | uuid.UUID, *, user: UserType) -> TokenUsage:
        """
        Token totals across every run in one thread.

        Args:
            thread_id: Thread to total
            user: User for permission checking

        Returns:
            TokenUsage with prompt, completion and overall totals

        Raises:
            PermissionDenied: If user has no VIEW_THREAD permission for the thread
            ValueError: If the thread does not exist
        """
        await cls.get_permissions(thread_id, user=user)
        return TokenUsage(**await Trace.objects.for_thread(thread_id).atoken_usage())

    @classmethod
    async def message_token_usage(
        cls, message_id: str | uuid.UUID, *, user: UserType
    ) -> TokenUsage:
        """
        Token totals for one message's run.

        Args:
            message_id: Message to total
            user: User for permission checking

        Returns:
            TokenUsage with prompt, completion and overall totals

        Raises:
            PermissionDenied: If user has no VIEW_THREAD permission for the thread
            ValueError: If the message does not exist
        """
        await cls.get_permissions(await cls._thread_message(message_id), user=user)
        return TokenUsage(**await Trace.objects.for_message(message_id).atoken_usage())

    @classmethod
    async def get_permissions(cls, thread_id: str | uuid.UUID, *, user: UserType) -> None:
        """Enforce the thread's view permission"""
        if await ThreadService.get_thread(str(thread_id), user=user) is None:
            raise ValueError("Thread not found")

    @classmethod
    async def _thread_message(cls, message_id: str | uuid.UUID) -> uuid.UUID:
        """Resolve which thread a message belongs to."""
        thread_id = (
            await Message.objects.filter(id=message_id)
            .values_list(
                "thread_id",
                flat=True,
            )
            .afirst()
        )
        if thread_id is None:
            raise ValueError("Message not found")
        return thread_id

    @classmethod
    async def _get_page(
        cls,
        qs: TraceQuerySet,
        operation_name: str | None,
        limit: int,
        offset: int,
    ) -> list[TraceOut]:
        """Apply the shared operation filter, ordering and slice."""
        if operation_name:
            qs = qs.filter(operation_name=operation_name)
        return [
            TraceOut.model_validate(row)
            async for row in qs.order_by("-started_at")[offset : offset + limit]
        ]


# ============================================================================
# Sync wrappers for use in sync contexts
# ============================================================================

thread_traces = async_to_sync(TraceService.thread_traces)
message_traces = async_to_sync(TraceService.message_traces)
thread_token_usage = async_to_sync(TraceService.thread_token_usage)
message_token_usage = async_to_sync(TraceService.message_token_usage)
