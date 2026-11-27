from __future__ import annotations

from typing import TYPE_CHECKING

from asgiref.sync import async_to_sync

from django_ai_sdk.conversation.models import Message
from django_ai_sdk.errors import NotFound
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
    @classmethod
    async def thread_traces(
        cls,
        thread_id: str | uuid.UUID,
        *,
        user: UserType,
        message_id: str | uuid.UUID | None = None,
        operation_name: str | None = None,
        limit: int | None = 100,
        offset: int = 0,
    ) -> list[TraceOut]:
        """Spans recorded for one thread, newest first.
        Raises PermissionDenied without VIEW_THREAD, ValueError if the thread does not exist."""
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
        limit: int | None = 100,
        offset: int = 0,
    ) -> list[TraceOut]:
        """Spans recorded for one message's run, newest first.
        Raises PermissionDenied without VIEW_THREAD, ValueError if the message does not exist."""
        await cls.get_permissions(await cls._thread_message(message_id), user=user)
        qs = Trace.objects.for_message(message_id)
        return await cls._get_page(qs, operation_name, limit, offset)

    @classmethod
    async def thread_token_usage(cls, thread_id: str | uuid.UUID, *, user: UserType) -> TokenUsage:
        """Raises PermissionDenied without VIEW_THREAD, ValueError if the thread does not exist."""
        await cls.get_permissions(thread_id, user=user)
        return await cls._usage(Trace.objects.for_thread(thread_id))

    @classmethod
    async def message_token_usage(
        cls, message_id: str | uuid.UUID, *, user: UserType
    ) -> TokenUsage:
        """Raises PermissionDenied without VIEW_THREAD, ValueError if the message does not exist."""
        await cls.get_permissions(await cls._thread_message(message_id), user=user)
        return await cls._usage(Trace.objects.for_message(message_id))

    @classmethod
    async def _usage(cls, qs: TraceQuerySet) -> TokenUsage:
        totals = await qs.atoken_usage()
        by_subagent = await qs.asubagent_usage()
        return TokenUsage(
            prompt_tokens=totals["prompt_tokens"],
            completion_tokens=totals["completion_tokens"],
            total_tokens=totals["total_tokens"],
            by_subagent={
                agent_id: TokenUsage(
                    agent_name=usage["agent_name"],
                    prompt_tokens=usage["prompt_tokens"],
                    completion_tokens=usage["completion_tokens"],
                    total_tokens=usage["total_tokens"],
                )
                for agent_id, usage in by_subagent.items()
            },
        )

    @classmethod
    async def get_permissions(cls, thread_id: str | uuid.UUID, *, user: UserType) -> None:
        if await ThreadService.get_thread(str(thread_id), user=user) is None:
            raise NotFound("Thread not found")

    @classmethod
    async def _thread_message(cls, message_id: str | uuid.UUID) -> uuid.UUID:
        thread_id = (
            await Message.objects.filter(id=message_id)
            .values_list(
                "thread_id",
                flat=True,
            )
            .afirst()
        )
        if thread_id is None:
            raise NotFound("Message not found")
        return thread_id

    @classmethod
    async def _get_page(
        cls,
        qs: TraceQuerySet,
        operation_name: str | None,
        limit: int | None,
        offset: int,
    ) -> list[TraceOut]:
        if operation_name:
            qs = qs.filter(operation_name=operation_name)
        return [
            TraceOut.model_validate(row)
            async for row in qs.order_by("-started_at")[
                offset : offset + limit if limit is not None else None
            ]
        ]


thread_traces = async_to_sync(TraceService.thread_traces)
message_traces = async_to_sync(TraceService.message_traces)
thread_token_usage = async_to_sync(TraceService.thread_token_usage)
message_token_usage = async_to_sync(TraceService.message_token_usage)
