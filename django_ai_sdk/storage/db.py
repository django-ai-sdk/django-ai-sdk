from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from asgiref.sync import sync_to_async
from django.core.exceptions import ValidationError
from django.db import DatabaseError, models, transaction
from django.db.models import Count
from django.utils import timezone

from django_ai_sdk.common import THREAD_TITLE_MAX_LENGTH
from django_ai_sdk.conversation.models import Message, MessageFeedback, Thread
from django_ai_sdk.errors import NotFound
from django_ai_sdk.logger import get_logger
from django_ai_sdk.storage.base import (
    BaseStorageAdapter,
    StorageAdapterRegistry,
    StorageType,
)
from django_ai_sdk.storage.schemas import ThreadInfo

if TYPE_CHECKING:
    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser

    from django_ai_sdk.common import ChatMessage

logger = get_logger(__name__)


class DbStorageAdapter(BaseStorageAdapter):
    def __init__(self, thread_id: str) -> None:
        super().__init__(thread_id)
        self._thread: Thread | None = None

    @classmethod
    async def create_thread(
        cls,
        title: str,
        metadata: dict | None = None,
        user: AbstractBaseUser | AnonymousUser | None = None,
        thread_id: str | None = None,
    ) -> str:
        thread_id = thread_id or str(uuid.uuid4())
        user_id = str(user.pk) if user and user.is_authenticated else None

        thread = await Thread.objects.acreate(
            id=thread_id, title=title, metadata=metadata or {}, user_id=user_id
        )
        logger.debug(f"Created database thread: {thread_id}")
        return str(thread.id)

    @classmethod
    async def get_thread(cls, thread_id: str) -> ThreadInfo | None:
        try:
            thread = await Thread.objects.annotate(
                msg_count=Count("messages", filter=models.Q(messages__is_deleted=False))
            ).aget(id=thread_id)

            return ThreadInfo(
                id=str(thread.id),
                title=thread.title,
                agent_id=thread.metadata.get("agent_id", ""),
                model=thread.metadata.get("model", ""),
                user_id=str(thread.user_id) if thread.user_id else None,
                created_at=thread.created_at,
                updated_at=thread.updated_at,
                metadata=thread.metadata,
                message_count=thread.msg_count,
                file_memory_id=str(thread.file_memory_id) if thread.file_memory_id else None,
            )
        except (Thread.DoesNotExist, ValidationError):
            return None

    @classmethod
    async def list_threads(
        cls,
        user: AbstractBaseUser | AnonymousUser | None = None,
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[ThreadInfo]:
        queryset = Thread.objects.all()
        user_id = str(user.pk) if user and user.is_authenticated else None
        if user_id:
            queryset = queryset.filter(user_id=user_id)

        queryset = queryset.annotate(
            msg_count=Count("messages", filter=models.Q(messages__is_deleted=False))
        ).order_by("-updated_at")

        if limit is not None:
            queryset = queryset[offset : offset + limit]
        elif offset:
            queryset = queryset[offset:]

        threads = []
        async for thread in queryset:
            threads.append(
                ThreadInfo(
                    id=str(thread.id),
                    title=thread.title,
                    agent_id=thread.metadata.get("agent_id", ""),
                    model=thread.metadata.get("model", ""),
                    user_id=str(thread.user_id) if thread.user_id else None,
                    created_at=thread.created_at,
                    updated_at=thread.updated_at,
                    metadata=thread.metadata,
                    message_count=thread.msg_count,
                    file_memory_id=str(thread.file_memory_id) if thread.file_memory_id else None,
                )
            )
        return threads

    @classmethod
    async def update_thread(
        cls, thread_id: str, title: str | None = None, metadata: dict | None = None
    ) -> bool:
        try:
            await sync_to_async(cls._update_thread_locked)(thread_id, title, metadata)
            return True
        except (Thread.DoesNotExist, ValidationError):
            return False

    @staticmethod
    def _update_thread_locked(thread_id: str, title: str | None, metadata: dict | None) -> None:
        with transaction.atomic():
            thread = Thread.objects.select_for_update().get(id=thread_id)
            if title is not None:
                thread.title = title[:THREAD_TITLE_MAX_LENGTH]
            if metadata is not None:
                thread.metadata.update(metadata)
            thread.save(update_fields=["title", "metadata", "updated_at"])

    @classmethod
    async def delete_thread(cls, thread_id: str) -> bool:
        try:
            thread = await Thread.objects.aget(id=thread_id)
            await thread.messages.all().adelete()
            await thread.adelete()
            return True
        except (Thread.DoesNotExist, ValidationError):
            return False

    async def load_thread(self) -> Thread:
        if self._thread is None:
            logger.debug(f"Loading thread from database: {self.thread_id}")
            try:
                self._thread = await Thread.objects.aget(id=self.thread_id)
                logger.debug(f"Thread loaded successfully: {self._thread.title or 'Untitled'}")
            except Thread.DoesNotExist:
                logger.error(f"Thread not found in database: {self.thread_id}")
                raise NotFound(f"Thread with id {self.thread_id} not found")
        return self._thread

    async def get_messages(self) -> list[ChatMessage]:
        logger.debug(f"Fetching conversation history from database: {self.thread_id}")
        thread = await self.load_thread()

        messages_list = []
        qs = thread.messages.filter(is_deleted=False).order_by("created_at")
        async for msg in qs:
            messages_list.append(msg)

        message_ids = [msg.id for msg in messages_list]
        all_feedbacks = {}
        if message_ids:
            async for fb in MessageFeedback.objects.filter(message_id__in=message_ids):
                if fb.message_id not in all_feedbacks:
                    all_feedbacks[fb.message_id] = []
                all_feedbacks[fb.message_id].append(
                    {
                        "id": str(fb.id),
                        "user_id": str(fb.user_id) if fb.user_id else None,
                        "rating": fb.rating,
                        "feedback": fb.feedback,
                        "created_at": fb.created_at.isoformat() if fb.created_at else None,
                    }
                )

        messages = []
        for msg in messages_list:
            chat_message = msg.to_chat_message()
            chat_message.id = str(msg.id)
            chat_message.metadata["feedbacks"] = all_feedbacks.get(msg.id, [])
            messages.append(chat_message)

        logger.debug(f"Retrieved {len(messages)} messages with feedbacks from database")
        return messages

    async def store_chat_message(self, chat_message: ChatMessage) -> str:
        logger.debug(
            f"Direct message storage: role={chat_message.role}, content_length={len(chat_message.content)}"
        )
        thread = await self.load_thread()
        message = Message.from_chat_message(thread, chat_message)
        await message.asave()
        logger.debug(f"Message saved directly with ID: {message.id}")
        return str(message.id)

    async def storage_callback(self, chat_message: ChatMessage) -> str | None:
        logger.debug(
            f"Storing message via callback: role={chat_message.role}, content_length={len(chat_message.content)}, tool_calls={len(chat_message.tool_calls)}"
        )
        try:
            thread = await self.load_thread()
            logger.debug(f"Thread loaded: id={thread.id}, title={thread.title}")

            message = Message.from_chat_message(thread, chat_message)
            logger.debug(f"Creating message: thread_id={message.thread_id}, thread.id={thread.id}")

            await message.asave()
            logger.debug(f"Message saved: id={message.id}, thread={message.thread_id}")
            return str(message.id)
        except DatabaseError:
            logger.exception(
                f"Database storage failed for thread {self.thread_id}. Content length: {len(chat_message.content) if chat_message.content else 0}"
            )
            return None

    async def rate_message(
        self,
        message_id: str,
        rating: int | None,
        feedback: str = "",
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> bool:
        from django_ai_sdk.conversation.models import MessageFeedback

        user = user if (user and user.is_authenticated) else None
        try:
            await Message.objects.aget(id=message_id, thread_id=self.thread_id)
            if rating is not None:
                await MessageFeedback.objects.aupdate_or_create(
                    message_id=message_id,
                    user=user,
                    defaults={"rating": rating, "feedback": feedback},
                )
                logger.debug(f"Saved feedback for message {message_id}: rating={rating}")
            else:
                await MessageFeedback.objects.filter(message_id=message_id, user=user).adelete()
                logger.debug(f"Deleted feedback for message {message_id}")
            return True
        except Message.DoesNotExist:
            return False

    async def delete_message(self, message_id: str) -> bool:
        try:
            message = await Message.objects.aget(id=message_id, thread_id=self.thread_id)
            message.is_deleted = True
            message.deleted_at = timezone.now()
            await message.asave()
            logger.debug(f"Soft deleted message {message_id}")
            return True
        except Message.DoesNotExist:
            return False

    async def restore_message(self, message_id: str) -> bool:
        try:
            message = await Message.objects.aget(id=message_id, thread_id=self.thread_id)
            message.is_deleted = False
            message.deleted_at = None
            await message.asave()
            logger.debug(f"Restored message {message_id}")
            return True
        except Message.DoesNotExist:
            return False


StorageAdapterRegistry.register(DbStorageAdapter, StorageType.DATABASE)
