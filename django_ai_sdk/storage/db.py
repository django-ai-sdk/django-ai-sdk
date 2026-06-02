from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from django.core.exceptions import ValidationError
from django.db import models
from django.db.models import Count
from django.utils import timezone

from django_ai_sdk.conversation.models import Message, MessageFeedback, Thread
from django_ai_sdk.logger import get_logger
from django_ai_sdk.storage.base import (
    BaseStorageAdapter,
    StorageAdapterRegistry,
    StorageType,
)
from django_ai_sdk.storage.schemas import ThreadInfo

if TYPE_CHECKING:
    from django.contrib.auth.models import AbstractUser

    from django_ai_sdk.common import ChatMessage

logger = get_logger(__name__)


class DbStorageAdapter(BaseStorageAdapter):
    """
    Store conversations in Django Thread/Message models.

    This adapter provides automatic storage of ChatMessages in the Django
    database using the Thread and Message models. It provides callbacks to
    StreamWriter for automatic storage after streaming completes.
    """

    def __init__(self, thread_id: str) -> None:
        """
        Initialize database storage for a specific thread.

        Args:
            thread_id: UUID string of the Thread model instance
        """
        super().__init__(thread_id)
        self._thread: Thread | None = None

    # ============================================================================
    # CLASS METHODS - Thread Management
    # ============================================================================

    @classmethod
    async def create_thread(
        cls,
        title: str,
        metadata: dict | None = None,
        user: AbstractUser | None = None,
        thread_id: str | None = None,
    ) -> str:
        """
        Create a new thread in the database.

        Args:
            title: Thread title
            metadata: Should include assistant_id, model
            user: Optional user
            thread_id: Optional custom thread ID

        Returns:
            Thread ID (UUID string)
        """
        # Use provided thread_id or generate new UUID
        thread_id = thread_id or str(uuid.uuid4())
        user_id = str(user.pk) if user and user.is_authenticated else None

        # Create in database
        thread = await Thread.objects.acreate(
            id=thread_id, title=title, metadata=metadata or {}, user_id=user_id
        )
        logger.debug(f"Created database thread: {thread_id}")
        return str(thread.id)

    @classmethod
    async def get_thread(cls, thread_id: str) -> ThreadInfo | None:
        """Get thread metadata by ID."""
        try:
            thread = await Thread.objects.annotate(
                msg_count=Count("messages", filter=models.Q(messages__is_deleted=False))
            ).aget(id=thread_id)

            return ThreadInfo(
                id=str(thread.id),
                title=thread.title,
                assistant_id=thread.metadata.get("assistant_id", ""),
                model=thread.metadata.get("model", ""),
                user_id=str(thread.user_id) if thread.user_id else None,
                created_at=thread.created_at,
                updated_at=thread.updated_at,
                metadata=thread.metadata,
                message_count=thread.msg_count,
            )
        except (Thread.DoesNotExist, ValidationError):
            return None

    @classmethod
    async def list_threads(cls, user: AbstractUser | None = None) -> list[ThreadInfo]:
        """List all threads from database."""
        queryset = Thread.objects.all()
        user_id = str(user.pk) if user and user.is_authenticated else None
        if user_id:
            queryset = queryset.filter(user_id=user_id)

        queryset = queryset.annotate(
            msg_count=Count("messages", filter=models.Q(messages__is_deleted=False))
        ).order_by("-updated_at")

        threads = []
        async for thread in queryset:
            threads.append(
                ThreadInfo(
                    id=str(thread.id),
                    title=thread.title,
                    assistant_id=thread.metadata.get("assistant_id", ""),
                    model=thread.metadata.get("model", ""),
                    user_id=str(thread.user_id) if thread.user_id else None,
                    created_at=thread.created_at,
                    updated_at=thread.updated_at,
                    metadata=thread.metadata,
                    message_count=thread.msg_count,
                )
            )
        return threads

    @classmethod
    async def update_thread(
        cls, thread_id: str, title: str | None = None, metadata: dict | None = None
    ) -> bool:
        """Update thread metadata."""
        try:
            thread = await Thread.objects.aget(id=thread_id)

            if title is not None:
                thread.title = title
            if metadata is not None:
                # Merge metadata
                thread.metadata.update(metadata)

            await thread.asave()
            return True
        except (Thread.DoesNotExist, ValidationError):
            return False

    @classmethod
    async def delete_thread(cls, thread_id: str) -> bool:
        """Delete thread and all its messages from database."""
        try:
            thread = await Thread.objects.aget(id=thread_id)
            # Delete all messages first (cascade or manually)
            await thread.messages.all().adelete()
            # Delete thread
            await thread.adelete()
            return True
        except (Thread.DoesNotExist, ValidationError):
            return False

    # ============================================================================
    # INSTANCE METHODS - Thread-Specific Operations
    # ============================================================================

    async def load_thread(self) -> Thread:
        """Lazy-load Thread model to avoid database queries during init."""
        if self._thread is None:
            logger.debug(f"Loading thread from database: {self.thread_id}")
            try:
                self._thread = await Thread.objects.aget(id=self.thread_id)
                logger.debug(f"Thread loaded successfully: {self._thread.title or 'Untitled'}")
            except Thread.DoesNotExist:
                logger.error(f"Thread not found in database: {self.thread_id}")
                raise ValueError(f"Thread with id {self.thread_id} not found")
        return self._thread

    async def get_messages(self) -> list[ChatMessage]:
        """
        Retrieve all ChatMessages for this thread from database.
        Excludes deleted messages. Includes all feedbacks in metadata.

        Returns:
            List of ChatMessage objects ordered by creation time
        """

        logger.debug(f"Fetching conversation history from database: {self.thread_id}")
        thread = await self.load_thread()

        # Get all messages for this thread
        messages_list = []
        async for msg in thread.messages.filter(is_deleted=False).order_by("created_at"):
            messages_list.append(msg)

        # Batch-fetch all feedbacks for these messages
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

        # Convert to ChatMessages with feedbacks in metadata
        messages = []
        for msg in messages_list:
            chat_message = msg.to_chat_message()
            chat_message.id = str(msg.id)
            chat_message.metadata["feedbacks"] = all_feedbacks.get(msg.id, [])
            messages.append(chat_message)

        logger.debug(f"Retrieved {len(messages)} messages with feedbacks from database")
        return messages

    async def store_chat_message(self, chat_message: ChatMessage) -> str:
        """
        Store ChatMessage in database.

        Args:
            chat_message: ChatMessage instance to store

        Returns:
            String UUID of the created Message instance
        """
        logger.debug(
            f"Direct message storage: role={chat_message.role}, content_length={len(chat_message.content)}"
        )
        thread = await self.load_thread()
        message = Message.from_chat_message(thread, chat_message)
        await message.asave()
        logger.debug(f"Message saved directly with ID: {message.id}")
        return str(message.id)

    async def storage_callback(self, chat_message: ChatMessage) -> str | None:
        """
        Store assistant ChatMessage in database when called by StreamWriter.finalize().
        """
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
        except Exception:
            logger.exception(
                f"Database storage failed for thread {self.thread_id}. Content length: {len(chat_message.content) if chat_message.content else 0}"
            )
            return None

    async def rate_message(
        self,
        message_id: str,
        rating: int | None,
        feedback: str = "",
        user: AbstractUser | None = None,
    ) -> bool:
        """Rate a message in this thread."""
        from django_ai_sdk.conversation.models import MessageFeedback

        try:
            # Verify message exists and belongs to this thread
            await Message.objects.aget(id=message_id, thread_id=self.thread_id)
            if rating is not None:
                # Try to get existing feedback
                try:
                    fb = await MessageFeedback.objects.aget(message_id=message_id, user=user)
                    # Update existing
                    fb.rating = rating
                    fb.feedback = feedback
                    await fb.asave(update_fields=["rating", "feedback"])
                    logger.debug(f"Updated feedback for message {message_id}: rating={rating}")
                except MessageFeedback.DoesNotExist:
                    # Create new
                    await MessageFeedback.objects.acreate(
                        message_id=message_id,
                        user=user,
                        rating=rating,
                        feedback=feedback,
                    )
                    logger.debug(f"Created feedback for message {message_id}: rating={rating}")
            else:
                # Delete feedback when rating is None
                await MessageFeedback.objects.filter(message_id=message_id, user=user).adelete()
                logger.debug(f"Deleted feedback for message {message_id}")
            return True
        except Message.DoesNotExist:
            return False

    async def delete_message(self, message_id: str) -> bool:
        """Soft delete a message in this thread."""
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
        """Restore a soft-deleted message."""
        try:
            message = await Message.objects.aget(id=message_id, thread_id=self.thread_id)
            message.is_deleted = False
            message.deleted_at = None
            await message.asave()
            logger.debug(f"Restored message {message_id}")
            return True
        except Message.DoesNotExist:
            return False


# Register with StorageAdapterRegistry
StorageAdapterRegistry.register(DbStorageAdapter, StorageType.DATABASE)
