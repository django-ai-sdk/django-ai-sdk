from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel, Field

from django_ai_sdk.common import ChatMessage
from django_ai_sdk.logger import get_logger
from django_ai_sdk.storage.base import (
    BaseStorageAdapter,
    StorageAdapterRegistry,
    StorageType,
)
from django_ai_sdk.storage.schemas import ThreadInfo

if TYPE_CHECKING:
    from django.contrib.auth.models import AbstractUser

logger = get_logger(__name__)


class MemoryMessage(BaseModel):
    """
    In-memory representation of a conversation message.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    thread_id: str
    result: dict
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Message management fields
    feedbacks: list[dict] = Field(default_factory=list)  # List of {user_id, rating, feedback}
    is_deleted: bool = False
    deleted_at: datetime | None = None

    def to_chat_message(self) -> ChatMessage:
        """Convert stored data to ChatMessage object."""
        # ID is already in self.result from model_dump(), model_validate restores it
        chat_message = ChatMessage.model_validate(self.result)
        # Include feedbacks in metadata
        chat_message.metadata["feedbacks"] = self.feedbacks
        return chat_message

    @classmethod
    def from_chat_message(cls, thread_id: str, chat_message: ChatMessage) -> MemoryMessage:
        """Create MemoryMessage from a ChatMessage - ID must be provided by adapter."""
        return cls(
            id=chat_message.id,
            thread_id=thread_id,
            result=chat_message.model_dump(),
        )


class MemoryThread(BaseModel):
    """In-memory representation of a conversation thread."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    assistant_id: str = ""
    model: str = ""
    user_id: str | None = None
    metadata: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class MemoryStore:
    """
    Singleton in-memory store for threads and messages.
    """

    threads: ClassVar[dict[str, MemoryThread]] = {}
    messages: ClassVar[dict[str, list[MemoryMessage]]] = {}

    # ============================================================================
    # Thread Operations
    # ============================================================================

    @classmethod
    def create_thread(
        cls,
        thread_id: str,
        title: str = "",
        assistant_id: str = "",
        model: str = "",
        user_id: str | None = None,
        metadata: dict | None = None,
    ) -> MemoryThread:
        """Create and store a new thread."""
        thread = MemoryThread(
            id=thread_id,
            title=title,
            assistant_id=assistant_id,
            model=model,
            user_id=user_id,
            metadata=metadata or {},
        )
        cls.threads[thread_id] = thread
        cls.messages.setdefault(thread_id, [])
        return thread

    @classmethod
    def get_thread(cls, thread_id: str) -> MemoryThread | None:
        """Get thread by ID."""
        return cls.threads.get(thread_id)

    @classmethod
    def list_threads(cls, user_id: str | None = None) -> list[MemoryThread]:
        """List all threads, optionally filtered by user."""
        threads = list(cls.threads.values())
        if user_id:
            threads = [t for t in threads if t.user_id == user_id]
        return threads

    @classmethod
    def update_thread(
        cls, thread_id: str, title: str | None = None, metadata: dict | None = None
    ) -> MemoryThread | None:
        """Update thread metadata."""
        thread = cls.threads.get(thread_id)
        if not thread:
            return None

        if title is not None:
            thread.title = title
        if metadata is not None:
            # Merge metadata
            thread.metadata.update(metadata)

        thread.updated_at = datetime.now(UTC)
        return thread

    @classmethod
    def delete_thread(cls, thread_id: str) -> bool:
        """Delete thread and all its messages."""
        if thread_id in cls.threads:
            del cls.threads[thread_id]
            if thread_id in cls.messages:
                del cls.messages[thread_id]
            return True
        return False

    # ============================================================================
    # Message Operations
    # ============================================================================

    @classmethod
    def add_message(cls, thread_id: str, message: MemoryMessage) -> None:
        """Add message to thread."""
        cls.messages.setdefault(thread_id, [])
        cls.messages[thread_id].append(message)
        # Update thread timestamp, is needed to handle ordering
        thread = cls.threads.get(thread_id)
        if thread:
            thread.updated_at = datetime.now(UTC)

    @classmethod
    def get_messages(cls, thread_id: str, include_deleted: bool = False) -> list[MemoryMessage]:
        """Get messages for thread, optionally including deleted ones."""
        messages = cls.messages.get(thread_id, [])

        # Filter out soft-deleted messages
        if not include_deleted:
            messages = [m for m in messages if not m.is_deleted]
        return messages

    @classmethod
    def get_message(cls, message_id: str) -> MemoryMessage | None:
        """Find message by ID across all threads."""
        for messages in cls.messages.values():
            for msg in messages:
                if msg.id == message_id:
                    return msg
        return None

    @classmethod
    def rate_message(
        cls, message_id: str, rating: int | None, feedback: str = "", user_id: str | None = None
    ) -> bool:
        """Rate a message."""
        message = cls.get_message(message_id)
        if message:
            if rating is not None:
                # Update or create feedback for this user
                existing_feedback = None
                for fb in message.feedbacks:
                    if fb.get("user_id") == user_id:
                        existing_feedback = fb
                        break

                if existing_feedback:
                    existing_feedback["rating"] = rating
                    existing_feedback["feedback"] = feedback
                else:
                    message.feedbacks.append(
                        {"user_id": user_id, "rating": rating, "feedback": feedback}
                    )
            else:
                # Delete feedback when rating is None
                message.feedbacks[:] = [
                    fb for fb in message.feedbacks if fb.get("user_id") != user_id
                ]
            return True
        return False

    @classmethod
    def delete_message(cls, message_id: str) -> bool:
        """Soft delete a message."""
        message = cls.get_message(message_id)
        if message:
            message.is_deleted = True
            message.deleted_at = datetime.now(UTC)
            return True
        return False

    @classmethod
    def restore_message(cls, message_id: str) -> bool:
        """Restore a soft-deleted message."""
        message = cls.get_message(message_id)
        if message:
            message.is_deleted = False
            message.deleted_at = None
            return True
        return False

    @classmethod
    def clear(cls) -> None:
        """Clear all stored threads and messages."""
        cls.threads.clear()
        cls.messages.clear()


class MemoryStorageAdapter(BaseStorageAdapter):
    """
    Store conversations in in-memory Pydantic models.
    """

    def __init__(self, thread_id: str) -> None:
        """
        Initialize memory storage for a specific thread.

        Args:
            thread_id: Unique identifier for the conversation thread
        """
        super().__init__(thread_id)
        self._thread: MemoryThread | None = None

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
        Create a new thread in memory.

        Args:
            title: Thread title
            metadata: Should include assistant_id, model
            user: Optional user
            thread_id: Optional custom thread ID

        Returns:
            Thread ID (UUID string)
        """
        thread_id = thread_id or str(uuid.uuid4())
        assistant_id = metadata.get("assistant_id", "") if metadata else ""
        model = metadata.get("model", "") if metadata else ""
        user_id = str(user.pk) if user and user.is_authenticated else None

        MemoryStore.create_thread(
            thread_id=thread_id,
            title=title,
            assistant_id=assistant_id,
            model=model,
            user_id=user_id,
            metadata=metadata or {},
        )
        logger.debug(f"Created memory thread: {thread_id}")
        return thread_id

    @classmethod
    async def get_thread(cls, thread_id: str) -> ThreadInfo | None:
        """Get thread metadata by ID."""
        thread = MemoryStore.get_thread(thread_id)
        if not thread:
            return None

        messages = MemoryStore.get_messages(thread_id, include_deleted=False)
        return ThreadInfo(
            id=thread.id,
            title=thread.title,
            assistant_id=thread.assistant_id,
            model=thread.model,
            user_id=thread.user_id,
            created_at=thread.created_at,
            updated_at=thread.updated_at,
            metadata=thread.metadata,
            message_count=len(messages),
        )

    @classmethod
    async def list_threads(cls, user: AbstractUser | None = None) -> list[ThreadInfo]:
        """List all threads in memory."""
        user_id = str(user.pk) if user and user.is_authenticated else None
        threads = MemoryStore.list_threads(user_id)
        result = []
        for thread in threads:
            messages = MemoryStore.get_messages(thread.id, include_deleted=False)
            result.append(
                ThreadInfo(
                    id=thread.id,
                    title=thread.title,
                    assistant_id=thread.assistant_id,
                    model=thread.model,
                    user_id=thread.user_id,
                    created_at=thread.created_at,
                    updated_at=thread.updated_at,
                    metadata=thread.metadata,
                    message_count=len(messages),
                )
            )
        return result

    @classmethod
    async def update_thread(
        cls, thread_id: str, title: str | None = None, metadata: dict | None = None
    ) -> bool:
        """Update thread metadata."""
        thread = MemoryStore.update_thread(thread_id, title, metadata)
        return thread is not None

    @classmethod
    async def delete_thread(cls, thread_id: str) -> bool:
        """Delete thread and all its messages."""
        return MemoryStore.delete_thread(thread_id)

    # ============================================================================
    # INSTANCE METHODS - Thread-Specific Operations
    # ============================================================================

    async def load_thread(self) -> MemoryThread:
        """Lazy-load the MemoryThread. Raises error if thread doesn't exist."""
        if self._thread is None:
            logger.debug(f"Loading thread from memory store: {self.thread_id}")
            self._thread = MemoryStore.get_thread(self.thread_id)
            if self._thread is None:
                raise ValueError(
                    f"Thread {self.thread_id} not found in memory store. Thread must be explicitly created before storing messages."
                )
            logger.debug(f"Thread ready: {self._thread.title or 'Untitled'}")
        return self._thread

    async def get_messages(self) -> list[ChatMessage]:
        """Retrieve all ChatMessages for this thread (excluding deleted)."""
        logger.debug(f"Fetching conversation history from memory: {self.thread_id}")
        await self.load_thread()
        messages = MemoryStore.get_messages(self.thread_id, include_deleted=False)
        chat_messages = [msg.to_chat_message() for msg in messages]
        logger.debug(f"Retrieved {len(chat_messages)} messages from memory")
        return chat_messages

    async def store_chat_message(self, chat_message: ChatMessage) -> str:
        """Store ChatMessage to this thread."""
        logger.debug(
            f"Direct message storage: role={chat_message.role}, content_length={len(chat_message.content)}"
        )
        await self.load_thread()
        message = MemoryMessage.from_chat_message(self.thread_id, chat_message)
        MemoryStore.add_message(self.thread_id, message)
        logger.debug(f"Message saved directly with ID: {message.id}")
        return message.id

    async def storage_callback(self, chat_message: ChatMessage) -> str | None:
        """Callback for StreamWriter to auto-store messages."""
        logger.debug(
            f"Storing message via callback: role={chat_message.role}, content_length={len(chat_message.content)}"
        )
        try:
            await self.load_thread()
            message = MemoryMessage.from_chat_message(self.thread_id, chat_message)
            MemoryStore.add_message(self.thread_id, message)
            logger.debug(f"Message saved to memory store with ID: {message.id}")
            return message.id
        except (ValueError, KeyError, RuntimeError) as error:
            logger.error(f"Memory storage failed: {error}")
            return None

    async def rate_message(
        self,
        message_id: str,
        rating: int | None,
        feedback: str = "",
        user: AbstractUser | None = None,
    ) -> bool:
        """Rate a message in this thread."""
        user_id = str(user.pk) if user and user.is_authenticated else None
        success = MemoryStore.rate_message(message_id, rating, feedback, user_id)
        if success:
            logger.debug(f"Rated message {message_id}: {rating}")
        return success

    async def delete_message(self, message_id: str) -> bool:
        """Soft delete a message in this thread."""
        success = MemoryStore.delete_message(message_id)
        if success:
            logger.debug(f"Soft deleted message {message_id}")
        return success

    async def restore_message(self, message_id: str) -> bool:
        """Restore a soft-deleted message."""
        success = MemoryStore.restore_message(message_id)
        if success:
            logger.debug(f"Restored message {message_id}")
        return success


# Register with StorageAdapterRegistry
StorageAdapterRegistry.register(MemoryStorageAdapter, StorageType.MEMORY)
