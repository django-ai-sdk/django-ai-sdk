import traceback
import uuid

from django.utils import timezone

from django_ai_sdk.common import ChatMessage
from django_ai_sdk.conversation.models import Message, Thread
from django_ai_sdk.logger import get_logger
from django_ai_sdk.storage.base import (
    BaseStorageAdapter,
    StorageAdapterRegistry,
    StorageType,
)
from django_ai_sdk.storage.schemas import ThreadInfo

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
        user_id: str | None = None,
        thread_id: str | None = None,
    ) -> str:
        """
        Create a new thread in the database.

        Args:
            title: Thread title
            metadata: Should include assistant_id, model
            user_id: Optional user ID
            thread_id: Optional custom thread ID

        Returns:
            Thread ID (UUID string)
        """
        # Use provided thread_id or generate new UUID
        thread_id = thread_id or str(uuid.uuid4())

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
            thread = await Thread.objects.aget(id=thread_id)
            # Get message count (excluding deleted)
            message_count = await thread.messages.filter(is_deleted=False).acount()

            return ThreadInfo(
                id=str(thread.id),
                title=thread.title,
                assistant_id=thread.metadata.get("assistant_id", ""),
                model=thread.metadata.get("model", ""),
                user_id=str(thread.user_id) if thread.user_id else None,
                created_at=thread.created_at,
                updated_at=thread.updated_at,
                metadata=thread.metadata,
                message_count=message_count,
            )
        except Thread.DoesNotExist:
            return None

    @classmethod
    async def list_threads(cls, user_id: str | None = None) -> list[ThreadInfo]:
        """List all threads from database."""
        queryset = Thread.objects.all()
        if user_id:
            queryset = queryset.filter(user_id=user_id)

        threads = []
        async for thread in queryset.order_by("-updated_at"):
            message_count = await thread.messages.filter(is_deleted=False).acount()
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
                    message_count=message_count,
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
        except Thread.DoesNotExist:
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
        except Thread.DoesNotExist:
            return False

    @classmethod
    async def delete_all_threads(cls) -> int:
        """Delete all threads and their messages from database."""
        from django_ai_sdk.conversation.models import Message

        # Get count before deleting
        count = await Thread.objects.acount() or 0
        # Delete all messages first
        await Message.objects.all().adelete()
        # Delete all threads
        await Thread.objects.all().adelete()
        return count

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
        Excludes deleted messages.

        Returns:
            List of ChatMessage objects ordered by creation time
        """
        logger.debug(f"Fetching conversation history from database: {self.thread_id}")
        thread = await self.load_thread()
        messages = []
        # Filter out deleted messages
        async for msg in thread.messages.filter(is_deleted=False).order_by("created_at"):
            chat_message = msg.to_chat_message()
            chat_message.id = str(msg.id)
            # Include rating metadata for protocol conversion
            chat_message.metadata["rating"] = msg.rating
            chat_message.metadata["rating_comment"] = msg.rating_comment
            messages.append(chat_message)
        logger.debug(f"Retrieved {len(messages)} messages from database")
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
        except Exception as database_error:
            logger.error(
                f"Database storage failed: {database_error}\nThread ID: {self.thread_id}\nMessage thread_id: {message.thread_id if 'message' in locals() else 'N/A'}\nMessage content: {chat_message.content[:200] if chat_message.content else 'None'}...\nStack trace:\n{traceback.format_exc()}"
            )
            return None

    async def rate_message(
        self, message_id: str, rating: int | None, rating_comment: str = ""
    ) -> bool:
        """Rate a message in this thread."""
        try:
            message = await Message.objects.aget(id=message_id, thread_id=self.thread_id)
            message.rating = rating
            message.rating_comment = rating_comment
            await message.asave()
            logger.debug(
                f"Rated message {message_id}: rating={rating}, comment={rating_comment[:50] if rating_comment else 'empty'}"
            )
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
