import uuid
from typing import Any

from django.contrib.auth.models import User
from django.db import models
from django.utils import timezone

from django_ai_sdk.common import ChatMessage

from .managers import MessageManager, ThreadManager


class Thread(models.Model):
    """
    Represents a conversation thread that can contain multiple messages.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=255, blank=True, default="")
    user = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="ai_threads", null=True, blank=True
    )
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)
    metadata = models.JSONField(default=dict, blank=True)

    # Reverse relation type hint
    messages: models.Manager["Message"]
    user_id: str | None

    # Custom manager
    objects = ThreadManager()

    class Meta:
        db_table = "django_ai_sdk_threads"
        ordering = ["-updated_at"]

    def __str__(self) -> str:
        return self.title or f"Thread {self.id}"

    @property
    def message_count(self) -> int:
        return self.messages.count()

    @property
    def last_message(self) -> "Message | None":
        return self.messages.order_by("-created_at").first()

    def add_user_message(self, chat_message: ChatMessage) -> "Message":
        """Add user ChatMessage to this thread."""
        message = Message.from_chat_message(self, chat_message)
        message.save()
        return message


class Message(models.Model):
    """
    Represents a single message within a conversation thread.
    Stores complete ChatMessage as JSON for unified storage.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    thread = models.ForeignKey(Thread, on_delete=models.CASCADE, related_name="messages")
    created_at = models.DateTimeField(default=timezone.now)

    # Type hints for related fields
    thread_id: str

    # Store complete ChatMessage as JSON
    result = models.JSONField()

    # User feedback
    rating = models.SmallIntegerField(
        null=True,
        blank=True,
        choices=[(1, "good"), (-1, "bad")],
    )

    # Soft delete
    is_deleted = models.BooleanField(default=False)
    deleted_at = models.DateTimeField(null=True, blank=True)

    # Custom manager
    objects = MessageManager()

    class Meta:
        db_table = "django_ai_sdk_messages"
        ordering = ["created_at"]
        indexes = [
            models.Index(fields=["thread", "created_at"]),
        ]

    def __str__(self) -> str:
        chat_msg = self.to_chat_message()
        content_preview = (
            chat_msg.content[:50] + "..." if len(chat_msg.content) > 50 else chat_msg.content
        )
        return f"{chat_msg.role}: {content_preview}"

    async def asave(self, *args: Any, **kwargs: Any) -> None:
        # Update thread's updated_at timestamp when a message is saved
        await super().asave(*args, **kwargs)
        # Use atomic update to avoid race conditions
        await Thread.objects.filter(id=self.thread_id).aupdate(updated_at=timezone.now())

    def to_chat_message(self) -> ChatMessage:
        """Convert stored JSON data to ChatMessage object."""
        return ChatMessage.model_validate(self.result)

    @classmethod
    def from_chat_message(cls, thread: Thread, chat_message: ChatMessage) -> "Message":
        """Create Message from ChatMessage - ID must be provided by adapter."""
        # Validate that ID is provided by adapter (contract enforcement)
        if not chat_message.id:
            raise ValueError(
                "ChatMessage.id is required but was not provided. ID must be generated at adapter level."
            )
        try:
            message_id = uuid.UUID(chat_message.id)
        except ValueError as e:
            raise ValueError(
                f"Invalid UUID in ChatMessage.id: {chat_message.id}. ID must be a valid UUID string."
            ) from e
        return cls(id=message_id, thread=thread, result=chat_message.model_dump())

    def rate(self, rating_value: int) -> "Message":
        """Rate this message as good (1) or bad (-1)."""
        self.rating = rating_value
        return self

    def delete_message(self) -> "Message":
        """Soft delete this message."""
        self.is_deleted = True
        self.deleted_at = timezone.now()
        return self

    def restore_message(self) -> "Message":
        """Restore a soft-deleted message."""
        self.is_deleted = False
        self.deleted_at = None
        return self
