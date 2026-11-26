from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from django.conf import settings
from django.db import models
from django.utils import timezone

from django_ai_sdk.common import THREAD_TITLE_MAX_LENGTH, ChatMessage

from .managers import MessageManager, ThreadManager

if TYPE_CHECKING:
    from django.db.models.fields.related_descriptors import RelatedManager


class Thread(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=THREAD_TITLE_MAX_LENGTH, blank=True, default="")
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="threads",
        null=True,
        blank=True,
    )
    file_memory = models.ForeignKey(
        "django_ai_sdk.Memory",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="thread_files",
    )
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)
    metadata = models.JSONField(default=dict, blank=True)

    messages: RelatedManager[Message]
    user_id: str | None
    file_memory_id: str | None

    msg_count: int

    objects = ThreadManager()

    class Meta:
        db_table = "django_ai_sdk_threads"
        ordering = ["-updated_at"]
        indexes = [
            models.Index(fields=["user"]),
            models.Index(fields=["-updated_at"]),
            models.Index(fields=["-created_at"]),
        ]

    def __str__(self) -> str:
        return self.title or f"Thread {self.id}"

    @property
    def message_count(self) -> int:
        return self.messages.count()

    @property
    def last_message(self) -> Message | None:
        return self.messages.order_by("-created_at").first()

    def add_user_message(self, chat_message: ChatMessage) -> Message:
        message = Message.from_chat_message(self, chat_message)
        message.save()
        return message


class Message(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    thread = models.ForeignKey(Thread, on_delete=models.CASCADE, related_name="messages")
    created_at = models.DateTimeField(default=timezone.now)

    thread_id: str

    result = models.JSONField()

    is_deleted = models.BooleanField(default=False)
    deleted_at = models.DateTimeField(null=True, blank=True)

    objects = MessageManager()

    class Meta:
        db_table = "django_ai_sdk_messages"
        ordering = ["created_at"]
        indexes = [
            models.Index(fields=["thread", "created_at"]),
            models.Index(fields=["is_deleted"]),
            models.Index(fields=["thread", "created_at", "is_deleted"]),
        ]

    def __str__(self) -> str:
        chat_msg = self.to_chat_message()
        content_preview = (
            chat_msg.content[:50] + "..." if len(chat_msg.content) > 50 else chat_msg.content
        )
        return f"{chat_msg.role}: {content_preview}"

    async def asave(self, *args: Any, **kwargs: Any) -> None:
        await super().asave(*args, **kwargs)
        await Thread.objects.filter(id=self.thread_id).aupdate(updated_at=timezone.now())

    def to_chat_message(self) -> ChatMessage:
        chat_message = ChatMessage.model_validate(self.result)
        chat_message.created_at = self.created_at.isoformat() if self.created_at else ""
        return chat_message

    @classmethod
    def from_chat_message(cls, thread: Thread, chat_message: ChatMessage) -> Message:
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

    def delete_message(self) -> Message:
        self.is_deleted = True
        self.deleted_at = timezone.now()
        return self

    def restore_message(self) -> Message:
        self.is_deleted = False
        self.deleted_at = None
        return self


class MessageFeedback(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    message = models.ForeignKey(Message, on_delete=models.CASCADE, related_name="feedbacks")
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="message_feedbacks",
    )
    rating = models.SmallIntegerField(choices=[(1, "good"), (-1, "bad")])
    feedback = models.CharField(max_length=255, blank=True, default="")
    created_at = models.DateTimeField(default=timezone.now)

    message_id: str
    user_id: str | None

    class Meta:
        db_table = "django_ai_sdk_message_feedbacks"
        unique_together = [("message", "user")]

    def __str__(self) -> str:
        return f"Feedback on message {self.message_id}: {self.rating}"
