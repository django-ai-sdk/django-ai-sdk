from __future__ import annotations

from typing import Any

from django.db.models.signals import post_delete
from django.dispatch import receiver

from django_ai_sdk.conversation.models import MessageImage


@receiver(post_delete, sender=MessageImage)
def delete_message_image_file(
    sender: type[MessageImage], instance: MessageImage, **kwargs: Any
) -> None:
    """Remove the stored blob when its MessageImage row is deleted.

    FileField does not delete the underlying file on row deletion; without this
    the storage backend would accumulate orphaned images as messages/threads are
    hard-deleted (soft-deleted messages keep their files, by design).
    """
    if instance.file:
        instance.file.delete(save=False)
