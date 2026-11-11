"""Forget oversight content when a thread is deleted (the oversight guide's Forgetting
says what goes and what stays), and drop cached policies when AI_SDK_OVERSIGHT changes.
"""

from __future__ import annotations

from typing import Any

from django.core.signals import setting_changed
from django.db import transaction
from django.db.models.signals import post_delete

from django_ai_sdk.logger import get_logger

logger = get_logger(__name__)


def forget_deleted_thread(sender: Any, instance: Any, **kwargs: Any) -> None:
    """Forget a deleted thread's oversight content."""
    from django_ai_sdk.oversight import custody
    from django_ai_sdk.oversight.models import Turn, TurnContent

    thread_id = str(instance.pk)
    scope = custody.scope_for(thread_id)
    try:
        # Savepoint, so a failure here does not break the thread delete.
        with transaction.atomic():
            if (
                Turn.objects.filter(thread_id=thread_id).exists()
                or TurnContent.objects.filter(scope=scope).exists()
            ):
                custody.forget(scope)
    except Exception:
        logger.exception("Oversight could not forget deleted thread {}", thread_id)


def settings_changed(setting: str, **kwargs: Any) -> None:
    if setting == "AI_SDK_OVERSIGHT":
        from django_ai_sdk.oversight.resolve import invalidate

        invalidate()


def connect() -> None:
    from django_ai_sdk.conversation.models import Thread

    post_delete.connect(
        forget_deleted_thread,
        sender=Thread,
        dispatch_uid="django_ai_sdk.oversight.forget_deleted_thread",
    )
    setting_changed.connect(
        settings_changed, dispatch_uid="django_ai_sdk.oversight.settings_changed"
    )


__all__ = ["connect", "forget_deleted_thread"]
