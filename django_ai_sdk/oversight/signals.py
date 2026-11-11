"""Forget oversight content when a thread or user is deleted (the oversight guide's
Forgetting says what goes and what stays), and drop cached policies when
AI_SDK_OVERSIGHT changes.
"""

from __future__ import annotations

from itertools import batched
from typing import Any

from django.core.signals import setting_changed
from django.db import transaction
from django.db.models.signals import post_delete

from django_ai_sdk.logger import get_logger

logger = get_logger(__name__)

# How many threadless turns to forget per delete statement.
FORGET_BATCH = 500


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


def forget_deleted_user(sender: Any, instance: Any, **kwargs: Any) -> None:
    """Forget a deleted user on the record and the content of their threadless turns."""
    actor_id = str(instance.pk)
    try:
        # Savepoint, so a failure here does not break the user delete.
        with transaction.atomic():
            forget_actor(actor_id)
    except Exception:
        logger.exception(
            "Oversight could not forget deleted user {}; run "
            "`manage.py oversight_forget_actor {}` to try again",
            actor_id,
            actor_id,
        )


def forget_actor(actor_id: str) -> int:
    """Clear `actor_id` from the turns it asked, after forgetting those turns' own content.

    A turn in a thread keeps its content until that thread is deleted (a user's own
    threads go with them). A turn without a thread is its own scope, `turn:<id>`:
    its content rows, and reviewers' notes on it, go in one delete per batch of
    FORGET_BATCH. The answers, matches and reviews stay.
    Returns how many turns named the actor.

    Safe to run again: `manage.py oversight_forget_actor <id>` does, for a user whose
    delete could not reach the record (the failure is logged).
    """
    from django_ai_sdk.oversight import custody
    from django_ai_sdk.oversight.models import ReviewItem, Turn, TurnContent

    if not actor_id:
        return 0
    turns = Turn.objects.filter(actor_id=actor_id)
    threadless = turns.filter(thread_id="").values_list("pk", flat=True).iterator(FORGET_BATCH)
    for batch in batched(threadless, FORGET_BATCH):
        # The turns stay and SET_NULL clears their content link.
        TurnContent.objects.filter(
            scope__in=[custody.scope_for(turn_id=turn_id) for turn_id in batch]
        ).delete()
        ReviewItem.objects.filter(turn_id__in=batch).exclude(note="").update(note="")
    # Open turns get a marker, so they store no content when they close.
    for turn_id in turns.filter(thread_id="", status=Turn.Status.OPEN).values_list("pk", flat=True):
        custody.forget(custody.scope_for(turn_id=turn_id))
    return turns.update(actor_id="")


def settings_changed(setting: str, **kwargs: Any) -> None:
    if setting == "AI_SDK_OVERSIGHT":
        from django_ai_sdk.oversight.resolve import invalidate

        invalidate()


def connect() -> None:
    from django.conf import settings

    from django_ai_sdk.conversation.models import Thread

    post_delete.connect(
        forget_deleted_thread,
        sender=Thread,
        dispatch_uid="django_ai_sdk.oversight.forget_deleted_thread",
    )
    post_delete.connect(
        forget_deleted_user,
        sender=settings.AUTH_USER_MODEL,
        dispatch_uid="django_ai_sdk.oversight.forget_deleted_user",
    )
    setting_changed.connect(
        settings_changed, dispatch_uid="django_ai_sdk.oversight.settings_changed"
    )


__all__ = ["connect", "forget_actor", "forget_deleted_thread", "forget_deleted_user"]
