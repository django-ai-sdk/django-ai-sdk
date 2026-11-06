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
                custody.get().forget(scope)
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
    """Forget a user on the turns they asked. Returns how many turns named them.

    Clears `Turn.actor_id`. A turn in a thread keeps its content until the thread is
    deleted. A turn without a thread is its own `turn:<id>` scope and is forgotten
    here. Access records keep the reader id but lose the name. Answers, matches and
    reviews stay.

    Safe to run again. `manage.py oversight_forget_actor <id>` calls it when the
    delete signal failed.
    """
    from django_ai_sdk.oversight import custody
    from django_ai_sdk.oversight.models import AccessRecord, Turn, TurnContent

    if not actor_id:
        return 0
    custodian = custody.get()
    turns = Turn.objects.filter(actor_id=actor_id)
    plain = custody.is_plain()
    threadless = turns.filter(thread_id="").values_list("pk", flat=True).iterator(FORGET_BATCH)
    for batch in batched(threadless, FORGET_BATCH):
        scopes = [custody.scope_for(turn_id=turn_id) for turn_id in batch]
        if plain:
            # Faster path. The turns stay and SET_NULL clears their content link.
            TurnContent.objects.filter(scope__in=scopes).delete()
        else:
            for scope in scopes:
                custodian.forget(scope)
    AccessRecord.objects.filter(reader_id=actor_id).exclude(reader_label=actor_id).update(
        reader_label=actor_id
    )
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
