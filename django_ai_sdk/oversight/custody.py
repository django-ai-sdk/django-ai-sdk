"""How a turn's content is kept and forgotten.

Content is stored as it is (masked per CAPTURE) under a scope, the thread's or the
turn's own. Forgetting a scope deletes its content, blanks reviewers' notes on its
turns and refuses new content for it.
Every function runs in the caller's transaction.
"""

from __future__ import annotations

from typing import Any

# Prefixes of the scopes `scope_for` returns.
THREAD_SCOPE = "thread:"
TURN_SCOPE = "turn:"
# `TurnContent.capture` of the marker for a scope forgotten while a turn was open.
FORGOTTEN = "forgotten"


class Forgotten(Exception):
    """The scope was forgotten, so nothing new may be stored for it."""


def scope_for(thread_id: Any = "", turn_id: Any = "") -> str:
    """Return the scope a turn's content is kept and forgotten under.

    The thread's, or the turn's own when it has no thread.
    """
    return f"{THREAD_SCOPE}{thread_id}" if thread_id else f"{TURN_SCOPE}{turn_id}"


def store(scope: str, content: dict[str, Any]) -> dict[str, Any]:
    """Return the value to store for `content`. Raises Forgotten for a forgotten scope."""
    from django_ai_sdk.oversight.models import TurnContent

    if TurnContent.objects.filter(scope=scope, capture=FORGOTTEN).exists():
        raise Forgotten(scope)
    return content


def forget(scope: str) -> None:
    """Forget everything kept for `scope` ("thread:<id>", "turn:<id>")."""
    from django_ai_sdk.oversight.models import ReviewItem, Turn, TurnContent
    from django_ai_sdk.utils import as_uuid

    # The turns stay. Their content link is cleared by SET_NULL.
    TurnContent.objects.filter(scope=scope).delete()
    # A turn still open here would store its content when it closes. Leave a
    # marker that refuses it.
    if scope.startswith(THREAD_SCOPE):
        turns = Turn.objects.filter(thread_id=scope.removeprefix(THREAD_SCOPE))
    else:
        turns = Turn.objects.filter(pk=as_uuid(scope.removeprefix(TURN_SCOPE)))
    ReviewItem.objects.filter(turn__in=turns).exclude(note="").update(note="")
    if turns.filter(status=Turn.Status.OPEN).exists():
        TurnContent.objects.create(scope=scope, capture=FORGOTTEN)


__all__ = ["Forgotten", "forget", "scope_for", "store"]
