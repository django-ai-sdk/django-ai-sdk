"""How a turn's content is stored, read back and forgotten.

Core's `PlainCustody` stores content as is (masked per CAPTURE) and deletes it to
forget. An app installs its own with `install()` in `AppConfig.ready()`; every method
runs in the caller's transaction (the oversight manual's Custody section).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterable

    from django.db.models import Model


# Prefixes of the scopes `scope_for` returns.
THREAD_SCOPE = "thread:"
TURN_SCOPE = "turn:"
# `TurnContent.capture` of PlainCustody's marker for a scope forgotten while a turn was open.
FORGOTTEN = "forgotten"


class Forgotten(Exception):
    """The scope was forgotten, so nothing new may be stored for it."""


def scope_for(thread_id: Any = "", turn_id: Any = "") -> str:
    """Return the scope a turn's content is kept and forgotten under.

    The thread's, or the turn's own when it has no thread.
    """
    return f"{THREAD_SCOPE}{thread_id}" if thread_id else f"{TURN_SCOPE}{turn_id}"


class Custody(Protocol):
    def store(self, scope: str, content: dict[str, Any]) -> dict[str, Any]:
        """Return the value to store for `content`. May raise Forgotten."""
        ...

    def load_many(self, items: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any] | None]:
        """Return the content behind each (scope, stored) pair, in order and in a fixed
        number of queries. None for an item that can no longer be read."""
        ...

    def forget(self, scope: str) -> None:
        """Forget everything kept for `scope` ("thread:<id>", "turn:<id>")."""
        ...

    def chained(self, items: list[tuple[str, Model]]) -> None:
        """Rows just written, as (kind, row): "turn", "answer", "match".

        Kinds are "turn", "answer", "match" and "access" (an `AccessRecord`). A
        turn arrives again on each update: once when its gate ends and again when
        it closes.
        "aside" is a `TurnContent` whose judge text or reviewer notes were added to.
        """
        ...

    def pruning(self, turn_ids: Iterable[Any]) -> None:
        """Turns about to be deleted by retention."""
        ...


class PlainCustody:
    """Content kept as it is; forgetting deletes it; nothing is chained."""

    def store(self, scope: str, content: dict[str, Any]) -> dict[str, Any]:
        from django_ai_sdk.oversight.models import TurnContent

        if TurnContent.objects.filter(scope=scope, capture=FORGOTTEN).exists():
            raise Forgotten(scope)
        return content

    def load_many(self, items: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any] | None]:
        return [stored for _scope, stored in items]

    def forget(self, scope: str) -> None:
        from django_ai_sdk.oversight.models import Turn, TurnContent
        from django_ai_sdk.utils import as_uuid

        # The turns stay. Their content link is cleared by SET_NULL.
        TurnContent.objects.filter(scope=scope).delete()
        # A turn still open here would store its content when it closes. Leave a
        # marker that refuses it, as the shredded key does under ledger custody.
        if scope.startswith(THREAD_SCOPE):
            turns = Turn.objects.filter(thread_id=scope.removeprefix(THREAD_SCOPE))
        else:
            turns = Turn.objects.filter(pk=as_uuid(scope.removeprefix(TURN_SCOPE)))
        if turns.filter(status=Turn.Status.OPEN).exists():
            TurnContent.objects.create(scope=scope, capture=FORGOTTEN)

    def chained(self, items: list[tuple[str, Model]]) -> None:
        return None

    def pruning(self, turn_ids: Iterable[Any]) -> None:
        return None


_custody: Custody = PlainCustody()


def install(custody: Custody | None) -> None:
    """Make `custody` the one every turn uses; None puts PlainCustody back."""
    global _custody
    _custody = custody or PlainCustody()


def get() -> Custody:
    return _custody


def is_plain() -> bool:
    """Whether content is kept as it is (no custody app installed)."""
    return isinstance(_custody, PlainCustody)


__all__ = [
    "Custody",
    "Forgotten",
    "PlainCustody",
    "get",
    "install",
    "is_plain",
    "scope_for",
]
