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


class Forgotten(Exception):
    """The scope was forgotten, so nothing new may be stored for it."""


def scope_for(thread_id: Any = "", turn_id: Any = "") -> str:
    """Return the scope a turn's content is kept and forgotten under.

    The thread's, or the turn's own when it has no thread.
    """
    return f"{THREAD_SCOPE}{thread_id}" if thread_id else f"{TURN_SCOPE}{turn_id}"


class Custody(Protocol):
    def actor(self, user_pk: str) -> str:
        """The `Turn.actor_id` to store for this user. Must not query. "" maps to ""."""
        ...

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
        """Receive rows just written as (kind, row) pairs.

        Kinds are "turn", "answer", "match", "access" (an `AccessRecord`),
        "override" (a runtime switch) and "authored" (an authored policy), as
        named in `ledger.chain.KINDS`. A turn arrives again on each update: once
        when its gate ends and again when it closes.
        """
        ...

    def pruning(self, turn_ids: Iterable[Any], *, access_ids: Iterable[Any] = ()) -> None:
        """Called before retention deletes these turns (with their answers, matches
        and content) and these access records."""
        ...

    def expiring(self, content_ids: Iterable[Any]) -> None:
        """Called before retention deletes these content rows."""
        ...


class PlainCustody:
    """The default custody. Stores content as is and names users by primary key."""

    def actor(self, user_pk: str) -> str:
        return user_pk

    def store(self, scope: str, content: dict[str, Any]) -> dict[str, Any]:
        return content

    def load_many(self, items: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any] | None]:
        return [stored for _scope, stored in items]

    def forget(self, scope: str) -> None:
        from django_ai_sdk.oversight.models import TurnContent

        # The turns stay. Their content link is cleared by SET_NULL.
        TurnContent.objects.filter(scope=scope).delete()

    def chained(self, items: list[tuple[str, Model]]) -> None:
        return None

    def pruning(self, turn_ids: Iterable[Any], *, access_ids: Iterable[Any] = ()) -> None:
        return None

    def expiring(self, content_ids: Iterable[Any]) -> None:
        return None


_custody: Custody = PlainCustody()

# The protocol's methods, which `install` requires.
METHODS = tuple(
    name for name, value in vars(Custody).items() if callable(value) and not name.startswith("_")
)


def install(custody: Custody | None) -> None:
    """Use `custody` for every turn. None restores PlainCustody.

    Raises TypeError if a protocol method is missing, so an outdated custody fails
    at install time instead of during a turn.
    """
    global _custody
    if custody is not None:
        missing = [name for name in METHODS if not callable(getattr(custody, name, None))]
        if missing:
            raise TypeError(
                f"{type(custody).__name__} is not a Custody: it lacks {', '.join(missing)}"
            )
    _custody = custody or PlainCustody()


def get() -> Custody:
    return _custody


def actor_names(user_pk: str) -> set[str]:
    """All actor ids the record may hold for this user.

    That is the installed custody's name and the primary key, which turns
    recorded before that custody was installed use. Empty for "".
    """
    if not user_pk:
        return set()
    return {user_pk, _custody.actor(user_pk)} - {""}


def is_plain() -> bool:
    """Whether content is kept as it is (no custody app installed)."""
    return isinstance(_custody, PlainCustody)


__all__ = [
    "Custody",
    "Forgotten",
    "PlainCustody",
    "actor_names",
    "get",
    "install",
    "is_plain",
    "scope_for",
]
