"""Keep each process's policy registry in sync with the authored policies in the database.

`AuthoredPolicyLoader.load` runs before policy lookups, at most once per
RUNTIME_CACHE_SECONDS. It reads every authored policy and registers or unregisters it.
A policy that is off registers with its parents and options only.

A definition that no longer validates (for example, a parent was removed in a deploy)
is unregistered, so turns that name it are refused instead of running without it. If
the database cannot be read, the registered policies stay. Any change rechecks every
authored policy, because a changed parent can break its children.
"""

from __future__ import annotations

from typing import Any

from asgiref.sync import sync_to_async

from django_ai_sdk.logger import get_logger
from django_ai_sdk.oversight.authoring.definitions import (
    DefinitionError,
    check_definition,
    in_force,
    is_authored,
)
from django_ai_sdk.oversight.registry import get_policy_class, register_policy, unregister_policy

logger = get_logger(__name__)


class AuthoredPolicyLoader:
    """Policy loader for authored policies. One instance per process."""

    def __init__(self) -> None:
        # (row count, newest change) from the last read. Unchanged means nothing to do.
        self.stamp: tuple[int, Any] | None = None
        # name -> when the loaded definition was saved
        self.loaded: dict[str, Any] = {}
        # name -> why it failed to load
        self.broken: dict[str, str] = {}

    async def load(self) -> None:
        """Read the authored policies now and register any changes."""
        await sync_to_async(self._load)()

    def _load(self) -> None:
        from django.db.models import Count, Max

        from django_ai_sdk.oversight.authoring.models import AuthoredPolicy

        stamp_row = AuthoredPolicy.objects.aggregate(count=Count("id"), newest=Max("updated_at"))
        stamp = (stamp_row["count"], stamp_row["newest"])
        if stamp == self.stamp:
            return
        # Register parents before children so a child can find its authored parent.
        pending = {row.name: row for row in AuthoredPolicy.objects.all()}
        for name in [loaded for loaded in self.loaded if loaded not in pending]:
            self._unregister(name)
        progress = True
        while pending and progress:
            progress = False
            for name, row in list(pending.items()):
                waiting = [
                    parent for parent in row.extends or [] if parent in pending and parent != name
                ]
                if waiting:
                    continue
                pending.pop(name)
                progress = True
                self._register(row)
        for name in pending:
            # Whatever is left is in a cycle, so none of it loads.
            self.broken[name] = "it extends a policy that extends it"
            self._unregister(name, keep_broken=True)
        self.stamp = stamp

    def _register(self, row: Any) -> None:
        name = row.name
        try:
            cls, _ = check_definition(name, in_force(row.definition(), active=row.active))
        except DefinitionError as exc:
            self.broken[name] = str(exc)
            logger.error("Authored policy {!r} does not load: {}", name, exc)
            self._unregister(name, keep_broken=True)
            return
        register_policy(cls)
        self.loaded[name] = row.updated_at
        self.broken.pop(name, None)

    def _unregister(self, name: str, *, keep_broken: bool = False) -> None:
        if is_authored(get_policy_class(name)):
            unregister_policy(name)
        self.loaded.pop(name, None)
        if not keep_broken:
            self.broken.pop(name, None)


loader = AuthoredPolicyLoader()


__all__ = ["AuthoredPolicyLoader", "loader"]
