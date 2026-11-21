"""Keep each process's policy registry in sync with the authored policies in the database.

A definition that no longer validates is unregistered, so turns naming it are
blocked rather than running without it; if the database cannot be read, the
registered policies stay.
"""

from __future__ import annotations

from typing import Any

from asgiref.sync import sync_to_async

from django_ai_sdk.logger import get_logger
from django_ai_sdk.oversight.authoring.definitions import (
    DefinitionError,
    check_definition,
    is_authored,
)
from django_ai_sdk.oversight.registry import get_policy_class, register_policy, unregister_policy

logger = get_logger(__name__)


class AuthoredPolicyLoader:
    """One instance per process."""

    def __init__(self) -> None:
        # name -> updated_at of every row at the last read. Each row is compared, so a
        # save that commits late or with a skewed clock counts.
        self.stamp: dict[str, Any] | None = None
        self.loaded: dict[str, Any] = {}
        self.broken: dict[str, str] = {}

    async def load(self) -> None:
        await sync_to_async(self.load_sync)()

    def load_sync(self) -> None:
        from django_ai_sdk.oversight.authoring.models import AuthoredPolicy

        stamp = dict(AuthoredPolicy.objects.values_list("name", "updated_at"))
        if stamp == self.stamp:
            return
        for name in [loaded for loaded in self.loaded if loaded not in stamp]:
            self._unregister(name)
        for row in AuthoredPolicy.objects.all():
            self._register(row)
        self.stamp = stamp

    def _register(self, row: Any) -> None:
        name = row.name
        definition = row.definition()
        if not row.active:
            # Off: its own lenses and rules go, what it extends keeps running.
            definition |= {"lenses": [], "rules": []}
        try:
            cls, _ = check_definition(name, definition)
        except DefinitionError as exc:
            self.broken[name] = str(exc)
            logger.error("Authored policy {!r} does not load: {}", name, exc)
            self._unregister(name, keep_broken=True)
            return
        # Explicitly: the registry keeps the first of two classes with one name.
        if is_authored(get_policy_class(name)):
            unregister_policy(name)
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
