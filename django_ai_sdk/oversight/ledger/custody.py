"""LedgerCustody: pseudonymizes who asked, encrypts turn content in the vault, and chains
the record.

Stored content is `{"vault": <envelope>, "digest": <keyed digest>, "aside": <keyed digest>}`.
The content entry hashes `digest`, which skips `content.SIDE_SECTIONS` so it stays stable
as judges and reviewers add to them. `aside` digests those sections alone, is only there
when they hold something, and is chained again each time they change.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from django.db import transaction

from django_ai_sdk.logger import get_logger
from django_ai_sdk.oversight import custody
from django_ai_sdk.oversight.content import SIDE_SECTIONS
from django_ai_sdk.oversight.ledger import chain
from django_ai_sdk.vault import (
    KeyMismatch,
    ScopeShredded,
    VaultError,
    decrypt_json,
    digest,
    encrypt_json,
    is_envelope,
    key_rows,
    shred,
    write_key,
)
from django_ai_sdk.vault.crypto import stored_bytes

if TYPE_CHECKING:
    from collections.abc import Iterable

    from django.db.models import Model

    from django_ai_sdk.oversight.models import Turn

logger = get_logger(__name__)


def content_digest(scope: str, content: dict[str, Any], *, row: Any = None) -> str:
    """Keyed digest of the content without its side sections. The chain hashes it.

    Pass the scope's vault key `row` to skip loading and unwrapping it again.
    """
    hashed = {key: value for key, value in content.items() if key not in SIDE_SECTIONS}
    # As stored, so the digest matches the content when read back.
    return digest(scope, stored_bytes(hashed), row=row)


def aside_digest(scope: str, content: dict[str, Any], *, row: Any = None) -> str | None:
    """Keyed digest of the side sections (judge text, reviewer notes). None when both are empty."""
    quotes = {key: content.get(key) or {} for key in sorted(SIDE_SECTIONS)}
    if not any(quotes.values()):
        return None
    return digest(scope, stored_bytes(quotes), row=row)


class LedgerCustody:
    """Custody that encrypts content, chains the record and pseudonymizes actors."""

    def actor(self, user_pk: str) -> str:
        from django_ai_sdk.oversight.ledger.privacy import pseudonym

        return pseudonym(user_pk)

    def former_actors(self, user_pk: str) -> list[str]:
        """Pseudonyms under former secrets, so forgetting still finds those turns."""
        from django_ai_sdk.oversight.ledger.privacy import pseudonyms

        return pseudonyms(user_pk)[1:]

    def store(self, scope: str, content: dict[str, Any]) -> dict[str, Any]:
        # One key row for the envelope and both digests: one unwrap, and a shred
        # landing in between cannot fail the turn halfway.
        try:
            row = write_key(scope)
            stored = {
                "vault": encrypt_json(scope, content, row=row),
                "digest": content_digest(scope, content, row=row),
            }
            if aside := aside_digest(scope, content, row=row):
                stored["aside"] = aside
        except ScopeShredded as exc:
            raise custody.Forgotten(scope) from exc
        return stored

    def _open(self, scope: str, stored: dict[str, Any], rows: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(stored, dict) or "vault" not in stored:
            return stored  # Stored before the ledger was installed.
        envelope = stored["vault"]
        if not is_envelope(envelope) or envelope["scope"] != scope:
            raise KeyMismatch(f"The content of {scope!r} was kept under another scope.")
        return decrypt_json(envelope, rows=rows)

    def load_many(self, items: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any] | None]:
        rows = key_rows([s.get("vault") for _, s in items if isinstance(s, dict)])
        loaded: list[dict[str, Any] | None] = []
        for scope, stored in items:
            try:
                loaded.append(self._open(scope, stored, rows))
            except VaultError:
                logger.warning("The content kept under {} does not open here", scope)
                loaded.append(None)
        return loaded

    def forget(self, scope: str) -> None:
        from django_ai_sdk.oversight.models import TurnContent

        with transaction.atomic():
            key, shredded = shred(scope)
            if shredded:
                chain.append_many([(chain.FORGET, key)])
            rows = TurnContent.objects.filter(scope=scope)
            ids = list(rows.values_list("pk", flat=True))
            if ids:
                chain.erase(ids)
                # The turns stay. SET_NULL clears their content link.
                rows.delete()

    def chained(self, items: list[tuple[str, Model]]) -> None:
        from django_ai_sdk.oversight.ledger.models import LedgerEntry
        from django_ai_sdk.oversight.models import TurnContent

        turns = [cast("Turn", row) for kind, row in items if kind == chain.TURN]
        unchained = {str(turn.content_id) for turn in turns if turn.content_id is not None}
        unchained -= set(
            LedgerEntry.objects.filter(kind=chain.CONTENT, ref__in=unchained).values_list(
                "ref", flat=True
            )
        )
        quoted = {
            str(pk)
            for pk in TurnContent.objects.filter(
                pk__in=unchained, stored__has_key="aside"
            ).values_list("pk", flat=True)
        }
        expanded: list[tuple[str, Model]] = []
        for kind, row in items:
            content_id = str(cast("Turn", row).content_id) if kind == chain.TURN else ""
            if content_id in unchained:
                # A content row is chained once, just before its turn's first entry.
                unchained.discard(content_id)
                expanded.append((chain.CONTENT, TurnContent(pk=content_id)))
                if content_id in quoted:
                    expanded.append((chain.ASIDE, TurnContent(pk=content_id)))
            expanded.append((kind, row))
        chain.append_many(expanded)

    def pruning(self, turn_ids: Iterable[Any], *, access_ids: Iterable[Any] = ()) -> None:
        chain.prune(turn_ids, access_ids=access_ids)

    def expiring(self, content_ids: Iterable[Any]) -> None:
        chain.prune(content_ids=content_ids)


__all__ = ["LedgerCustody", "aside_digest", "content_digest"]
