"""Encrypt bytes, JSON and text per scope. Shred scopes and rewrap their keys.

Each scope gets its own data key on first use, wrapped by the current master key.
Encrypting returns an envelope (a JSON dict or a text token). It names the scope
and the key row, never the key itself. Shredding a scope clears its key, so its
envelopes stop decrypting and new writes raise ScopeShredded.

Everything here is synchronous and hits the database. Call it from sync code or
through sync_to_async.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
from typing import TYPE_CHECKING, Any

from django.db import transaction
from django.utils import timezone

from django_ai_sdk.utils import as_uuid
from django_ai_sdk.vault import crypto
from django_ai_sdk.vault.errors import KeyMismatch, ScopeShredded
from django_ai_sdk.vault.keys import provider

if TYPE_CHECKING:
    from django_ai_sdk.vault.models import DataKey

# Envelope format version. Bump it for a new format.
VERSION = 1
# Text tokens are this prefix plus the base64url envelope.
TEXT_PREFIX = f"vault{VERSION}:"
# Data keys rewrapped per transaction.
REWRAP_BATCH = 500


def _key_aad(scope: str) -> bytes:
    return b"data key\n" + scope.encode()


def _content_aad(scope: str, key: Any) -> bytes:
    return f"content\n{scope}\n{key}".encode()


# Attribute that caches a row's unwrapped key, with the inputs it came from.
_UNWRAPPED = "_vault_unwrapped"


def _unwrap(row: DataKey) -> bytes:
    """Unwrap the row's data key and cache it on the row object.

    Rows from one `key_rows` call are shared across many envelopes. Caching means
    one KeyProvider round trip per key instead of one per envelope.
    """
    if not row.wrapped:
        # Only a hand-edited row lacks its wrapped key without being shredded.
        raise KeyMismatch(f"The data key of {row.scope!r} has no wrapped key.")
    source = (row.scope, row.key_id, bytes(row.wrapped))
    cached = row.__dict__.get(_UNWRAPPED)
    if cached is not None and cached[0] == source:
        return cached[1]
    key = provider().unwrap(source[1], source[2], _key_aad(row.scope))
    row.__dict__[_UNWRAPPED] = (source, key)
    return key


def _key_for_write(scope: str) -> tuple[DataKey, bytes]:
    """Return the scope's key row and plaintext key, creating them on first use.

    Raises ScopeShredded when the scope was shredded, and KeyMismatch when no master
    key can unwrap it.
    """
    from django_ai_sdk.vault.models import DataKey

    row = DataKey.objects.filter(scope=scope).first()
    if row is None:
        key = crypto.new_key()
        key_id, wrapped = provider().wrap(key, _key_aad(scope))
        # Insert the key in one go so two concurrent first writes share one key,
        # instead of the second overwriting the first.
        row, created = DataKey.objects.get_or_create(
            scope=scope, defaults={"key_id": key_id, "wrapped": wrapped}
        )
        if created:
            return row, key
    return row, _open(row)


def _open(row: DataKey) -> bytes:
    if row.shredded:
        raise ScopeShredded(row.scope)
    return _unwrap(row)


def encrypt(scope: str, data: bytes) -> dict[str, Any]:
    """Encrypt `data` under the scope's key into an envelope. Raises ScopeShredded if shredded."""
    row, key = _key_for_write(scope)
    blob = crypto.encrypt_bytes(key, data, _content_aad(scope, row.id))
    return {"v": VERSION, "key": str(row.id), "scope": scope, "ct": base64.b64encode(blob).decode()}


def key_rows(envelopes: list[Any]) -> dict[str, DataKey]:
    """Load the key rows for `envelopes` in one query, for `decrypt(..., rows=)`."""
    from django_ai_sdk.vault.models import DataKey

    ids = {e["key"] for e in envelopes if is_envelope(e)}
    valid = [pk for key in ids if (pk := as_uuid(key))]
    return {str(row.pk): row for row in DataKey.objects.filter(pk__in=valid)}


def decrypt(envelope: Any, *, rows: dict[str, DataKey] | None = None) -> bytes | None:
    """Decrypt an envelope from `encrypt`. None when its scope was shredded.

    Pass `rows` from `key_rows` to skip the key row query. Raises KeyMismatch when
    it cannot be decrypted here, for example a bad envelope, a missing or mismatched
    key row, an unconfigured master key, or altered bytes.
    """
    if not is_envelope(envelope):
        raise KeyMismatch("Not a vault envelope.")
    if rows is None:
        rows = key_rows([envelope])
    row = rows.get(envelope["key"])
    if row is None or row.scope != envelope["scope"]:
        raise KeyMismatch(f"The data key of {envelope['scope']!r} is not here.")
    if row.shredded:
        return None
    try:
        blob = base64.b64decode(envelope["ct"], validate=True)
    except (binascii.Error, ValueError) as exc:
        raise KeyMismatch("The envelope was altered.") from exc
    return crypto.decrypt_bytes(_unwrap(row), blob, _content_aad(row.scope, row.id))


def is_envelope(value: Any) -> bool:
    """Check `value` has the shape of an envelope from `encrypt`."""
    return (
        isinstance(value, dict)
        and value.get("v") == VERSION
        and all(isinstance(value.get(k), str) for k in ("key", "scope", "ct"))
    )


def encrypt_json(scope: str, value: Any) -> dict[str, Any]:
    """Encrypt `value` as canonical JSON into an envelope."""
    return encrypt(scope, crypto.canonical(value))


def decrypt_json(envelope: Any, *, rows: dict[str, DataKey] | None = None) -> Any:
    """Decrypt a value from `encrypt_json`. None when its scope was shredded."""
    data = decrypt(envelope, rows=rows)
    return None if data is None else json.loads(data)


def encrypt_text(scope: str, text: str) -> str:
    """Encrypt `text` into a text token ("vault1:...")."""
    envelope = encrypt(scope, text.encode())
    return TEXT_PREFIX + base64.urlsafe_b64encode(crypto.canonical(envelope)).decode()


def decrypt_text(token: str) -> str | None:
    """Decrypt a token from `encrypt_text`. None when its scope was shredded."""
    if not token.startswith(TEXT_PREFIX):
        raise KeyMismatch("Not a vault text token.")
    try:
        envelope = json.loads(base64.urlsafe_b64decode(token[len(TEXT_PREFIX) :]))
    except (binascii.Error, ValueError) as exc:
        raise KeyMismatch("The text token was altered.") from exc
    data = decrypt(envelope)
    return None if data is None else data.decode()


def digest(scope: str, data: bytes, *, row: DataKey | None = None) -> str:
    """Keyed hash of `data` under the scope's key.

    It is stable for the same data. Being keyed, nobody without the key can confirm
    a guess of short content.

    Pass the scope's `row` from `key_rows` to skip the query. Raises ScopeShredded
    when the scope was shredded, and KeyMismatch when it has no key yet, `row` is
    another scope's, or no master key can unwrap it.
    """
    from django_ai_sdk.vault.models import DataKey

    if row is None:
        row = DataKey.objects.filter(scope=scope).first()
        if row is None:
            raise KeyMismatch(f"{scope!r} has no data key.")
    elif row.scope != scope:
        raise KeyMismatch(f"The data key passed is not the one of {scope!r}.")
    key = _open(row)
    subkey = hashlib.sha256(b"django-ai-sdk vault digest\n" + key).digest()
    return hmac.new(subkey, data, hashlib.sha256).hexdigest()


def is_shredded(scope: str) -> bool:
    """Whether `scope` was shredded."""
    from django_ai_sdk.vault.models import DataKey

    return DataKey.objects.filter(scope=scope, shredded_at__isnull=False).exists()


def shred(scope: str) -> tuple[DataKey, bool]:
    """Destroy the scope's key, or mark a keyless scope as forgotten.

    Returns the row, and False when it was already shredded. Data under the scope
    stops decrypting and new writes raise ScopeShredded.
    """
    from django_ai_sdk.vault.models import DataKey

    now = timezone.now()
    with transaction.atomic():
        row, created = DataKey.objects.get_or_create(scope=scope, defaults={"shredded_at": now})
        if created:
            return row, True
        row = DataKey.objects.select_for_update().get(pk=row.pk)
        if row.shredded:
            return row, False
        row.wrapped = None
        row.shredded_at = now
        row.save(update_fields=["wrapped", "shredded_at"])
    return row, True


def rewrap(*, batch: int = REWRAP_BATCH) -> tuple[int, list[str]]:
    """Rewrap every data key under the current master key, one batch per transaction.

    Returns the number moved and the scopes whose keys no master key could unwrap.
    Those are left alone. Safe to rerun, and safe while turns are being written.
    """
    from django_ai_sdk.vault.models import DataKey

    current = provider().key_ids()[0]
    moved = 0
    stuck: list[str] = []
    pending = DataKey.objects.filter(shredded_at__isnull=True).exclude(key_id=current)
    while True:
        ids = list(
            pending.exclude(scope__in=stuck).order_by("pk").values_list("pk", flat=True)[:batch]
        )
        if not ids:
            return moved, stuck
        with transaction.atomic():
            for row in DataKey.objects.select_for_update().filter(pk__in=ids):
                if row.shredded or row.key_id == current:
                    continue
                try:
                    key = _unwrap(row)
                except KeyMismatch:
                    stuck.append(row.scope)
                    continue
                row.key_id, row.wrapped = provider().wrap(key, _key_aad(row.scope))
                row.rewrapped_at = timezone.now()
                row.save(update_fields=["key_id", "wrapped", "rewrapped_at"])
                moved += 1


__all__ = [
    "REWRAP_BATCH",
    "TEXT_PREFIX",
    "VERSION",
    "decrypt",
    "decrypt_json",
    "decrypt_text",
    "digest",
    "encrypt",
    "encrypt_json",
    "encrypt_text",
    "is_shredded",
    "is_envelope",
    "key_rows",
    "rewrap",
    "shred",
]
