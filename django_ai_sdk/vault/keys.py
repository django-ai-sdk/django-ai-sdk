"""Master keys, which wrap each scope's data key.

A `KeyProvider` wraps data keys under its current master key and unwraps keys
wrapped under any master key it still has. Each wrapped key stores the master key
id. To rotate, put the new key first in `MASTER_KEYS`, keep the old one after it,
run `manage.py vault_rewrap`, then drop the old one. Data keys under an unknown
key id fail `ai_sdk.vault.E002`.

The default provider reads `AI_SDK_VAULT["MASTER_KEYS"]`. Set
`AI_SDK_VAULT["KEY_PROVIDER"]` to use a KMS instead, so the master key never
leaves it. A provider needs only three methods.
"""

from __future__ import annotations

import functools
import hashlib
from typing import Protocol

from django.core.exceptions import ImproperlyConfigured
from django.utils.module_loading import import_string

from django_ai_sdk.vault import crypto
from django_ai_sdk.vault.errors import KeyMismatch


class KeyProvider(Protocol):
    def key_ids(self) -> list[str]:
        """Ids of the usable master keys. The first is the one `wrap` uses.

        Raises ImproperlyConfigured when there are none, or one is unusable.
        """
        ...

    def wrap(self, data_key: bytes, aad: bytes) -> tuple[str, bytes]:
        """Wrap `data_key` under the current key with `aad`. Return (key id, wrapped)."""
        ...

    def unwrap(self, key_id: str, wrapped: bytes, aad: bytes) -> bytes:
        """Unwrap a data key wrapped under `key_id`. Raise KeyMismatch if it fails."""
        ...


def key_id(master_key: bytes) -> str:
    """A master key's public id, 16 hex characters of its hash."""
    return hashlib.sha256(b"django-ai-sdk vault key id\n" + master_key).hexdigest()[:16]


@functools.lru_cache(maxsize=8)
def _decoded(raws: tuple[str, ...]) -> tuple[tuple[str, bytes], ...]:
    """Decode MASTER_KEYS into (id, key) pairs, current first. Cached per setting value."""
    keys: dict[str, bytes] = {}
    for raw in raws:
        key = crypto.decode_key_b64(raw, "Every AI_SDK_VAULT['MASTER_KEYS'] entry")
        keys.setdefault(key_id(key), key)
    return tuple(keys.items())


class SettingsKeyProvider:
    """Uses the master keys in AI_SDK_VAULT["MASTER_KEYS"], current first."""

    def _keys(self) -> dict[str, bytes]:
        from django_ai_sdk.vault.conf import setting

        configured = setting("MASTER_KEYS") or []
        # Skip blanks, such as an unset env var for a previous key.
        raws = [raw for raw in configured if raw and str(raw).strip()]
        if not raws:
            raise ImproperlyConfigured(
                "AI_SDK_VAULT['MASTER_KEYS'] is empty: the vault has no key to keep anything under."
            )
        if not crypto.has_cryptography():
            raise ImproperlyConfigured(
                "AI_SDK_VAULT['MASTER_KEYS'] is set, but the cryptography package is not "
                "installed: pip install 'django-ai-sdk[vault]'."
            )
        return dict(_decoded(tuple(str(raw) for raw in raws)))

    def key_ids(self) -> list[str]:
        return list(self._keys())

    def wrap(self, data_key: bytes, aad: bytes) -> tuple[str, bytes]:
        keys = self._keys()
        current = next(iter(keys))
        return current, crypto.encrypt_bytes(keys[current], data_key, aad)

    def unwrap(self, key_id: str, wrapped: bytes, aad: bytes) -> bytes:
        master = self._keys().get(key_id)
        if master is None:
            raise KeyMismatch(f"No AI_SDK_VAULT['MASTER_KEYS'] entry has key id {key_id!r}.")
        return crypto.decrypt_bytes(master, wrapped, aad)


def provider() -> KeyProvider:
    """The configured KeyProvider. Raises ImproperlyConfigured if it does not import."""
    from django_ai_sdk.vault.conf import setting

    path = setting("KEY_PROVIDER")
    if not path:
        return SettingsKeyProvider()
    try:
        return import_string(path)()
    except ImportError as exc:
        raise ImproperlyConfigured(
            f"AI_SDK_VAULT['KEY_PROVIDER'] {path!r} does not import: {exc}"
        ) from exc


def generate_master_key() -> str:
    """Generate a MASTER_KEYS entry (base64 of 32 random bytes)."""
    return crypto.new_key_b64()


__all__ = [
    "KeyProvider",
    "SettingsKeyProvider",
    "generate_master_key",
    "key_id",
    "provider",
]
