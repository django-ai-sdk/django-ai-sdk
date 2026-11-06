"""AES-GCM helpers and canonical JSON encoding.

Needs the `cryptography` package (`pip install 'django-ai-sdk[vault]'`). It is
imported lazily, on first encrypt or decrypt.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
from datetime import datetime
from typing import Any

from django.core.exceptions import ImproperlyConfigured

from django_ai_sdk.vault.errors import KeyMismatch

# Size of the random nonce prefixed to every ciphertext.
NONCE = 12
# Size of data keys and master keys (AES-256).
KEY_BYTES = 32


def has_cryptography() -> bool:
    try:
        import cryptography  # noqa: F401
    except ImportError:
        return False
    return True


def canonical(value: Any) -> bytes:
    """Encode `value` as stable JSON bytes (sorted keys, no spaces, ISO dates, UTF-8)."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=_default
    ).encode()


def as_stored(value: Any) -> Any:
    """`value` as it reads back from canonical JSON: tuples become lists, keys strings."""
    return json.loads(canonical(value))


def stored_bytes(value: Any) -> bytes:
    """Canonical JSON of `value` as it reads back, so a hash matches the stored copy."""
    return canonical(as_stored(value))


def _default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def new_key() -> bytes:
    return os.urandom(KEY_BYTES)


def new_key_b64() -> str:
    """A new key as a settings value: base64 of KEY_BYTES random bytes."""
    return base64.b64encode(new_key()).decode()


def decode_key_b64(raw: str, label: str) -> bytes:
    """Decode a key setting. Raise ImproperlyConfigured unless it is base64 of KEY_BYTES.

    `label` names the setting in the error, such as "AI_SDK_VAULT['MASTER_KEYS'] entry".
    """
    try:
        key = base64.b64decode((raw or "").strip(), validate=True)
    except (binascii.Error, ValueError):
        key = b""
    if len(key) != KEY_BYTES:
        raise ImproperlyConfigured(f"{label} must be base64 of {KEY_BYTES} bytes.")
    return key


def _aead(key: bytes) -> Any:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    return AESGCM(key)


def encrypt_bytes(key: bytes, data: bytes, aad: bytes) -> bytes:
    """Encrypt `data` with a fresh nonce prefix. Decrypting needs the same `aad`."""
    nonce = os.urandom(NONCE)
    return nonce + _aead(key).encrypt(nonce, data, aad)


def decrypt_bytes(key: bytes, blob: bytes, aad: bytes) -> bytes:
    """Decrypt a blob from `encrypt_bytes`. Raise KeyMismatch on a wrong key, blob or aad."""
    from cryptography.exceptions import InvalidTag

    try:
        return _aead(key).decrypt(blob[:NONCE], blob[NONCE:], aad)
    except (InvalidTag, ValueError) as exc:
        raise KeyMismatch("The key does not open this, or it was altered.") from exc


__all__ = [
    "KEY_BYTES",
    "NONCE",
    "as_stored",
    "canonical",
    "decode_key_b64",
    "decrypt_bytes",
    "encrypt_bytes",
    "has_cryptography",
    "new_key",
    "new_key_b64",
    "stored_bytes",
]
