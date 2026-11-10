"""The vault: encrypts data per scope, each scope under its own key, and forgets a scope
by destroying its key (`shred`).

An opt-in app that does not need oversight; docs/content/oversight-ledger.md, "The vault",
covers install, keys and rotation.
"""

from __future__ import annotations

from django_ai_sdk.vault.errors import KeyMismatch, ScopeShredded, VaultError
from django_ai_sdk.vault.keys import KeyProvider, SettingsKeyProvider
from django_ai_sdk.vault.scopes import (
    decrypt,
    decrypt_json,
    decrypt_text,
    digest,
    encrypt,
    encrypt_json,
    encrypt_text,
    is_envelope,
    is_shredded,
    key_rows,
    shred,
    write_key,
)

__all__ = [
    "ScopeShredded",
    "KeyMismatch",
    "KeyProvider",
    "SettingsKeyProvider",
    "VaultError",
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
    "shred",
    "write_key",
]
