---
title: Vault
type: docs
weight: 13
---

**Keep data under a key per scope, and forget a scope by destroying its key.**

The vault (`django_ai_sdk.vault`) is an opt-in app that knows nothing of oversight: any app of yours may keep data in it. The [ledger](/oversight-ledger/) is the first user, and installs with it. Add it to `INSTALLED_APPS` and run `migrate`; it needs the `cryptography` package and at least one master key (`manage.py vault_keys` prints one).

## Keeping data

Each scope (any string: `thread:42`, `invoice:7`) gets a data key on first use: AES-256-GCM, wrapped by the current master key, stored as a `DataKey` row (table `django_ai_sdk_data_keys`) with that master key's id. What it keeps comes back as an envelope naming the scope and the key row, never the key:

```python
from django_ai_sdk import vault

envelope = vault.encrypt_json("invoice:7", {"iban": "NL91ABNA0417164300"})
vault.decrypt_json(envelope)          # {"iban": "NL91ABNA0417164300"}

token = vault.encrypt_text("invoice:7", "Paid by Jan")
vault.decrypt_text(token)             # "Paid by Jan"

vault.shred("invoice:7")
vault.decrypt_json(envelope)          # None: the key is gone
vault.is_shredded("invoice:7")        # True
vault.encrypt_text("invoice:7", "x")  # raises vault.ScopeShredded
```

- `encrypt(scope, data)` / `decrypt(envelope)` take bytes; `encrypt_json` / `decrypt_json` take a value as canonical JSON; `encrypt_text` / `decrypt_text` a string, as a `vault1:` token.
- `encrypt*` raise `ScopeShredded` for a shredded scope, and `is_shredded(scope)` says whether it is. `decrypt*` return None once the scope is shredded, and raise `KeyMismatch` (a `VaultError`) when the envelope does not open here: altered bytes, an envelope moved to another scope, a key row that is gone, or a master key that is not configured.
- `shred(scope)` returns `(key row, shredded now)`. A scope shredded before its first use keeps nothing either.
- `rewrap(batch=500)` wraps every data key under the current master key and returns how many moved and the scopes none of the master keys opens; `vault_rewrap` runs it.
- `is_envelope(value)` says whether a value has an envelope's shape; `key_rows(envelopes)` reads the key rows of many envelopes in one query, to pass as `decrypt(..., rows=)`.
- `digest(scope, data)` is a keyed hash of bytes under the scope's existing key: the same data in two scopes hashes differently, and nobody without the key finds a short text by guessing it.
- `write_key(scope)` returns the scope's key row, created on first use, to pass as `row=` to `encrypt`, `encrypt_json` and `digest`: several writes under one key unwrap it once.

All of it is synchronous and touches the database: call it from sync code, or through `sync_to_async`.

## Master keys and rotation

`MASTER_KEYS` is a list, current first. Each key's id is 16 hex characters of a hash of it, recorded on every data key it wraps. To rotate:

1. Put the new key first and keep the old one after it: `[env("VAULT_MASTER_KEY"), env("VAULT_PREVIOUS_MASTER_KEY")]`. New data keys are wrapped under the new one; old ones still open.
2. Run `manage.py vault_rewrap`: every data key is unwrapped and wrapped again under the current key, 500 per transaction. It fails, naming them, when a key opens under none of the configured master keys.
3. Drop the old key. `check --deploy --database default` warns while data keys still use it (`ai_sdk.vault.W001`), and errors once a data key's master key is gone (`ai_sdk.vault.E002`).

A blank entry (an unset environment variable for the previous key) is skipped.

A key management service can hold the master key instead: set `AI_SDK_VAULT["KEY_PROVIDER"]` to the dotted path of a class with three methods, the `KeyProvider` protocol:

```python
from django_ai_sdk.vault import KeyProvider


class KmsKeys(KeyProvider):
    def key_ids(self) -> list[str]: ...                                # current first
    def wrap(self, data_key: bytes, aad: bytes) -> tuple[str, bytes]: ...   # (key id, wrapped)
    def unwrap(self, key_id: str, wrapped: bytes, aad: bytes) -> bytes: ...
```

The default is `SettingsKeyProvider`, which reads `MASTER_KEYS`. `vault.keys.generate_master_key()` returns a new `MASTER_KEYS` entry, as `vault_keys` prints.

## Vault settings

`AI_SDK_VAULT`, one dict:

| Key | Default | What |
| --- | --- | --- |
| `MASTER_KEYS` | `[]` | base64 of 32 random bytes each, current first. Empty: the vault keeps nothing (`ai_sdk.vault.E001`). |
| `KEY_PROVIDER` | `""` | Dotted path of a `KeyProvider`. Empty: `SettingsKeyProvider` over `MASTER_KEYS`. |
