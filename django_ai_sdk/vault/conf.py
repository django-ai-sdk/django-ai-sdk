"""The AI_SDK_VAULT setting. docs/content/oversight-ledger.md lists the keys."""

from __future__ import annotations

from django_ai_sdk.checks import AppSettings, names, of_type, secret

setting = AppSettings(
    "AI_SDK_VAULT",
    {
        # Current key first; later ones only unwrap until `vault_rewrap` moves their data.
        "MASTER_KEYS": (
            [],
            secret(names),
            "A list of keys, current first. `manage.py vault_keys` prints one.",
        ),
        # "" means `keys.SettingsKeyProvider`, which uses MASTER_KEYS.
        "KEY_PROVIDER": (
            "",
            of_type(str, label="a dotted path"),
            "A dotted path to a KeyProvider.",
        ),
    },
)

__all__ = ["setting"]
