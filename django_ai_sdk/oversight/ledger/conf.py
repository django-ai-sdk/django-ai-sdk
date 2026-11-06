"""The AI_SDK_OVERSIGHT_LEDGER setting. docs/content/oversight-ledger.md lists the keys."""

from __future__ import annotations

from typing import Any

from django.conf import settings

from django_ai_sdk.checks import AppSettings, number, of_type

MIN_SECRET_CHARS = 32
_text = of_type(str, label="a string")


def _secret(value: Any) -> str:
    if problem := _text(value):
        return problem
    if 0 < len(value) < MIN_SECRET_CHARS:
        return f"is shorter than {MIN_SECRET_CHARS} characters"
    return ""


setting = AppSettings(
    "AI_SDK_OVERSIGHT_LEDGER",
    {
        # Base64 Ed25519 private key. Empty: checkpoints are unsigned (W001).
        "SIGNING_KEY": ("", _text, "`manage.py ledger_keys` prints one."),
        # The log's name: the checkpoint origin line and the key name it is signed under.
        # "" means the first ALLOWED_HOSTS entry + "/oversight".
        "ORIGIN": ("", _text, 'An origin line such as "example.com/oversight".'),
        # Empty: derived from SECRET_KEY (W002), so rotating it renames everyone.
        "PSEUDONYM_SECRET": (
            "",
            _secret,
            f"A random string of at least {MIN_SECRET_CHARS} characters.",
        ),
        # W003 warns when the chain grew but no checkpoint was written for this many days.
        "CHECKPOINT_DAYS": (2, number(1), "A number of days, such as 2."),
    },
)


def origin() -> str:
    """The origin line of this deployment's checkpoints, and the name they are signed under."""
    configured = setting("ORIGIN")
    if configured:
        return configured
    hosts = [host for host in getattr(settings, "ALLOWED_HOSTS", []) if host not in {"*", ""}]
    return f"{hosts[0] if hosts else 'localhost'}/oversight"


__all__ = ["MIN_SECRET_CHARS", "origin", "setting"]
