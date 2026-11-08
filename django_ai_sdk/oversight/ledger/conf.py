"""The AI_SDK_LEDGER setting. docs/content/oversight-ledger.md lists the keys."""

from __future__ import annotations

from typing import Any

from django.conf import settings

from django_ai_sdk.checks import AppSettings, of_type, secret

MIN_SECRET_CHARS = 32
_text = of_type(str, label="a string")


def _pseudonym_secrets(value: Any) -> str:
    """A string, or a list of them with the current one first, each long enough."""
    values = value if isinstance(value, list | tuple) else [value]
    for item in values:
        if not isinstance(item, str):
            return "is not a string or a list of strings"
        if 0 < len(item) < MIN_SECRET_CHARS:
            return f"holds a secret shorter than {MIN_SECRET_CHARS} characters"
    return ""


setting = AppSettings(
    "AI_SDK_LEDGER",
    {
        # Base64 Ed25519 private key. Empty: checkpoints are unsigned (W001).
        "SIGNING_KEY": ("", secret(_text), "`manage.py ledger_keys` prints one."),
        # The log's name: the checkpoint origin line and the key name it is signed under.
        # "" means the first ALLOWED_HOSTS entry + "/oversight".
        "ORIGIN": ("", _text, 'An origin line such as "example.com/oversight".'),
        # Empty: derived from SECRET_KEY and SECRET_KEY_FALLBACKS (W002). A list keeps
        # former secrets, current first, so erasure still finds turns named under them.
        "PSEUDONYM_SECRET": (
            "",
            secret(_pseudonym_secrets),
            f"A random string of at least {MIN_SECRET_CHARS} characters, or a list of "
            "them with the current one first.",
        ),
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
