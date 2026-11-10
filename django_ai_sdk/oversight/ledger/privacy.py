"""Pseudonyms for who asked a turn: "u-" plus 80 bits of an HMAC of the user's key under
PSEUDONYM_SECRET, and revealing them.

No table maps pseudonyms back: `find_user_by_pseudonym` hashes every user until one
matches, which anyone with the secret could do, so guard it and reveal through the
audited `LedgerService.reveal`.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
from typing import Any

from django.conf import settings

PREFIX = "u-"
# 80 bits, so collisions between users are not a practical concern.
BYTES = 10
# User primary keys read per query in `find_user_by_pseudonym`.
REVEAL_BATCH = 2_000


def _secrets() -> list[bytes]:
    """The pseudonym secrets, current first. Former ones still find old pseudonyms."""
    from django_ai_sdk.oversight.ledger.conf import setting

    configured = setting("PSEUDONYM_SECRET")
    if configured:
        values = configured if isinstance(configured, list | tuple) else [configured]
        return [str(value).encode() for value in values if value]
    # Derive separate keys so SECRET_KEY is never used directly.
    keys = [settings.SECRET_KEY, *getattr(settings, "SECRET_KEY_FALLBACKS", [])]
    return [hashlib.sha256(b"django-ai-sdk ledger pseudonym\n" + k.encode()).digest() for k in keys]


def _secret() -> bytes:
    return _secrets()[0]


def pseudonyms(user_pk: Any) -> list[str]:
    """Every pseudonym `user_pk` may have on the record, current first."""
    if user_pk is None or str(user_pk) == "":
        return []
    return [_pseudonym(secret, user_pk) for secret in _secrets()]


def pseudonym(user_pk: Any) -> str:
    """The pseudonym for `user_pk`, or "" when there is no user."""
    if user_pk is None or str(user_pk) == "":
        return ""
    return _pseudonym(_secret(), user_pk)


def _pseudonym(secret: bytes, user_pk: Any) -> str:
    digest = hmac.new(secret, f"user:{user_pk}".encode(), hashlib.sha256).digest()
    return PREFIX + base64.b32encode(digest[:BYTES]).decode().lower()


def is_pseudonym(value: str) -> bool:
    return value.startswith(PREFIX) and len(value) == len(PREFIX) + BYTES * 8 // 5


def find_user_by_pseudonym(name: str) -> Any:
    """The user whose pseudonym is `name`, or None.

    Synchronous. Scans every user's primary key in batches.
    """
    from django.contrib.auth import get_user_model

    if not is_pseudonym(name):
        return None
    users = get_user_model()._default_manager
    secrets = _secrets()
    last = None
    while True:
        page = users.order_by("pk")
        if last is not None:
            page = page.filter(pk__gt=last)
        pks = list(page.values_list("pk", flat=True)[:REVEAL_BATCH])
        for pk in pks:
            if any(hmac.compare_digest(_pseudonym(secret, pk), name) for secret in secrets):
                return users.get(pk=pk)
        if len(pks) < REVEAL_BATCH:
            return None
        last = pks[-1]


__all__ = ["BYTES", "PREFIX", "find_user_by_pseudonym", "is_pseudonym", "pseudonym", "pseudonyms"]
