"""Discord interaction signing, Ed25519 over the timestamp and body."""

from __future__ import annotations

import time

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

# Deliberately stricter than the fifteen minutes a Discord interaction token stays
# valid for: a replayed delivery is refused long before the token it carries would
# have expired on its own.
MAX_AGE_SECONDS = 300


def is_signed_by_discord(public_key: str, timestamp: str, signature: str, body: bytes) -> bool:
    """Whether body carries a current Discord signature made for public_key."""
    if not public_key or not timestamp or not signature:
        return False
    try:
        age = abs(time.time() - int(timestamp))
    except ValueError:
        return False
    if age > MAX_AGE_SECONDS:
        return False

    try:
        key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key))
        key.verify(bytes.fromhex(signature), timestamp.encode() + body)
    except (ValueError, InvalidSignature):
        return False
    return True
