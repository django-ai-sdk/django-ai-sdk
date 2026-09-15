"""Slack request signing, version v0."""

from __future__ import annotations

import hashlib
import hmac
import time

VERSION = "v0"

# Slack's own recommendation. A replayed request outside this window is refused even
# though its signature is still valid.
MAX_AGE_SECONDS = 300


def is_signed_by_slack(signing_secret: str, timestamp: str, signature: str, body: bytes) -> bool:
    """Whether body carries a current Slack signature made with signing_secret."""
    if not signing_secret or not timestamp or not signature:
        return False
    try:
        age = abs(time.time() - int(timestamp))
    except ValueError:
        return False
    if age > MAX_AGE_SECONDS:
        return False

    basestring = b"%s:%s:%s" % (VERSION.encode(), timestamp.encode(), body)
    expected = (
        f"{VERSION}=" + hmac.new(signing_secret.encode(), basestring, hashlib.sha256).hexdigest()
    )
    return hmac.compare_digest(expected, signature)
