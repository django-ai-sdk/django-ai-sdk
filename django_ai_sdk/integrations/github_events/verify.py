"""GitHub webhook signing, HMAC-SHA256 over the body."""

from __future__ import annotations

import hashlib
import hmac

PREFIX = "sha256="


def is_signed_by_github(webhook_secret: str, signature: str, body: bytes) -> bool:
    """Check the GitHub sha256 signature of `body`.

    GitHub signs no timestamp, so this cannot catch a replay. The integration's
    long dedup window does that instead. The deprecated sha1 scheme is not
    accepted.
    """
    if not webhook_secret or not signature.startswith(PREFIX):
        return False
    expected = hmac.new(webhook_secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(PREFIX + expected, signature)
