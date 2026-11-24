from __future__ import annotations

import hashlib
import hmac

PREFIX = "sha256="


def is_signed_by_github(webhook_secret: str, signature: str, body: bytes) -> bool:
    """Check the GitHub sha256 signature of `body`; sha1 is not accepted.

    No timestamp is signed: replay protection is the integration's dedup window.
    """
    if not webhook_secret or not signature.startswith(PREFIX):
        return False
    expected = hmac.new(webhook_secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(PREFIX + expected, signature)
