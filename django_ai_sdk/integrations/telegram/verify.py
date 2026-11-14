"""Telegram webhook secret check."""

from __future__ import annotations

from django.utils.crypto import constant_time_compare


def is_sent_by_telegram(webhook_secret: str, secret_header: str) -> bool:
    """Check the secret Telegram echoes back from setWebhook.

    Telegram signs nothing, so this is the only proof. Refuse an empty secret, or
    a request without the header would match.
    """
    return bool(webhook_secret) and constant_time_compare(secret_header, webhook_secret)
