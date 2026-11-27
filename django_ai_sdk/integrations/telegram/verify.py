from __future__ import annotations

from django.utils.crypto import constant_time_compare


def is_sent_by_telegram(webhook_secret: str, secret_header: str) -> bool:
    return bool(webhook_secret) and constant_time_compare(secret_header, webhook_secret)
