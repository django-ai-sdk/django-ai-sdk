"""Integrations that receive messages pushed in by a platform.

Implement verify, parse and reply on a WebhookIntegration, include webhooks/urls.py,
and run a worker. See docs/content/manual/webhooks.md.
"""

from __future__ import annotations

from django_ai_sdk.integrations.webhooks.base import (
    InboundEvent,
    WebhookIntegration,
    current_event,
)

__all__ = ["InboundEvent", "WebhookIntegration", "current_event"]
