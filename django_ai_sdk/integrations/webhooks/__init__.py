"""Inbound events: an integration a platform pushes messages into.

An agent answers in a channel by implementing verify, parse and reply on a
WebhookIntegration, mounting webhooks/urls.py, and running a worker. `slack`,
`discord` and `telegram` ship as worked examples.

Reference: docs/content/manual/webhooks.md. Walkthrough:
docs/content/integrations.md#receiving-webhooks.
"""

from __future__ import annotations

from django_ai_sdk.integrations.webhooks.base import InboundEvent, WebhookIntegration

__all__ = ["InboundEvent", "WebhookIntegration"]
