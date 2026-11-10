"""The review API over the oversight record, which the HTTP layers are built on.

Optional: core never imports it. Every method checks the `oversight` permission domain and
its scope hooks, and every read writes an `AccessRecord` (docs/content/manual/oversight.md,
Review API).
"""

from __future__ import annotations

from django_ai_sdk.oversight.services.metrics import MetricReads
from django_ai_sdk.oversight.services.reviews import ReviewActions
from django_ai_sdk.oversight.services.turns import TurnReads


class OversightService(TurnReads, ReviewActions, MetricReads):
    """Read turns, work the review queues, count what happened."""


__all__ = ["OversightService"]
