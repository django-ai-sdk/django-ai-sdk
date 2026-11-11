"""Counts over a window of turns: outcomes, report-mode outcomes, rules and queues.

Everything is a SQL aggregate over the turns the reader may reach (`scope_turns`).
"""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

from django.db.models import Count, F, Q
from django.utils import timezone

from django_ai_sdk.oversight.schemas import MetricsOut
from django_ai_sdk.oversight.services.access import METRICS, OversightAccess, count_by

if TYPE_CHECKING:
    from django_ai_sdk.types import UserType

# The longest window `summarize` counts over.
MAX_DAYS = 365


class MetricReads(OversightAccess):
    @classmethod
    async def summarize(
        cls, *, user: UserType, days: int = 14, agent_id: str = "", policy: str = ""
    ) -> MetricsOut:
        """Count the last `days` of turns, optionally for one agent or policy."""
        from django_ai_sdk.oversight.models import ReviewItem, RuleMatch, Turn

        await cls.has_perms(user, METRICS)
        since = timezone.now() - timedelta(days=max(1, min(days, MAX_DAYS)))
        narrow = {
            key: value for key, value in {"agent_id": agent_id, "policy": policy}.items() if value
        }
        turns = await cls.scoped(
            user, METRICS, Turn.objects.filter(created_at__gte=since, **narrow)
        )
        # Open reviews are counted regardless of age.
        matches = RuleMatch.objects.filter(turn__in=turns.values("pk"))
        waiting = await cls.scoped(
            user,
            METRICS,
            ReviewItem.objects.filter(
                status=ReviewItem.Status.OPEN,
                **{f"turn__{key}": value for key, value in narrow.items()},
            ),
        )
        return MetricsOut(
            since=since,
            turns=await turns.acount(),
            by_outcome=await count_by(turns, "outcome"),
            by_report_outcome=await count_by(
                turns.exclude(report_outcome="").exclude(report_outcome=F("outcome")),
                "report_outcome",
            ),
            by_channel=await count_by(turns, "channel"),
            by_agent=await count_by(turns, "agent_id"),
            rules=[
                row
                async for row in matches.order_by("rule", "mode")
                .values("rule", "mode")
                .annotate(matches=Count("id"), acted=Count("id", filter=Q(acted=True)))
            ],
            queues=await count_by(waiting, "queue"),
        )


__all__ = ["MetricReads"]
