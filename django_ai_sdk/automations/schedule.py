"""Cron schedules for automations."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from django.core.exceptions import ImproperlyConfigured

if TYPE_CHECKING:
    from datetime import datetime, tzinfo

# APScheduler 3.x counts weekdays from Monday = 0, standard cron from Sunday = 0,
# so "1-5" would silently mean Tue-Sat. Names mean the same in both.
_NUMERIC_DAY = re.compile(r"\d")


def _cron_trigger_cls() -> Any:
    # Only cron needs APScheduler, so it ships in the `automations` extra and is
    # imported lazily. Only CronTrigger is used. The SDK runs no scheduler itself.
    try:
        from apscheduler.triggers.cron import CronTrigger
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise ImproperlyConfigured(
            "Cron schedules need APScheduler. Install django-ai-sdk[automations]."
        ) from exc
    return CronTrigger


class Cron:
    """Fire on a standard 5-field cron expression, read in `tz` (UTC by default).

    Stored times stay UTC, so "0 9 * * *" holds across a DST change. APScheduler's
    CronTrigger validates the fields and handles DST. Days of the week must be names
    ("mon-fri"), not numbers.
    """

    def __init__(self, expression: str, tz: str = "UTC") -> None:
        self.expression = expression.strip()
        self.tz = tz or "UTC"
        self._zone = _load_zone(self.tz)

        fields = self.expression.split()
        if len(fields) != 5:
            raise ImproperlyConfigured(
                f"{expression!r} is not a valid 5-field cron expression "
                "(minute hour day-of-month month day-of-week)."
            )
        minute, hour, day, month, day_of_week = fields
        if _NUMERIC_DAY.search(day_of_week):
            raise ImproperlyConfigured(
                f"{expression!r}: write the day of the week by name (mon-fri, sun), not by number."
            )
        try:
            self._trigger = _cron_trigger_cls()(
                minute=minute,
                hour=hour,
                day=day,
                month=month,
                day_of_week=day_of_week,
                timezone=self._zone,
            )
        except ValueError as exc:
            raise ImproperlyConfigured(
                f"{expression!r} is not a valid 5-field cron expression "
                "(minute hour day-of-month month day-of-week)."
            ) from exc

    def next_after(self, moment: datetime) -> datetime:
        """The next occurrence strictly after `moment`, returned in `moment`'s own zone."""
        local = moment.astimezone(self._zone)
        # Passing `local` as the previous fire time skips an exact match. With None,
        # `local` itself could come back.
        next_fire = self._trigger.get_next_fire_time(local, local)
        return next_fire.astimezone(moment.tzinfo)

    def __str__(self) -> str:
        return f"{self.expression} ({self.tz})"

    def __repr__(self) -> str:
        return f"Cron({self.expression!r}, tz={self.tz!r})"


def _load_zone(name: str) -> tzinfo:
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        # Refuse instead of falling back to UTC, which would fire an hour off for
        # half the year.
        raise ImproperlyConfigured(
            f"Unknown timezone {name!r} on a cron schedule. Use an IANA name such as "
            "'Europe/Amsterdam'."
        ) from exc


__all__ = ["Cron"]
