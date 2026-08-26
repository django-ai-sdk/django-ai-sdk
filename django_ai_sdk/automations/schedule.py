"""Cron schedules for automations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from croniter import croniter as _croniter
from django.core.exceptions import ImproperlyConfigured

if TYPE_CHECKING:
    from datetime import datetime, tzinfo

logger = logging.getLogger(__name__)


class Cron:
    """Fire on a standard 5-field cron expression, read in `tz` (UTC by default).

    Everything stored stays UTC, so "0 9 * * *" holds across a DST boundary.
    """

    def __init__(self, expression: str, tz: str = "UTC") -> None:
        self.expression = expression.strip()
        self.tz = tz or "UTC"
        if not _croniter.is_valid(self.expression):
            raise ImproperlyConfigured(
                f"{expression!r} is not a valid 5-field cron expression "
                "(minute hour day-of-month month day-of-week)."
            )
        self._zone = _load_zone(self.tz)

    def next_after(self, moment: datetime) -> datetime:
        """The next occurrence after `moment`, returned in `moment`'s own zone."""
        local = moment.astimezone(self._zone)
        return _croniter(self.expression, local).get_next(type(moment)).astimezone(moment.tzinfo)

    def __str__(self) -> str:
        return f"{self.expression} ({self.tz})"

    def __repr__(self) -> str:
        return f"Cron({self.expression!r}, tz={self.tz!r})"


def _load_zone(name: str) -> tzinfo:
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        logger.warning(
            "Unknown timezone %r on a cron schedule; reading the expression as UTC "
            "instead. Use an IANA name such as 'Europe/Amsterdam'.",
            name,
        )
        return ZoneInfo("UTC")


def timezone_available(name: str) -> bool:
    """Whether `name` is a resolvable IANA timezone."""
    try:
        ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        return False
    return True


__all__ = ["Cron", "timezone_available"]
