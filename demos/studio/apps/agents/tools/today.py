from __future__ import annotations

from django.utils import timezone
from haystack.tools import Tool


def get_datetime() -> dict:
    tz = timezone.get_current_timezone()
    nowtz = timezone.now().astimezone(tz)

    return {
        "today": nowtz.date().isoformat(),
        "current_time": nowtz.timetz().isoformat(),
    }


def get_today(**kwargs: object) -> Tool:
    return Tool(
        name="get_today",
        parameters={},
        description="Get current date and time",
        function=get_datetime,
    )
