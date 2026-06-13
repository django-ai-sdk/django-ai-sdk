"""Today's date and time tool."""

from django.utils import timezone
from haystack.tools import Tool


def get_datetime() -> dict:
    """Get current time and date in Europe/Amsterdam timezone."""
    tz = timezone.get_current_timezone()
    nowtz = timezone.now().astimezone(tz)

    return {
        "today": nowtz.date().isoformat(),
        "current_time": nowtz.timetz().isoformat(),
    }


def get_today(**kwargs) -> Tool:
    """Current date and time tool."""
    return Tool(
        name="get_today",
        parameters={},
        description="Get current date and time",
        function=get_datetime,
    )
