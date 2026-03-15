from typing import Any

from django.http import HttpRequest
from ninja import Router

from django_ai_sdk.tracking.utils import tracker

router = Router()


@router.get("")
def tracking_view(request: HttpRequest) -> dict[str, Any]:
    """Get current tracking state."""
    return tracker.get_state()
