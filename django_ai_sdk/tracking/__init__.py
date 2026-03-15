"""
DEPRECATED: I'm gonna remove this whole app I think
I had the idea to track AI calls, but this is to naive.
This might become something like a event system.
That would make logging and other analytics easier.
"""

from django_ai_sdk.tracking.utils import (
    OptimisticTracker,
    track_embedding,
    track_image,
    track_llm,
    tracker,
)

__all__ = [
    "OptimisticTracker",
    "track_embedding",
    "track_image",
    "track_llm",
    "tracker",
]
