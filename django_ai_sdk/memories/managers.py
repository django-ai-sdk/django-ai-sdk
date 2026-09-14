from __future__ import annotations

from typing import TYPE_CHECKING

from django.db import models
from django.db.models.functions import Coalesce, Length

if TYPE_CHECKING:
    from django.db.models import QuerySet


class EntryManager(models.Manager):
    """Manager for Entry"""

    def for_rag(self, thread_id: str) -> QuerySet:
        """Entries linked to this thread's memories, with `memory` preloaded."""
        return (
            self.filter(
                memory__thread_links__thread_id=thread_id,
                memory__thread_links__active=True,
            )
            .select_related("memory")
            .annotate(
                file_size=Coalesce("document__file_size", Length("content")),
            )
        )
