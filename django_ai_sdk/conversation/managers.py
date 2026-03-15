from datetime import timedelta
from typing import Any

from django.db import models
from django.db.models import QuerySet
from django.utils import timezone


class ThreadManager(models.Manager):
    """Custom manager for Thread model with helpful query methods."""

    def for_user(self, user: Any) -> QuerySet:
        """Get threads for a specific user."""
        return self.filter(user=user)

    def recent(self, days: int = 30) -> QuerySet:
        """Get threads from the last N days."""
        cutoff_date = timezone.now() - timedelta(days=days)
        return self.filter(created_at__gte=cutoff_date)

    def with_messages(self) -> QuerySet:
        """Get threads that have at least one message."""
        return self.filter(messages__isnull=False).distinct()

    def by_assistant(self, assistant_id: str) -> QuerySet:
        """Get threads for a specific assistant."""
        return self.filter(metadata__assistant_id=assistant_id)


class MessageManager(models.Manager):
    """Custom manager for Message model with helpful query methods."""

    def for_thread(self, thread: models.Model) -> QuerySet:
        """Get messages for a specific thread."""
        return self.filter(thread=thread)

    def by_role(self, role: str) -> QuerySet:
        """Get messages by role (user, assistant, system, tool).
        Note: This requires database query with JSON field extraction.
        """
        return self.filter(result__role=role)

    def with_tools(self) -> QuerySet:
        """Get messages that used tools."""
        return self.exclude(result__tool_calls=[])

    def user_messages(self) -> QuerySet:
        """Get user messages only."""
        return self.by_role("user")

    def assistant_messages(self) -> QuerySet:
        """Get assistant messages only."""
        return self.by_role("assistant")

    def conversation_history(self, thread: models.Model, include_system: bool = False) -> QuerySet:
        """Get conversation history for a thread, optionally including system messages."""
        queryset = self.filter(thread=thread).order_by("created_at")
        if not include_system:
            queryset = queryset.exclude(result__role="system")
        return queryset
