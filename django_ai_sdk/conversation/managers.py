from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

from django.db import models
from django.utils import timezone

if TYPE_CHECKING:
    from django.contrib.auth.base_user import AbstractBaseUser
    from django.db.models import QuerySet


class ThreadManager(models.Manager):
    def for_user(self, user: AbstractBaseUser) -> QuerySet:
        return self.filter(user=user)

    def recent(self, days: int = 30) -> QuerySet:
        cutoff_date = timezone.now() - timedelta(days=days)
        return self.filter(created_at__gte=cutoff_date)

    def with_messages(self) -> QuerySet:
        return self.filter(messages__isnull=False).distinct()

    def by_agent(self, agent_id: str) -> QuerySet:
        return self.filter(metadata__agent_id=agent_id)


class MessageManager(models.Manager):
    def for_thread(self, thread: models.Model) -> QuerySet:
        return self.filter(thread=thread)

    def by_role(self, role: str) -> QuerySet:
        return self.filter(result__role=role)

    def with_tools(self) -> QuerySet:
        return self.exclude(result__tool_calls=[])

    def user_messages(self) -> QuerySet:
        return self.by_role("user")

    def agent_messages(self) -> QuerySet:
        return self.by_role("assistant")

    def conversation_history(self, thread: models.Model, include_system: bool = False) -> QuerySet:
        queryset = self.filter(thread=thread).order_by("created_at")
        if not include_system:
            queryset = queryset.exclude(result__role="system")
        return queryset
