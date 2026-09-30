from __future__ import annotations

from django.apps import AppConfig


class UsersConfig(AppConfig):
    """Demo user management. Hosts the email-only custom user model."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.users"
    verbose_name = "Users"
