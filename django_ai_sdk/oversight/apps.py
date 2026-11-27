from __future__ import annotations

from typing import TYPE_CHECKING, Any

from django.apps import AppConfig
from django.conf import settings
from django.core.checks import Tags
from django.core.checks import register as register_check
from django.core.exceptions import ImproperlyConfigured
from django.core.signals import setting_changed
from django.db import transaction
from django.db.models.signals import post_delete
from django.utils.module_loading import autodiscover_modules

from django_ai_sdk.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = get_logger(__name__)


class OversightConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "django_ai_sdk.oversight"
    label = "django_ai_sdk_oversight"
    verbose_name = "Oversight"

    def ready(self) -> None:
        import django_ai_sdk.oversight.packs  # noqa: F401

        autodiscover_modules("oversight")

        post_delete.connect(
            forget_deleted_thread,
            sender="django_ai_sdk.Thread",
            dispatch_uid="django_ai_sdk.oversight.forget_deleted_thread",
        )
        post_delete.connect(
            forget_deleted_user,
            sender=settings.AUTH_USER_MODEL,
            dispatch_uid="django_ai_sdk.oversight.forget_deleted_user",
        )
        setting_changed.connect(
            drop_cached_policies, dispatch_uid="django_ai_sdk.oversight.settings_changed"
        )

        from django_ai_sdk.oversight.checks import (
            check_oversight,
            check_oversight_deploy,
            check_oversight_retention,
        )

        register_check(check_oversight)
        register_check(check_oversight_deploy, deploy=True)
        register_check(check_oversight_retention, Tags.database, deploy=True)


def forget_deleted_thread(sender: Any, instance: Any, **kwargs: Any) -> None:
    from django_ai_sdk.oversight.content import forget_thread
    from django_ai_sdk.oversight.models import Turn

    thread_id = str(instance.pk)
    try:
        with transaction.atomic():
            if Turn.objects.filter(thread_id=thread_id).exists():
                forget_thread(thread_id)
    except Exception:
        logger.exception("Oversight could not forget deleted thread {}", thread_id)


def forget_deleted_user(sender: Any, instance: Any, **kwargs: Any) -> None:
    from django_ai_sdk.oversight.content import forget_actor

    actor_id = str(instance.pk)
    try:
        with transaction.atomic():
            forget_actor(actor_id)
    except Exception:
        logger.exception(
            "Oversight could not forget deleted user {}; run "
            "`manage.py oversight_forget_actor {}` to try again",
            actor_id,
            actor_id,
        )


def drop_cached_policies(setting: str, **kwargs: Any) -> None:
    if setting == "AI_SDK_OVERSIGHT":
        from django_ai_sdk.oversight.registry import invalidate_policy

        invalidate_policy()


def require_apps(app_name: str, *needed: str) -> None:
    from django.apps import apps

    for name in needed:
        if not apps.is_installed(name):
            raise ImproperlyConfigured(f"{app_name} needs {name} in INSTALLED_APPS")


MODEL_PERMISSIONS: dict[Any, str] = {}
ROUTE_MODULES: list[str] = []


def extend(*, permissions: Mapping[Any, str] | None = None, routes: str = "") -> None:
    MODEL_PERMISSIONS.update(permissions or {})
    if routes and routes not in ROUTE_MODULES:
        ROUTE_MODULES.append(routes)
