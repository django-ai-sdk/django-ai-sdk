from __future__ import annotations

from django.apps import AppConfig


class TracingConfig(AppConfig):
    """opt-in tracing app"""

    name = "django_ai_sdk.tracing"
    label = "django_ai_sdk_tracing"
    verbose_name = "Django AI SDK tracing"

    def ready(self) -> None:
        from haystack import tracing

        from django_ai_sdk.tracing.tracer import DefaultTracer

        tracing.enable_tracing(DefaultTracer())
