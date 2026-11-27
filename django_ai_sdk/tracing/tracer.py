from __future__ import annotations

import contextlib
import time
from typing import TYPE_CHECKING, Any

from django.utils import timezone
from haystack.tracing import Span, Tracer
from haystack.tracing.utils import coerce_tag_value

from django_ai_sdk.agents.subagent import SUBAGENT_ID_TAG, SUBAGENT_NAME_TAG
from django_ai_sdk.logger import logger
from django_ai_sdk.pricing import token_counts
from django_ai_sdk.tracing import context
from django_ai_sdk.tracing.models import Trace
from django_ai_sdk.utils import resolve_setting

if TYPE_CHECKING:
    from collections.abc import Generator


def _write(rows: list[Trace]) -> None:
    if not rows:
        return
    try:
        if context.on_event_loop():
            # Haystack's trace() is a sync contextmanager even under run_async
            context.schedule(Trace.objects.abulk_create(rows))
        else:
            Trace.objects.bulk_create(rows)
    except Exception as exc:
        logger.opt(exception=exc).error("Trace write failed: {}", exc)


def _reply_meta(value: dict[str, Any]) -> dict[str, Any]:
    reply = value.get("replies") or value.get("last_message")
    if isinstance(reply, list):
        reply = reply[-1] if reply else None
    if isinstance(meta := getattr(reply, "meta", None), dict) and meta:
        return meta
    meta = value.get("meta")
    if isinstance(meta, list):
        meta = meta[0] if meta else {}
    return meta if isinstance(meta, dict) else {}


class TelemetrySpan(Span):
    def __init__(self, operation_name: str, parent_span: Span | None = None) -> None:
        self._t0 = time.monotonic()
        self._trace = Trace(operation_name=operation_name, started_at=timezone.now(), tags={})
        self._excluded = frozenset(resolve_setting("AI_SDK_TRACING_EXCLUDED_TAGS", ()))

        if isinstance(parent_span, TelemetrySpan):
            parent = parent_span._trace
            self._trace.parent_id = parent.id
            self._trace.thread_id = parent.thread_id
            self._trace.message_id = parent.message_id
            self._root = parent_span._root
        else:
            ids = context.current() or {}
            self._trace.thread_id = ids.get("thread_id")
            self._trace.message_id = ids.get("message_id")
            self._root = self
            self._rows: list[Trace] = []

        self._root._rows.append(self._trace)

    def set_tag(self, key: str, value: Any) -> None:
        if key == SUBAGENT_NAME_TAG and isinstance(value, str):
            self._trace.agent_name = value
        if key == SUBAGENT_ID_TAG and isinstance(value, str):
            self._trace.agent_id = value or None
        if key in self._excluded:
            return
        self._trace.tags[key] = coerce_tag_value(value)

    def set_content_tag(self, key: str, value: Any) -> None:
        self._get_usage(value)
        super().set_content_tag(key, value)

    def _get_usage(self, value: Any) -> None:
        if not isinstance(value, dict):
            return
        meta = _reply_meta(value)
        if model := meta.get("model"):
            self._trace.model_name = model
        # token_usage is the Agent's running total, exposed on its own output.
        usage = meta.get("usage") or value.get("token_usage") or {}
        if not isinstance(usage, dict):
            return
        prompt, completion = token_counts(usage)
        if prompt is not None:
            self._trace.prompt_tokens = prompt
        if completion is not None:
            self._trace.completion_tokens = completion
        if isinstance(total := usage.get("total_tokens"), int):
            self._trace.total_tokens = total

    def _end(self) -> None:
        self._trace.ended_at = timezone.now()
        self._trace.duration_ms = (time.monotonic() - self._t0) * 1000
        if self._root is self:
            _write(self._rows)


class DefaultTracer(Tracer):
    @contextlib.contextmanager
    def trace(
        self,
        operation_name: str,
        tags: dict[str, Any] | None = None,
        parent_span: Span | None = None,
    ) -> Generator[Span]:
        span = TelemetrySpan(operation_name, parent_span or context.current_span())
        if tags:
            span.set_tags(tags)
        try:
            with context.active_span(span):
                yield span
        finally:
            span._end()

    def current_span(self) -> Span | None:
        return context.current_span()

    async def aflush(self) -> None:
        await context.aflush()
