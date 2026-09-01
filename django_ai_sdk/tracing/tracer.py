from __future__ import annotations

import contextlib
import time
from typing import TYPE_CHECKING, Any

from django.conf import settings
from django.utils import timezone
from haystack.tracing import Span, Tracer
from haystack.tracing.utils import coerce_tag_value

from django_ai_sdk.logger import logger
from django_ai_sdk.tracing import context
from django_ai_sdk.tracing.models import Trace

if TYPE_CHECKING:
    from collections.abc import Generator


def _write(rows: list[Trace]) -> None:
    """Persist a whole trace tree"""
    if not rows:
        return
    try:
        if context.on_event_loop():
            # Haystack's trace() is a sync contextmanager even under run_async
            context.schedule(Trace.objects.abulk_create(rows))
        else:
            Trace.objects.bulk_create(rows)
    except Exception as exc:
        logger.error("Trace write failed: {}", exc, exc_info=exc)


def _reply_meta(value: dict[str, Any]) -> dict[str, Any]:
    """Find the metadata dict of a component or agent output"""
    reply = value.get("replies") or value.get("last_message")
    if isinstance(reply, list):
        reply = reply[-1] if reply else None
    if isinstance(meta := getattr(reply, "meta", None), dict) and meta:
        return meta
    # Put it in a top-level meta, one entry per reply.
    meta = value.get("meta")
    if isinstance(meta, list):
        meta = meta[0] if meta else {}
    return meta if isinstance(meta, dict) else {}


class TelemetrySpan(Span):
    """A span, buffered as one ``Trace`` row"""

    def __init__(self, operation_name: str, parent_span: Span | None = None) -> None:
        self._t0 = time.monotonic()
        self._trace = Trace(operation_name=operation_name, started_at=timezone.now(), tags={})
        self._excluded = frozenset(getattr(settings, "AI_SDK_TRACING_EXCLUDED_TAGS", ()))

        if isinstance(parent_span, TelemetrySpan):
            parent = parent_span._trace
            self._trace.parent_id = parent.id
            self._trace.thread_id = parent.thread_id
            self._trace.message_id = parent.message_id
            self._root = parent_span._root
        else:
            # Root spans are created on the thread context
            ids = context.current() or {}
            self._trace.thread_id = ids.get("thread_id")
            self._trace.message_id = ids.get("message_id")
            self._root = self
            # Only a root buffers
            self._rows: list[Trace] = []

        self._root._rows.append(self._trace)

    def set_tag(self, key: str, value: Any) -> None:
        """Set a single tag on the span"""
        # Checked before coercion
        if key in self._excluded:
            return
        self._trace.tags[key] = coerce_tag_value(value)

    def set_content_tag(self, key: str, value: Any) -> None:
        """Harvest token usage"""
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
        # The Responses API names them input/output tokens, Chat Completions
        # prompt/completion; both fill the same columns.
        for field, aliases in (
            ("prompt_tokens", ("prompt_tokens", "input_tokens")),
            ("completion_tokens", ("completion_tokens", "output_tokens")),
            ("total_tokens", ("total_tokens",)),
        ):
            for alias in aliases:
                if isinstance(count := usage.get(alias), int):
                    setattr(self._trace, field, count)
                    break

    def _end(self) -> None:
        self._trace.ended_at = timezone.now()
        self._trace.duration_ms = (time.monotonic() - self._t0) * 1000
        if self._root is self:
            # Rows are in creation order, so parents precede their children
            _write(self._rows)


class DefaultTracer(Tracer):
    """Persists every span as a ``Trace``"""

    @contextlib.contextmanager
    def trace(
        self,
        operation_name: str,
        tags: dict[str, Any] | None = None,
        parent_span: Span | None = None,
    ) -> Generator[Span]:
        """Trace the execution of a block of code."""
        span = TelemetrySpan(operation_name, parent_span or context.current_span())
        if tags:
            span.set_tags(tags)
        try:
            with context.active_span(span):
                yield span
        finally:
            span._end()

    def current_span(self) -> Span | None:
        """Return the currently active span."""
        return context.current_span()

    async def aflush(self) -> None:
        """Wait until every scheduled span write"""
        await context.aflush()
