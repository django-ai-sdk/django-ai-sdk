from __future__ import annotations

from typing import TYPE_CHECKING

from django.db import models

if TYPE_CHECKING:
    import uuid

_TOKEN_SUMS = {
    "prompt_tokens": models.Sum("prompt_tokens"),
    "completion_tokens": models.Sum("completion_tokens"),
    "total_tokens": models.Sum("total_tokens"),
}


def _zeroed(totals: dict[str, int | None]) -> dict[str, int]:
    """Sum() yields None when nothing matched; callers want a number."""
    return {key: value or 0 for key, value in totals.items()}


class TraceQuerySet(models.QuerySet):
    """Query helpers for spans, chainable after any filter."""

    def for_thread(self, thread_id: str | uuid.UUID) -> TraceQuerySet:
        """Spans produced for one thread."""
        return self.filter(thread_id=thread_id)

    def for_message(self, message_id: str | uuid.UUID) -> TraceQuerySet:
        """Spans produced for one message's run."""
        return self.filter(message_id=message_id)

    def roots(self) -> TraceQuerySet:
        """Only the top span of each run."""
        return self.filter(parent__isnull=True)

    def llm_calls(self) -> TraceQuerySet:
        """Only the spans that wrap a single LLM call.

        Token counts are recorded on those *and* aggregated again onto the
        agent's own span, so summing every row would count each call twice (or
        three times, for an Agent inside a Pipeline). Leaf spans are exactly the
        LLM calls in every path — an agent's ``step.llm`` spans, a chat
        generator's ``component.run`` span — and never a rollup, which by
        definition has the spans it aggregates as children.
        """
        return self.filter(children__isnull=True)

    def token_usage(self) -> dict[str, int]:
        """Total prompt, completion and overall tokens across these spans."""
        return _zeroed(self.llm_calls().aggregate(**_TOKEN_SUMS))

    async def atoken_usage(self) -> dict[str, int]:
        """Total prompt, completion and overall tokens across these spans."""
        return _zeroed(await self.llm_calls().aaggregate(**_TOKEN_SUMS))


TraceManager = models.Manager.from_queryset(TraceQuerySet)
