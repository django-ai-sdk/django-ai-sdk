from __future__ import annotations

from typing import TYPE_CHECKING, Any

from asgiref.sync import sync_to_async
from django.db import models

if TYPE_CHECKING:
    import uuid

# operation name of the span around a subagent run.
SUBAGENT_OPERATION = "django_ai_sdk.subagent.run"

_TOKEN_SUMS = {
    "prompt_tokens": models.Sum("prompt_tokens"),
    "completion_tokens": models.Sum("completion_tokens"),
    "total_tokens": models.Sum("total_tokens"),
}


def _zeroed(totals: dict[str, int | None]) -> dict[str, int]:
    """Sum() yields None when nothing matched; callers want a number."""
    return {key: totals.get(key) or 0 for key in _TOKEN_SUMS}


def _add_tokens(totals: dict[str, int], row: dict[str, Any]) -> None:
    """Accumulate one span's token columns into a running total."""
    for key in _TOKEN_SUMS:
        totals[key] += row.get(key) or 0


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
        """Only the spans that wrap a single LLM call."""
        return self.filter(children__isnull=True)

    def subagents(self, agent_id: str | uuid.UUID | None = None) -> TraceQuerySet:
        """The wrapper span around each subagent run, optionally one by id."""
        spans = self.filter(operation_name=SUBAGENT_OPERATION)
        return spans.filter(agent_id=agent_id) if agent_id is not None else spans

    def subagent_ids(self) -> list[str]:
        """Distinct subagent ids seen in these spans, sorted."""
        return sorted(
            str(agent_id)
            for agent_id in self.subagents()
            .exclude(agent_id__isnull=True)
            .values_list("agent_id", flat=True)
            .distinct()
        )

    async def asubagent_ids(self) -> list[str]:
        """Distinct subagent ids seen in these spans, sorted."""
        ids = self.subagents().exclude(agent_id__isnull=True).values_list("agent_id", flat=True)
        return sorted({str(agent_id) async for agent_id in ids})

    def subagent_names(self) -> list[str]:
        """Distinct subagent display names seen in these spans, sorted."""
        return sorted(
            self.subagents().exclude(agent_name="").values_list("agent_name", flat=True).distinct()
        )

    async def asubagent_names(self) -> list[str]:
        """Distinct subagent display names seen in these spans, sorted."""
        names = self.subagents().exclude(agent_name="").values_list("agent_name", flat=True)
        return sorted({name async for name in names})

    def subagent_usage(self) -> dict[str, dict[str, Any]]:
        """Token totals per subagent id, across each subagent's whole subtree."""
        # span id
        owners: dict[Any, tuple[str, str]] = {
            row["id"]: (str(row["agent_id"]), row["agent_name"])
            for row in self.subagents()
            .exclude(agent_id__isnull=True)
            .values("id", "agent_id", "agent_name")
        }
        if not owners:
            return {}

        totals: dict[str, dict[str, Any]] = {}
        # a nested subagent is both a start point and a child.
        seen: set[Any] = set(owners)
        frontier = owners
        while frontier:
            children = self.model.objects.filter(parent_id__in=list(frontier)).values(
                "id",
                "parent_id",
                "agent_id",
                "agent_name",
                "operation_name",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
            )
            next_frontier: dict[Any, tuple[str, str]] = {}
            for child in children:
                # a nested subagent owns its own subtree from here down.
                owner = (
                    (str(child["agent_id"]), child["agent_name"])
                    if child["operation_name"] == SUBAGENT_OPERATION and child["agent_id"]
                    else frontier[child["parent_id"]]
                )
                if child["id"] in seen:
                    continue
                seen.add(child["id"])
                next_frontier[child["id"]] = owner
                agent_id, agent_name = owner
                entry = totals.setdefault(agent_id, {"agent_name": agent_name, **_zeroed({})})
                _add_tokens(entry, child)
            frontier = next_frontier

        return totals

    async def asubagent_usage(self) -> dict[str, dict[str, Any]]:
        """Token totals per subagent id, across each subagent's whole subtree."""
        # ponytail: the walk is several dependent queries, so it runs in a thread
        return await sync_to_async(self.subagent_usage, thread_sensitive=True)()

    def token_usage(self) -> dict[str, int]:
        """Total prompt, completion and overall tokens across these spans."""
        return _zeroed(self.llm_calls().aggregate(**_TOKEN_SUMS))

    async def atoken_usage(self) -> dict[str, int]:
        """Total prompt, completion and overall tokens across these spans."""
        return _zeroed(await self.llm_calls().aaggregate(**_TOKEN_SUMS))


TraceManager = models.Manager.from_queryset(TraceQuerySet)
