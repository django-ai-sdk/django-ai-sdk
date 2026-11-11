"""Read turns: a filtered feed paged by cursor, or one turn in full.

`VIEW_OVERSIGHT_METRICS` shows a turn's columns, labels, answer values and rule
matches. Its content (question, answer, sources, reasons, reviewers' notes and
who asked) also needs `VIEW_OVERSIGHT_CONTENT` for that turn.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from asgiref.sync import sync_to_async
from django.db import connections
from django.db.models import Exists, OuterRef
from django.db.models.fields.json import KeyTransform
from django.utils import timezone

from django_ai_sdk.errors import UserError
from django_ai_sdk.oversight.content import load_content, load_contents
from django_ai_sdk.oversight.schemas import (
    LensAnswerOut,
    Page,
    ReviewOut,
    RuleMatchOut,
    TurnDetail,
    TurnFilter,
    TurnOut,
)
from django_ai_sdk.oversight.services.access import CONTENT, METRICS, OversightAccess
from django_ai_sdk.oversight.services.paging import keyset_page, page_size
from django_ai_sdk.permissions import PermissionDenied
from django_ai_sdk.utils import as_uuid

if TYPE_CHECKING:
    from django.db.models import QuerySet

    from django_ai_sdk.oversight.models import Turn
    from django_ai_sdk.oversight.policy import ResolvedPolicy
    from django_ai_sdk.types import UserType

PREVIEW_CHARS = 160
# Label filters become part of an ORM lookup, so only allow plain lens names.
LABEL = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,99}")
# Content keys shown on a turn. Answer texts are shown with their answer instead.
CONTENT_KEYS = frozenset(
    {
        "agent",
        "capture",
        "input",
        "output",
        "shown_input",
        "shown_output",
        "history",
        "sources",
        "tool_calls",
        "input_chars",
        "output_chars",
    }
)


class TurnReads(OversightAccess):
    @classmethod
    async def list_turns(
        cls,
        filters: TurnFilter | None = None,
        *,
        user: UserType,
        limit: int = 50,
        cursor: str = "",
    ) -> Page[TurnOut]:
        """List turns newest first. There is no total because paging is keyset-based."""
        from django_ai_sdk.oversight.models import Turn

        await cls.has_perms(user, METRICS)
        limit = page_size(limit)
        filters = filters or TurnFilter()
        qs = await cls.scoped(user, METRICS, Turn.objects.all())
        # Asked once: a custom permission class may query for it.
        may_content = await cls.has_perms(user, CONTENT, raise_on_deny=False)
        if filters.thread_id:
            # A thread id links turns to a conversation, so it needs content access.
            if not may_content:
                raise PermissionDenied("Filtering by thread takes VIEW_OVERSIGHT_CONTENT.")
            qs = await cls.scoped(user, CONTENT, qs)
        qs = _filtered(qs, filters)
        if may_content:
            qs = qs.select_related("content")
        turns, next_cursor = await keyset_page(qs, ("-created_at", "-pk"), limit, cursor)
        shown = await cls.content_ids(user, [t.id for t in turns], allowed=may_content)
        rows = await rows_of(turns, shown)
        return Page[TurnOut](items=rows, limit=limit, next_cursor=next_cursor)

    @classmethod
    async def get_turn(cls, turn_id: str, *, user: UserType) -> TurnDetail | None:
        """Get one turn with its answers, matches and reviews. None outside the reader's scope."""
        from django_ai_sdk.oversight.models import LensAnswer, ReviewItem, Turn
        from django_ai_sdk.oversight.resolve import aget_policy

        await cls.has_perms(user, METRICS)
        scoped = await cls.scoped(user, METRICS, Turn.objects.select_related("content"))
        pk = as_uuid(turn_id)
        turn = await scoped.filter(pk=pk).afirst() if pk else None
        if turn is None:
            return None
        may_content = await cls.has_perms(user, CONTENT, raise_on_deny=False)
        shown = turn.id in await cls.content_ids(user, [turn.id], allowed=may_content)
        content, state = await sync_to_async(content_state)(turn) if shown else (None, "withheld")
        answers = [a async for a in LensAnswer.objects.filter(turn=turn)]
        reviews = [r async for r in ReviewItem.objects.filter(turn=turn)]
        [row] = await rows_of([turn], {turn.id} if shown else set(), contents={turn.id: content})
        texts = (content or {}).get("texts") or {}
        resolved = await aget_policy(turn.policy)
        detail = TurnDetail(
            **row.model_dump(),
            content=visible_content(content),
            content_state=state,
            answers=[answer_out(a, resolved, texts if shown else None) for a in answers],
            reviews=[review_out(r, shown=shown) for r in reviews],
            error=turn.error,
            parent_turn_id=str(turn.parent_turn_id or ""),
            agent_fingerprint=turn.agent_fingerprint,
            reflect_state=turn.reflect_state,
            thread_id=turn.thread_id if shown else "",
            message_id=turn.message_id if shown else "",
        )
        return detail


def _filtered(qs: QuerySet, f: TurnFilter) -> QuerySet:
    """Apply `f` to `qs`. Rule and review filters use Exists so rows do not repeat."""
    from django_ai_sdk.oversight.models import ReviewItem, RuleMatch

    simple = {
        "outcome": f.outcome,
        "status": f.status,
        "agent_id": f.agent_id,
        "policy": f.policy,
        "severity": f.severity,
        "thread_id": f.thread_id,
    }
    qs = qs.filter(**{field: value for field, value in simple.items() if value})
    if f.report_outcome:
        # "Would have been X" only counts when the real outcome was not X.
        qs = qs.filter(report_outcome=f.report_outcome).exclude(outcome=f.report_outcome)
    if f.channel:
        qs = qs.filter(channel__startswith=f.channel)
    if f.label:
        qs = _with_label(qs, f.label, f.value)
    if f.rule:
        qs = qs.filter(Exists(RuleMatch.objects.filter(turn=OuterRef("pk"), rule=f.rule)))
    if f.review == "open":
        open_items = ReviewItem.objects.filter(turn=OuterRef("pk"), status=ReviewItem.Status.OPEN)
        qs = qs.filter(Exists(open_items))
    elif f.review in {"reviewed", "unreviewed"}:
        # Reviewed means a person closed a review.
        person = Exists(
            ReviewItem.objects.filter(turn=OuterRef("pk")).exclude(status=ReviewItem.Status.OPEN)
        )
        qs = qs.filter(person if f.review == "reviewed" else ~person)
    if f.since:
        qs = qs.filter(created_at__gte=f.since)
    if f.until:
        qs = qs.filter(created_at__lt=f.until)
    return qs


def _with_label(qs: QuerySet, label: str, value: str) -> QuerySet:
    """Filter turns that have `label`, optionally equal to `value`."""
    if not LABEL.fullmatch(label) or "__" in label:
        raise UserError("A label filter names a lens: letters, digits, - and _.")
    if value == "":
        return qs.filter(labels__has_key=label)
    if connections[qs.db].vendor == "postgresql":
        # Use containment so Postgres can use the GIN index instead of a scan.
        return qs.filter(labels__contains={label: _typed_value(value)})
    # A KeyTransform, so a label named like a lookup ("contains") stays a key.
    return qs.annotate(_label=KeyTransform(label, "labels")).filter(_label=_typed_value(value))


def _typed_value(value: str) -> Any:
    """Convert a query-string value to the JSON type labels store."""
    lowered = value.lower()
    if lowered in {"true", "yes"}:
        return True
    if lowered in {"false", "no"}:
        return False
    try:
        return float(value) if "." in value else int(value)
    except ValueError:
        return value


async def rows_of(
    turns: list[Turn],
    shown: set[Any],
    *,
    contents: dict[Any, dict[str, Any] | None] | None = None,
) -> list[TurnOut]:
    """Build feed rows for `turns`, with content only for ids in `shown`."""
    from django_ai_sdk.oversight.models import ReviewItem, RuleMatch

    ids = [t.id for t in turns]
    matches: dict[Any, list[RuleMatchOut]] = {}
    open_reviews: dict[Any, int] = {}
    if ids:
        async for m in RuleMatch.objects.filter(turn_id__in=ids).order_by("created_at"):
            matches.setdefault(m.turn_id, []).append(RuleMatchOut.model_validate(m))
        open_items = ReviewItem.objects.filter(turn_id__in=ids, status=ReviewItem.Status.OPEN)
        async for turn_id in open_items.values_list("turn_id", flat=True):
            open_reviews[turn_id] = open_reviews.get(turn_id, 0) + 1
    if contents is None:
        contents = await sync_to_async(load_contents)([t for t in turns if t.id in shown])
    return [
        row_of(
            t,
            contents.get(t.id) if t.id in shown else None,
            matches.get(t.id, []),
            open_reviews.get(t.id, 0),
            shown=t.id in shown,
        )
        for t in turns
    ]


def row_of(
    turn: Turn,
    content: dict[str, Any] | None,
    matches: list[RuleMatchOut],
    reviews_open: int,
    *,
    shown: bool,
) -> TurnOut:
    preview = None
    if content is not None:
        text = content.get("shown_input") or content.get("input")
        preview = text[:PREVIEW_CHARS] if isinstance(text, str) else None
    return TurnOut(
        id=str(turn.id),
        created_at=turn.created_at,
        status=turn.status,
        channel=turn.channel,
        agent_id=turn.agent_id,
        agent_name=turn.agent_name,
        policy=turn.policy,
        outcome=turn.outcome,
        report_outcome=turn.report_outcome,
        severity=turn.severity,
        labels=turn.labels or {},
        matches=matches,
        model=turn.model,
        tokens=turn.tokens,
        cost=turn.cost,
        latency_ms=turn.latency_ms,
        reviews_open=reviews_open,
        reflected=turn.reflect_state == turn.ReflectState.DONE,
        preview=preview,
        actor_id=turn.actor_id if shown else "",
    )


def content_state(turn: Turn) -> tuple[dict[str, Any] | None, str]:
    """Load the turn's content and its state (see `TurnDetail.content_state`)."""
    row = turn.content
    if row is None:
        return None, "none"
    if row.expires_at is not None and row.expires_at <= timezone.now():
        return None, "expired"
    content = load_content(turn)
    if content is None:
        return None, "none"
    return content, row.capture


def visible_content(content: dict[str, Any] | None) -> dict[str, Any] | None:
    """Keep only the content keys a reader sees on the turn itself (`CONTENT_KEYS`)."""
    if content is None:
        return None
    return {key: value for key, value in content.items() if key in CONTENT_KEYS}


def answer_out(
    row: Any, resolved: ResolvedPolicy | None, texts: dict[str, Any] | None
) -> LensAnswerOut:
    """Build one answer. `texts` adds its words from the content, keyed by answer id.

    Pass None for a reader who may not see the content.
    """
    lens = resolved.lenses.get(row.lens) if resolved else None
    kept: dict[str, Any] = (texts or {}).get(str(row.id)) or {}
    trail = row.trail
    said_trail = kept.get("trail")
    if isinstance(said_trail, list) and len(said_trail) == len(trail):
        trail = [{**step, "value": value} for step, value in zip(trail, said_trail, strict=True)]
    return LensAnswerOut(
        id=str(row.id),
        lens=row.lens,
        label=lens.display_label if lens else row.lens,
        kind=lens.answer.kind if lens else "",
        stage=row.stage,
        answerer=row.answerer,
        model=row.model,
        value=kept.get("value", row.value),
        confidence=row.confidence,
        error=row.error,
        cost=row.cost,
        latency_ms=row.latency_ms,
        created_at=row.created_at,
        trail=trail,
        rationale=kept.get("rationale", ""),
        spans=row.spans if texts is not None else [],
        items=kept.get("items") or {},
    )


def review_out(item: Any, row: TurnOut | None = None, *, shown: bool) -> ReviewOut:
    """Build one review item. `shown` is whether the reader may see the turn's content,
    which includes the reviewer's note."""
    return ReviewOut(
        id=str(item.id),
        turn_id=str(item.turn_id),
        queue=item.queue,
        rule=item.rule,
        status=item.status,
        created_at=item.created_at,
        resolved_at=item.resolved_at,
        resolved_by=item.resolved_by_id,
        agree=item.agree,
        note=item.note if shown else "",
        turn=row,
    )


__all__ = ["TurnReads", "visible_content"]
