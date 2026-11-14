"""The review API over the oversight record, which the HTTP layers are built on.

Optional: core never imports it. Every method checks the `oversight` permission domain and
its scope hooks (docs/content/manual/oversight.md, Review API).

`VIEW_OVERSIGHT_METRICS` shows a turn's columns, labels, answer values and rule matches. Its
content (question, answer, sources, reasons, reviewers' notes and who asked) also needs
`VIEW_OVERSIGHT_CONTENT` for that turn.

Lists are keyset-paged: a cursor is the sort values of a page's last row, base64-encoded
JSON. It is opaque to clients and survives rows being added in between.
"""

from __future__ import annotations

import base64
import binascii
import json
import re
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, cast
from uuid import UUID

from asgiref.sync import sync_to_async
from django.core.exceptions import ValidationError
from django.db import connections
from django.db.models import Count, DateTimeField, Exists, F, OuterRef, Q
from django.db.models.fields.json import KeyTransform
from django.utils import timezone

from django_ai_sdk.errors import UserError
from django_ai_sdk.oversight.content import expired, load_content
from django_ai_sdk.oversight.permissions import (
    CONTENT,
    METRICS,
    REVIEW,
    OversightAccess,
    reader_id,
)
from django_ai_sdk.oversight.registry import get_resolved_policy
from django_ai_sdk.oversight.schemas import (
    ContentState,
    LensAnswerOut,
    MetricsOut,
    Page,
    ResolveIn,
    ReviewOut,
    ReviewStatus,
    RuleCountOut,
    RuleMatchOut,
    TurnDetail,
    TurnFilter,
    TurnOut,
)
from django_ai_sdk.permissions import ConflictError, PermissionDenied
from django_ai_sdk.utils import as_uuid

if TYPE_CHECKING:
    from django.db.models import QuerySet

    from django_ai_sdk.oversight.models import Turn
    from django_ai_sdk.oversight.policy import ResolvedPolicy
    from django_ai_sdk.types import UserType

# Upper bound on rows per read, whatever `limit` asks for.
MAX_LIMIT = 100
# The longest window `get_metrics` counts over.
MAX_DAYS = 365
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


class OversightService(OversightAccess):
    """Read turns, work the review queues, count what happened."""

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
        from django_ai_sdk.oversight.models import ReviewItem, Turn

        await cls.has_perms(user, METRICS)
        limit = max(1, min(limit, MAX_LIMIT))
        filters = filters or TurnFilter()
        qs = await cls.scoped(user, METRICS, Turn.objects.all())
        # Asked once: a custom permission class may query for it.
        may_content = await cls.has_perms(user, CONTENT, raise_on_deny=False)
        if filters.thread_id:
            # A thread id links turns to a conversation, so it needs content access.
            if not may_content:
                raise PermissionDenied("Filtering by thread takes VIEW_OVERSIGHT_CONTENT.")
            qs = await cls.scoped(user, CONTENT, qs)
        qs = _filter_turns(qs, filters)
        if may_content:
            qs = qs.select_related("content")
        turns, next_cursor = await _keyset_page(qs, ("-created_at", "-pk"), limit, cursor)
        shown = await cls.content_ids(user, [t.id for t in turns], allowed=may_content)
        reviews = await cls.scoped(user, METRICS, ReviewItem.objects.all())
        rows = await _turn_rows(turns, shown, reviews)
        return Page[TurnOut](items=rows, limit=limit, next_cursor=next_cursor)

    @classmethod
    async def get_turn(cls, turn_id: str, *, user: UserType) -> TurnDetail | None:
        """Get one turn with its answers, matches and reviews. None outside the reader's scope."""
        from django_ai_sdk.oversight.models import LensAnswer, ReviewItem, Turn

        await cls.has_perms(user, METRICS)
        scoped = await cls.scoped(user, METRICS, Turn.objects.select_related("content"))
        pk = as_uuid(turn_id)
        turn = await scoped.filter(pk=pk).afirst() if pk else None
        if turn is None:
            return None
        may_content = await cls.has_perms(user, CONTENT, raise_on_deny=False)
        shown = turn.id in await cls.content_ids(user, [turn.id], allowed=may_content)
        content, state = await sync_to_async(_content_state)(turn) if shown else (None, "withheld")
        answers = [a async for a in LensAnswer.objects.filter(turn=turn)]
        scoped_reviews = await cls.scoped(user, METRICS, ReviewItem.objects.filter(turn=turn))
        reviews = [r async for r in scoped_reviews]
        [row] = await _turn_rows([turn], {turn.id} if shown else set(), scoped_reviews)
        texts = (content or {}).get("texts") or {}
        resolved = await get_resolved_policy(turn.policy)
        return TurnDetail(
            **row.model_dump(),
            content=visible_content(content),
            content_state=state,
            answers=[_answer_out(a, resolved, texts if shown else None) for a in answers],
            reviews=[_review_out(r, shown=shown) for r in reviews],
            error=turn.error,
            parent_turn_id=str(turn.parent_turn_id or ""),
            agent_fingerprint=turn.agent_fingerprint,
            reflect_state=turn.reflect_state,
            thread_id=turn.thread_id if shown else "",
            message_id=turn.message_id if shown else "",
        )

    @classmethod
    async def list_reviews(
        cls,
        *,
        user: UserType,
        queue: str = "",
        status: ReviewStatus = "open",
        limit: int = 50,
        cursor: str = "",
    ) -> Page[ReviewOut]:
        """List review items, oldest first.

        The turn's row is included only for readers with VIEW_OVERSIGHT_METRICS.
        Its content, and the reviewer's note, only where the reader may see it.
        """
        from django_ai_sdk.oversight.models import ReviewItem, Turn

        await cls.has_perms(user, REVIEW)
        limit = max(1, min(limit, MAX_LIMIT))
        with_rows = await cls.has_perms(user, METRICS, raise_on_deny=False)
        may_content = await cls.has_perms(user, CONTENT, raise_on_deny=False)
        related = ("turn", "turn__content") if may_content else ("turn",)
        qs = await cls.scoped(user, REVIEW, ReviewItem.objects.select_related(*related))
        if queue:
            qs = qs.filter(queue=queue)
        if status:
            qs = qs.filter(status=status)
        items, next_cursor = await _keyset_page(qs, ("created_at", "pk"), limit, cursor)
        turns = list({r.turn_id: r.turn for r in items}.values())
        shown = await cls.content_ids(user, [t.id for t in turns], allowed=may_content)
        rows = {}
        if with_rows:
            # A turn's row is metrics: only for the turns in the reader's METRICS scope.
            metric = Turn.objects.filter(pk__in=[t.id for t in turns])
            in_scope = {
                pk
                async for pk in (await cls.scoped(user, METRICS, metric)).values_list(
                    "pk", flat=True
                )
            }
            listed = [t for t in turns if t.id in in_scope]
            reviews = await cls.scoped(user, METRICS, ReviewItem.objects.all())
            rows = {row.id: row for row in await _turn_rows(listed, shown, reviews)}
        return Page[ReviewOut](
            items=[
                _review_out(r, rows.get(str(r.turn_id)), shown=r.turn_id in shown) for r in items
            ],
            limit=limit,
            next_cursor=next_cursor,
        )

    @classmethod
    async def list_queues(cls, *, user: UserType) -> dict[str, int]:
        """Count open review items per queue.

        `get_metrics` counts the same under VIEW_OVERSIGHT_METRICS. This one is for
        reviewers, who may lack that permission.
        """
        from django_ai_sdk.oversight.models import ReviewItem

        await cls.has_perms(user, REVIEW)
        qs = await cls.scoped(
            user, REVIEW, ReviewItem.objects.filter(status=ReviewItem.Status.OPEN)
        )
        return await _count_by(qs, "queue")

    @classmethod
    async def resolve_review(
        cls, item_id: str, answer: ResolveIn, *, user: UserType
    ) -> ReviewOut | None:
        """Close a review item with the reviewer's verdict, or dismiss it.

        The reviewer says whether the rule that queued the turn was right (`agree`), or
        dismisses the item. Returns None for an item outside the reader's scope. The item is
        claimed with a conditional update, so if anyone already closed it this raises
        ConflictError. Nobody reviews a turn of their own; a subagent's turn carries the actor
        of the turn that delegated it.
        """
        from django_ai_sdk.oversight.models import ReviewItem

        await cls.has_perms(user, REVIEW)
        if answer.dismiss == (answer.agree is not None):
            raise UserError("Say whether the rule was right (agree), or dismiss the item.")
        scoped = await cls.scoped(user, REVIEW, ReviewItem.objects.select_related("turn"))
        pk = as_uuid(item_id)
        item = await scoped.filter(pk=pk).afirst() if pk else None
        if item is None:
            return None
        me = reader_id(user)
        if me and item.turn.actor_id == me:
            raise PermissionDenied("Nobody reviews a turn of their own.")
        closed = {
            "status": ReviewItem.Status.DISMISSED if answer.dismiss else ReviewItem.Status.DONE,
            "resolved_at": timezone.now(),
            "resolved_by_id": me,
            "agree": answer.agree,
            "note": answer.note,
        }
        claimed = await ReviewItem.objects.filter(
            pk=item.pk, status=ReviewItem.Status.OPEN
        ).aupdate(**closed)
        if not claimed:
            raise ConflictError("This review is closed already.")
        for field, value in closed.items():
            setattr(item, field, value)
        shown = item.turn_id in await cls.content_ids(
            user, [item.turn_id], allowed=await cls.has_perms(user, CONTENT, raise_on_deny=False)
        )
        return _review_out(item, shown=shown)

    @classmethod
    async def get_metrics(
        cls, *, user: UserType, days: int = 14, agent_id: str = "", policy: str = ""
    ) -> MetricsOut:
        """Count the last `days` of turns, optionally for one agent or policy.

        Everything is a SQL aggregate over the turns the reader may reach (`scope_turns`).
        """
        from django_ai_sdk.oversight.models import ReviewItem, RuleMatch, Turn

        await cls.has_perms(user, METRICS)
        since = timezone.now() - timedelta(days=max(1, min(days, MAX_DAYS)))
        narrow = {
            key: value for key, value in {"agent_id": agent_id, "policy": policy}.items() if value
        }
        turns = await cls.scoped(
            user, METRICS, Turn.objects.filter(created_at__gte=since, **narrow)
        )
        # Open reviews are counted regardless of age.
        matches = RuleMatch.objects.filter(turn__in=turns.values("pk"))
        waiting = await cls.scoped(
            user,
            METRICS,
            ReviewItem.objects.filter(
                status=ReviewItem.Status.OPEN,
                **{f"turn__{key}": value for key, value in narrow.items()},
            ),
        )
        return MetricsOut(
            since=since,
            turns=await turns.acount(),
            by_outcome=await _count_by(turns, "outcome"),
            by_report_outcome=await _count_by(
                turns.exclude(report_outcome="").exclude(report_outcome=F("outcome")),
                "report_outcome",
            ),
            by_channel=await _count_by(turns, "channel"),
            by_agent=await _count_by(turns, "agent_id"),
            rules=[
                RuleCountOut(**row)
                async for row in matches.order_by("rule", "mode")
                .values("rule", "mode")
                .annotate(matches=Count("id"), acted=Count("id", filter=Q(acted=True)))
            ],
            queues=await _count_by(waiting, "queue"),
        )


async def _count_by(qs: QuerySet, field: str) -> dict[str, Any]:
    """Count the rows of `qs` per value of `field`."""
    return {
        row[field]: row["n"] async for row in qs.order_by().values(field).annotate(n=Count("pk"))
    }


async def _keyset_page(
    qs: QuerySet, order: tuple[str, ...], limit: int, cursor: str = ""
) -> tuple[list[Any], str]:
    """Return up to `limit` rows of `qs` in `order` after `cursor`, and the next cursor or ""."""
    fields = [key.lstrip("-") for key in order]
    if cursor:
        values = _decode_cursor(cursor, qs.model, fields)
        # Rows after the cursor: equal on the leading fields, past it on the next one.
        after = Q()
        for i, (key, name) in enumerate(zip(order, fields, strict=True)):
            past = "lt" if key.startswith("-") else "gt"
            equal = dict(zip(fields[:i], values, strict=False))
            after |= Q(**equal, **{f"{name}__{past}": values[i]})
        qs = qs.filter(after)
    rows = [row async for row in qs.order_by(*order)[: limit + 1]]
    if len(rows) <= limit:
        return rows, ""
    rows = rows[:limit]
    last = [getattr(rows[-1], name) for name in fields]
    plain = [
        v.isoformat() if isinstance(v, datetime) else str(v) if isinstance(v, UUID) else v
        for v in last
    ]
    return rows, base64.urlsafe_b64encode(json.dumps(plain).encode()).decode().rstrip("=")


def _decode_cursor(cursor: str, model: Any, fields: list[str]) -> list[Any]:
    try:
        raw = json.loads(base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)))
        if not isinstance(raw, list) or len(raw) != len(fields):
            raise ValueError(cursor)
        if not all(isinstance(value, str | int) for value in raw):
            raise TypeError(cursor)
        values = []
        for name, value in zip(fields, raw, strict=True):
            field = model._meta.pk if name == "pk" else model._meta.get_field(name)
            # Coerced here, so a tampered value is a bad cursor, not an error in the query.
            if isinstance(field, DateTimeField):
                moment = datetime.fromisoformat(value)
                values.append(moment if timezone.is_aware(moment) else timezone.make_aware(moment))
            else:
                values.append(field.to_python(value))
        return values
    except (ValueError, TypeError, binascii.Error, ValidationError) as exc:
        raise UserError("Not a cursor this list handed out.") from exc


def _filter_turns(qs: QuerySet, f: TurnFilter) -> QuerySet:
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
        qs = _filter_by_label(qs, f.label, f.value)
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


def _filter_by_label(qs: QuerySet, label: str, value: str) -> QuerySet:
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


async def _turn_rows(turns: list[Turn], shown: set[Any], reviews: QuerySet) -> list[TurnOut]:
    """Build feed rows for `turns`, with content only for ids in `shown`.

    `reviews` is the review items the reader's METRICS scope keeps, for the open counts.
    """
    from django_ai_sdk.oversight.models import ReviewItem, RuleMatch

    ids = [t.id for t in turns]
    matches: dict[Any, list[RuleMatchOut]] = {}
    open_reviews: dict[Any, int] = {}
    if ids:
        async for m in RuleMatch.objects.filter(turn_id__in=ids).order_by("created_at"):
            matches.setdefault(m.turn_id, []).append(RuleMatchOut.model_validate(m))
        open_items = reviews.filter(turn_id__in=ids, status=ReviewItem.Status.OPEN)
        async for turn_id in open_items.values_list("turn_id", flat=True):
            open_reviews[turn_id] = open_reviews.get(turn_id, 0) + 1
    return [
        _turn_out(
            t,
            load_content(t) if t.id in shown else None,
            matches.get(t.id, []),
            open_reviews.get(t.id, 0),
            shown=t.id in shown,
        )
        for t in turns
    ]


def _turn_out(
    turn: Turn,
    content: dict[str, Any] | None,
    matches: list[RuleMatchOut],
    reviews_open: int,
    *,
    shown: bool,
) -> TurnOut:
    text = (content or {}).get("shown_input") or (content or {}).get("input")
    preview = text[:PREVIEW_CHARS] if isinstance(text, str) else ""
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


def _content_state(turn: Turn) -> tuple[dict[str, Any] | None, ContentState]:
    """Load the turn's content and its state (see `TurnDetail.content_state`)."""
    row = turn.content
    if row is None:
        return None, "none"
    if expired(row):
        return None, "expired"
    content = load_content(turn)
    if content is None:
        return None, "none"
    return content, cast("ContentState", row.capture)


def visible_content(content: dict[str, Any] | None) -> dict[str, Any] | None:
    """Keep only the content keys a reader sees on the turn itself (`CONTENT_KEYS`)."""
    if content is None:
        return None
    return {key: value for key, value in content.items() if key in CONTENT_KEYS}


def _answer_out(
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


def _review_out(item: Any, row: TurnOut | None = None, *, shown: bool) -> ReviewOut:
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


__all__ = ["OversightService", "visible_content"]
