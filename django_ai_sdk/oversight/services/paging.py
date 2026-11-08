"""Keyset paging shared by the turns feed and the review queue.

A cursor is the sort values of a page's last row, base64-encoded JSON. It is opaque
to clients and survives rows being added in between.
"""

from __future__ import annotations

import base64
import binascii
import json
from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from django.core.exceptions import ValidationError
from django.db.models import DateTimeField, Q
from django.utils import timezone

from django_ai_sdk.errors import UserError

if TYPE_CHECKING:
    from django.db.models import QuerySet

# Upper bound on rows per read, whatever `limit` asks for.
MAX_LIMIT = 100


def page_size(limit: int) -> int:
    return max(1, min(limit, MAX_LIMIT))


async def keyset_page(
    qs: QuerySet, order: tuple[str, ...], limit: int, cursor: str = ""
) -> tuple[list[Any], str]:
    """Return up to `limit` rows of `qs` in `order` after `cursor`, and the next cursor or ""."""
    fields = [key.lstrip("-") for key in order]
    if cursor:
        values = _decode(cursor, qs.model, fields)
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


def cursor_query(cursor: str) -> dict[str, str]:
    """The cursor as it goes into an access record's query: absent on a first page."""
    return {"cursor": cursor} if cursor else {}


def _decode(cursor: str, model: Any, fields: list[str]) -> list[Any]:
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
            values.append(
                _moment(value) if isinstance(field, DateTimeField) else field.to_python(value)
            )
        return values
    except (ValueError, TypeError, binascii.Error, ValidationError) as exc:
        raise UserError("Not a cursor this list handed out.") from exc


def _moment(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed if timezone.is_aware(parsed) else timezone.make_aware(parsed)


__all__ = ["MAX_LIMIT", "cursor_query", "keyset_page", "page_size"]
