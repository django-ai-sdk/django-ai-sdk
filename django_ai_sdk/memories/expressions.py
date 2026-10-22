from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from django.db import DatabaseError
from django.db.models import BooleanField, F, Func

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from django.db.models import Model, QuerySet


def get_pattern(pattern: str, *, ignore_case: bool = True) -> re.Pattern[str]:
    """`pattern` as a regex"""
    flags = re.IGNORECASE if ignore_case else 0
    try:
        return re.compile(pattern, flags)
    except re.error:
        return re.compile(re.escape(pattern), flags)


class LineRegex(Func):
    """True where a line of `expression` matches `pattern`, decided in the database."""

    output_field = BooleanField()

    def __init__(self, expression: Any, pattern: str, *, ignore_case: bool = True) -> None:
        self.ignore_case = ignore_case
        self.pattern = pattern
        super().__init__(F(expression) if isinstance(expression, str) else expression)

    def as_postgresql(self, compiler: Any, connection: Any, **extra: Any) -> tuple[str, list]:
        text, params = compiler.compile(self.source_expressions[0])
        operator = "~*" if self.ignore_case else "~"
        return f"{text} {operator} %s", [*params, f"(?n){self.pattern}"]

    def as_sqlite(self, compiler: Any, connection: Any, **extra: Any) -> tuple[str, list]:
        text, params = compiler.compile(self.source_expressions[0])
        flags = "(?mi)" if self.ignore_case else "(?m)"
        return f"{text} REGEXP %s", [*params, f"{flags}{self.pattern}"]


async def agrep(
    queryset: QuerySet, regex: re.Pattern[str], *, field: str = "content"
) -> AsyncIterator[tuple[Model, int, str]]:
    """Each line of `field` that `regex` matches."""

    ignore_case = bool(regex.flags & re.IGNORECASE)
    try:
        rows = [
            row
            async for row in queryset.filter(
                LineRegex(field, regex.pattern, ignore_case=ignore_case)
            )
        ]
    except DatabaseError:
        # fallback, process in python if the database doesn't support regex
        rows = [row async for row in queryset]
    for row in rows:
        for number, line in enumerate((getattr(row, field) or "").splitlines(), 1):
            if regex.search(line):
                yield row, number, line
