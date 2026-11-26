from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from django.db import DataError, OperationalError
from django.db.models import BooleanField, F, Func

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from django.db.models import Model, QuerySet


def get_pattern(pattern: str, *, ignore_case: bool = True) -> re.Pattern[str]:
    flags = re.IGNORECASE if ignore_case else 0
    try:
        return re.compile(pattern, flags)
    except re.error:
        return re.compile(re.escape(pattern), flags)


class LineRegex(Func):
    output_field = BooleanField()

    def __init__(self, expression: Any, pattern: str, *, ignore_case: bool = True) -> None:
        self.ignore_case = ignore_case
        self.pattern = pattern
        super().__init__(F(expression) if isinstance(expression, str) else expression)

    def as_postgresql(self, compiler: Any, connection: Any, **extra: Any) -> tuple[str, list]:
        text, params = compiler.compile(self.source_expressions[0])
        operator = "~*" if self.ignore_case else "~"
        return f"{text} {operator} %s", [*params, f"(?n){self.pattern}"]

    def as_sqlite(self, compiler: Any, connection: Any, **extra: Any) -> tuple[str, list]:  # ty: ignore[incompatible-method-override]
        text, params = compiler.compile(self.source_expressions[0])
        flags = "(?mi)" if self.ignore_case else "(?m)"
        return f"{text} REGEXP %s", [*params, f"{flags}{self.pattern}"]


REGEX_REJECTED = (
    "no such function: regexp",
    "user-defined function raised exception",
    "invalid regular expression",
)


def is_regex_rejected(error: DataError | OperationalError) -> bool:
    message = str(error).lower()
    return any(reason in message for reason in REGEX_REJECTED)


async def agrep(
    queryset: QuerySet, regex: re.Pattern[str], *, field: str = "content"
) -> AsyncIterator[tuple[Model, int, str]]:
    ignore_case = bool(regex.flags & re.IGNORECASE)
    try:
        rows = [
            row
            async for row in queryset.filter(
                LineRegex(field, regex.pattern, ignore_case=ignore_case)
            )
        ]
    except (DataError, OperationalError) as error:
        if not is_regex_rejected(error):
            raise
        rows = [row async for row in queryset]
    for row in rows:
        for number, line in enumerate((getattr(row, field) or "").splitlines(), 1):
            if regex.search(line):
                yield row, number, line
