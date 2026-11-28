from __future__ import annotations

import json
import re
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping
from typing import Any

from django_ai_sdk.oversight.types import TextSpan

PatternTable = Mapping[str, str | Collection[str]] | Collection[str]
Verifier = Callable[[str], bool]


def luhn_ok(digits: str) -> bool:
    numbers = [int(d) for d in digits if d.isdigit()]
    if len(numbers) < 12:
        return False
    total = 0
    for index, digit in enumerate(reversed(numbers)):
        if index % 2:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def iban_ok(value: str) -> bool:
    """ISO 13616 mod-97."""
    compact = re.sub(r"[^A-Za-z0-9]", "", value).upper()
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{10,30}", compact):
        return False
    rearranged = compact[4:] + compact[:4]
    number = "".join(str(int(ch, 36)) for ch in rearranged)
    return int(number) % 97 == 1


_SEP = r"[\s\-./\u200b\u200c\u200d\u2060\ufeff]"

_AT = r"\s{0,3}[\[({<]\s{0,3}(?i:at|@)\s{0,3}[\])}>]\s{0,3}"
_DOT = r"(?:\.|\s{0,3}(?:[\[({<]\s{0,3}(?i:dot|\.)\s{0,3}[\])}>]|\s(?i:dot)\s)\s{0,3})"
_SPELLED_EMAIL = (
    rf"(?<![\w+-])[\w+-]{{1,64}}+(?:{_DOT}[\w+-]{{1,64}}+){{0,4}}"
    rf"{_AT}[\w-]{{1,63}}+(?:{_DOT}[\w-]{{1,63}}+){{1,8}}\b"
)

PII_PATTERNS: dict[str, str] = {
    "email": rf"\b[\w.+-]{{1,64}}[@\uff20][\w-]{{1,63}}(?:\.[\w-]{{1,63}}){{1,8}}\b|{_SPELLED_EMAIL}",
    "iban": rf"\b[A-Za-z]{{2}}\d{{2}}(?:{_SEP}?[A-Za-z0-9]{{4}}){{2,7}}(?:{_SEP}?[A-Za-z0-9]{{1,4}})?\b",
    "credit_card": r"\b(?:\d[ -]?){12,18}\d\b",
    "phone": r"(?<![\w+])(?:\+|00)\d{1,3}[\s-]?(?:\(0\)[\s-]?)?\d(?:[\s-]?\d){6,11}\b",
    "ip_address": r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b",
}
PII_VERIFY: dict[str, Verifier] = {"iban": iban_ok, "credit_card": luhn_ok}
PII_NOT_AFTER: dict[str, str] = {}
DEFAULT_PII_KINDS: tuple[str, ...] = ("email", "iban", "credit_card", "phone", "ip_address")

_PASSWORD_VALUE = (
    r"""(?:(["'])[^"'\n]{6,128}\1|(?=\S*?(?:\d|[!@#$%^&*_+=~|\\/<>{}\[\]-]\S))\S{6,})"""
)
_PEM_LINE = r"-----{} [A-Z ]{{0,24}}PRIVATE KEY(?: BLOCK)?-----"

SECRET_PATTERNS: dict[str, str] = {
    "api_key": r"\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}\b",
    "stripe_key": r"\b(?:sk|rk)_live_[A-Za-z0-9]{20,}\b",
    "google_api_key": r"\bAIza[0-9A-Za-z_-]{35}(?![0-9A-Za-z_-])",
    "github_token": r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b",
    "aws_key": r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
    "slack_token": r"\bxox[abposr]-[A-Za-z0-9-]{10,}\b",
    "jwt": r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b",
    "private_key": _PEM_LINE.format("BEGIN")
    + r"(?:[\s\S]*?"
    + _PEM_LINE.format("END")
    + r"|[\s\S]*)",
    "password_assignment": r"(?i)(?<![a-z0-9])(?:password|passwd|pwd)[\"']?\s*[:=]\s*"
    + _PASSWORD_VALUE,
}

WORD = re.compile(r"[^\W\d_]+", re.UNICODE)

REFUSAL_PATTERNS: list[str] = [
    r"\b(I can(?:no|')t help|I(?:'m| am) (?:not able|unable) to|I (?:won't|will not) "
    r"(?:help|provide)|sorry, (?:but )?I can(?:no|')t)"
]


def compile_patterns(patterns: PatternTable, flags: int = 0) -> list[tuple[str, re.Pattern[str]]]:
    groups = patterns.items() if isinstance(patterns, Mapping) else [("", patterns)]
    return [
        (label, re.compile(pattern, flags))
        for label, group in groups
        for pattern in ([group] if isinstance(group, str) else group)
    ]


def regex_spans(
    patterns: Iterable[tuple[str, re.Pattern[str]]], text: str, default_label: str = ""
) -> list[TextSpan]:
    return [
        TextSpan(start=match.start(), end=match.end(), label=label or default_label)
        for label, pattern in patterns
        for match in pattern.finditer(text)
    ]


def valid_spans(
    pattern: re.Pattern[str], valid: Verifier | None, text: str
) -> Iterator[tuple[int, int]]:
    pos = 0
    while (match := pattern.search(text, pos)) is not None:
        start, end = match.span()
        pos = max(end, start + 1)
        if valid is not None and not valid(match.group()):
            end = _shorter(pattern, valid, text, start, end)
            if not end:
                continue
            pos = end
        yield start, end


def _shorter(pattern: re.Pattern[str], valid: Verifier, text: str, start: int, end: int) -> int:
    for cut in range(end - 1, start, -1):
        if text[cut] not in " \t\n-" or not text[cut - 1].isalnum():
            continue
        if pattern.fullmatch(text, start, cut) and valid(text[start:cut]):
            return cut
    return 0


_SECRETS = compile_patterns(SECRET_PATTERNS)
_PII = compile_patterns(PII_PATTERNS)


def finds_personal_or_secret(text: str) -> bool:
    return bool(regex_spans(_SECRETS, text)) or any(
        next(valid_spans(pattern, PII_VERIFY.get(kind), text), None) for kind, pattern in _PII
    )


def register_pii_kind(
    name: str, pattern: str, *, verify: Verifier | None = None, not_after: str | None = None
) -> None:
    """Add a personal data kind that a PIILens may name in `kinds`, in code or in a stored definition.

    `verify` drops a match that fails it, such as a checksum. `not_after` is a regex for the
    text right before a match that makes it no match, such as an invoice label. Register
    before the first turn: in the app's `oversight` module or in `AppConfig.ready()`.
    """
    PII_PATTERNS[name] = pattern
    _PII.extend(compile_patterns({name: pattern}))
    if verify:
        PII_VERIFY[name] = verify
    if not_after:
        PII_NOT_AFTER[name] = not_after


def mask_secrets(text: str) -> str:
    if not text:
        return text
    spans = regex_spans(_SECRETS, text)
    return mask(text, spans) if spans else text


def mask_json(value: Any, change: Callable[[str], str]) -> Any:
    if isinstance(value, dict):
        changed = {k: mask_json(v, change) for k, v in value.items()}
        text = json.dumps(changed, ensure_ascii=False, default=str)
        return changed if mask_secrets(text) == text else change(text)
    if isinstance(value, list | tuple):
        return [mask_json(item, change) for item in value]
    return change(value) if isinstance(value, str) else value


def mask(text: str, spans: Iterable[TextSpan], template: str = "[{label}]") -> str:
    return mask_all(text, [(spans, template)])


def mask_all(text: str, groups: Iterable[tuple[Iterable[TextSpan], str]]) -> str:
    """Overlapping spans merge into the first one's label."""
    ordered = sorted(
        ((span, template) for spans, template in groups for span in spans),
        key=lambda item: (item[0].start, -item[0].end),
    )
    out: list[str] = []
    cursor = 0
    for span, template in ordered:
        start, end = max(span.start, 0), min(span.end, len(text))
        if end <= cursor or start >= end:
            continue
        if start < cursor:
            cursor = end
            continue
        out.append(text[cursor:start])
        out.append(template.format(label=(span.label or "masked").upper()))
        cursor = end
    out.append(text[cursor:])
    return "".join(out)
