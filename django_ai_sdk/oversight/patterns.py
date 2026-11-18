"""Built-in patterns (personal data, secrets, refusals), span finding and masking.

Personal data is checksum-verified where possible, keeping false positives low
enough for the gate. Every built-in pattern runs in linear time (bounded repeats).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping
from typing import Any

from django_ai_sdk.oversight.types import TextSpan

# A plain list, or a label of "", means the lens's own name.
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


def bsn_ok(value: str) -> bool:
    """Dutch BSN 11-check, the last digit weighted -1."""
    digits = re.sub(r"\D", "", value)
    if len(digits) == 8:
        digits = "0" + digits
    if len(digits) != 9 or digits == "000000000":
        return False
    weights = [9, 8, 7, 6, 5, 4, 3, 2, -1]
    return sum(int(d) * w for d, w in zip(digits, weights, strict=True)) % 11 == 0


# Letter pairs that make "1234 XX" a unit or word rather than a postcode, like
# "4096 MB" or "1984 IN". Dutch postcodes never use SA, SD or SS either.
_NOT_POSTCODE_LETTERS = (
    "SA SD SS KB MB GB TB PB OK AM PM AD BC CE HZ MS PX PT EM KM CM MM KG MG ML HP KW MW GW "
    "TV PC VS AN AS AT BE BY DO GO HE IF IN IS IT ME MY NO OF ON OR SO TO UP US WE"
).split()

# What may split the groups of a Dutch-written number (iban, bsn, phone_nl): a space, dash,
# dot or slash, or an invisible character.
_SEP = r"[\s\-./\u200b\u200c\u200d\u2060\ufeff]"

# An address spelled out to get past a filter: "jan [at] example [dot] nl", "jan (at) example.nl",
# "jan apenstaartje example punt nl". A bare "at" and "dot" are left alone: they are prose too.
# It starts where a run of name characters starts, and no run backtracks, so it stays linear.
_AT = r"\s{0,3}(?:[\[({<]\s{0,3}(?i:at|@)\s{0,3}[\])}>]|\s(?i:apenstaartje)\s)\s{0,3}"
_DOT = r"(?:\.|\s{0,3}(?:[\[({<]\s{0,3}(?i:dot|punt|\.)\s{0,3}[\])}>]|\s(?i:dot|punt)\s)\s{0,3})"
_SPELLED_EMAIL = (
    rf"(?<![\w+-])[\w+-]{{1,64}}+(?:{_DOT}[\w+-]{{1,64}}+){{0,4}}"
    rf"{_AT}[\w-]{{1,63}}+(?:{_DOT}[\w-]{{1,63}}+){{1,8}}\b"
)

# Personal data by kind. Checksummed kinds come first, because when two kinds
# overlap the earlier one keeps the span. Matched case-sensitively.
PII_PATTERNS: dict[str, str] = {
    # Bounded repeats so a long run of "a." can't backtrack for seconds.
    # A full-width @ too: it reads the same.
    "email": rf"\b[\w.+-]{{1,64}}[@\uff20][\w-]{{1,63}}(?:\.[\w-]{{1,63}}){{1,8}}\b|{_SPELLED_EMAIL}",
    # Any case, groups split by spaces, dashes, dots or invisible characters; mod-97 decides.
    "iban": rf"\b[A-Za-z]{{2}}\d{{2}}(?:{_SEP}?[A-Za-z0-9]{{4}}){{2,7}}(?:{_SEP}?[A-Za-z0-9]{{1,4}})?\b",
    "credit_card": r"\b(?:\d[ -]?){12,18}\d\b",
    # Nine digits, whole or in groups (1234.56.782, 111-222-333); the 11-check decides.
    # Not \b: a BSN glued to its label ("bsn111222333") has no word boundary before it.
    "bsn": rf"(?<!\d)(?:\d{{9}}|\d{{4}}{_SEP}\d{{2}}{_SEP}\d{{3}}|\d{{3}}{_SEP}\d{{3}}{_SEP}\d{{3}})(?!\d)",
    # International format, with a country code after + or 00.
    "phone": r"(?<![\w+])(?:\+|00)\d{1,3}[\s-]?(?:\(0\)[\s-]?)?\d(?:[\s-]?\d){6,11}\b",
    # Dutch national format: ten digits starting with 0, in any grouping (06-12345678,
    # 06 12 34 56 78, 06/12345678, (06) 12345678, 020 123 4567), but not a date and a
    # time (03-11-2026 14:00).
    "phone_nl": rf"(?<![\w+])(?!\d\d([-./])\d\d\1\d{{4}}\b)"
    rf"(?:\(0\d{{1,3}}\)\s?(?:{_SEP}?\d){{6,8}}|0(?:{_SEP}?\d){{9}})(?!\d)",
    "ip_address": r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b",
    "postcode_nl": rf"\b[1-9]\d{{3}}\s?(?!(?:{'|'.join(_NOT_POSTCODE_LETTERS)})\b)[A-Z]{{2}}\b",
}
PII_VERIFY: dict[str, Verifier] = {"iban": iban_ok, "credit_card": luhn_ok, "bsn": bsn_ok}
# What, right before a match, makes it something else: "invoice no. 202611188" is an invoice
# number, though one in eleven such numbers passes the BSN 11-check. Only one of these labels
# directly before the number counts, so "factuur 111222333" leaves a real BSN unmasked: that
# is the trade for not masking every eleventh invoice or order number.
PII_NOT_AFTER: dict[str, str] = {
    "bsn": r"(?i)\b(?:invoice|factuur|order|bestel|bestelling|referentie|reference|kenmerk)"
    r"(?:nummer|nr|number)?(?:\s*(?:no|nr|number|nummer|#)\.?)?\s*[:#]?\s*$",
}
DEFAULT_PII_KINDS: tuple[str, ...] = ("email", "iban", "credit_card", "phone", "ip_address")
# Used only when named.
COUNTRY_PII_KINDS: tuple[str, ...] = ("bsn", "phone_nl", "postcode_nl")

# A password value needs a digit, an inner symbol, or quotes. This skips prose like
# "my password: settings or profile?". A quoted one may hold spaces, as a passphrase does.
_PASSWORD_VALUE = (
    r"""(?:(["'])[^"'\n]{6,128}\1|(?=\S*?(?:\d|[!@#$%^&*_+=~|\\/<>{}\[\]-]\S))\S{6,})"""
)
# The body runs to the END line, or to the end of the text when that is cut off.
_PEM_LINE = r"-----{} [A-Z ]{{0,24}}PRIVATE KEY(?: BLOCK)?-----"

# Matched case-sensitively.
SECRET_PATTERNS: dict[str, str] = {
    # sk-..., sk-proj-..., sk-ant-... keys, shared by several model providers.
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
    # Also a JSON or dict key, like {"password": "..."} in tool-call arguments.
    "password_assignment": r"(?i)(?<![a-z0-9])(?:password|passwd|wachtwoord|pwd)[\"']?\s*[:=]\s*"
    + _PASSWORD_VALUE,
}

WORD = re.compile(r"[^\W\d_]+", re.UNICODE)

# English or Dutch. Matched case-insensitively.
REFUSAL_PATTERNS: list[str] = [
    r"\b(I can(?:no|')t help|I(?:'m| am) (?:not able|unable) to|I (?:won't|will not) "
    r"(?:help|provide)|sorry, (?:but )?I can(?:no|')t|ik kan (?:je )?(?:daar )?niet (?:mee )?helpen"
    r"|dat kan ik niet)"
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
    """Yield (start, end) for each match that passes `valid`.

    A failing match may have swallowed trailing text (an IBAN then "EUR"), so the
    longest shorter match ending at a space or hyphen that passes is used instead.
    """
    pos = 0
    while (match := pattern.search(text, pos)) is not None:
        start, end = match.span()
        pos = max(end, start + 1)
        if valid is not None and not valid(match.group()):
            end = _shorter(pattern, valid, text, start, end)
            if not end:
                continue
            # The leftover text may hold the next match.
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
    """Whether any built-in personal data kind (verified) or secret is in `text`."""
    return bool(regex_spans(_SECRETS, text)) or any(
        next(valid_spans(pattern, PII_VERIFY.get(kind), text), None) for kind, pattern in _PII
    )


def mask_secrets(text: str) -> str:
    if not text:
        return text
    spans = regex_spans(_SECRETS, text)
    return mask(text, spans) if spans else text


def mask_json(value: Any, change: Callable[[str], str]) -> Any:
    """Apply `change` to every string in a JSON value.

    A secret can span a key and its value ({"password": "..."}), so such a dict
    becomes its JSON text, passed to `change`.
    """
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
    """Mask every (spans, template) group in one pass, so offsets stay valid.

    Overlapping spans merge into the first one's label.
    """
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
