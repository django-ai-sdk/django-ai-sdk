from __future__ import annotations

import re

from django_ai_sdk.logger import get_logger

logger = get_logger(__name__)

_PATTERN = re.compile(
    # Canonical tag and bundled ids ("1,2").
    r"""<source\s+id=["']([\d,\s]+)["']\s*/>"""
    # Fullwidth brackets.
    r"|【(\d+)】"
    # Plain [N], skipping indexing (arr[2]), reference links ([text][1]) and links ([1](url)).
    r"|(?<![\]\w])\[(\d+)\](?![(\[:])"
)

# Citations never belong in code: inside inline `code` or ``` fences tags are removed.
_CODE_TOKEN = re.compile(r"`{3,}|`|\n")
_TAG = re.compile(r"""<source\s+id=["'][\d,\s]+["']\s*/>""")


def normalize_citations(
    text: str, valid_ids: set[int], code: str = ""
) -> tuple[str, set[int], str]:
    """
    Rewrite every recognized citation form outside code to ``<source id="N" />``.
    """

    cited: set[int] = set()

    def _replace(match: re.Match[str]) -> str:
        bundled, fullwidth, bracket = match.groups()
        raw = bundled or fullwidth or bracket
        ids = [int(part) for part in raw.split(",") if part.strip()]
        valid = [i for i in ids if i in valid_ids]
        if bracket or fullwidth:
            if not valid:
                return match.group(0)
        elif len(valid) < len(ids):
            logger.debug("Dropping citation ids not in this turn: {}", set(ids) - valid_ids)
        cited.update(valid)
        return " ".join(f'<source id="{i}" />' for i in valid)

    parts: list[str] = []
    pos = 0
    for token in _CODE_TOKEN.finditer(text):
        segment = text[pos : token.start()]
        parts.append(_TAG.sub("", segment) if code else _PATTERN.sub(_replace, segment))
        parts.append(token.group())
        pos = token.end()
        fence = len(token.group()) >= 3
        if token.group() == "\n":
            if code == "`":
                code = ""
        elif not code:
            code = "```" if fence else "`"
        elif code == "`" or fence:
            code = ""
    tail = text[pos:]
    parts.append(_TAG.sub("", tail) if code else _PATTERN.sub(_replace, tail))
    return "".join(parts), cited, code
