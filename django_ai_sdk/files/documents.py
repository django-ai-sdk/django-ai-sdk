from __future__ import annotations

import html
import re
import textwrap

from django_ai_sdk.files.previews import get_pdf_engine

__all__ = ["markdown_to_pdf"]

WIDTH = 80
TABLE_RULE = re.compile(r"^\|?[\s:|-]+\|?$")
INLINE = re.compile(r"(\*\*|\*|`)(\S(?:.*?\S)?)\1")


def keep_lines(markdown: str, width: int) -> list[str]:
    lines: list[str] = []
    code = False
    for line in html.unescape(markdown).splitlines():
        if line.lstrip().startswith("```"):
            code = not code
        if code or line.startswith(("#", "```")):
            lines.append(line)
        elif re.fullmatch(r"\s*([-*_])(\s*\1){2,}\s*", line):
            lines.append("")
        elif line.lstrip().startswith("|"):
            if not TABLE_RULE.match(line.strip()):
                cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
                lines.append(INLINE.sub(r"\2", " · ".join(cell for cell in cells if cell)))
        elif not line.strip():
            # it draws every empty line
            if lines and lines[-1]:
                lines.append("")
        else:
            line = INLINE.sub(r"\2", line)
            indent = " " * (len(line) - len(line.lstrip()))
            item = re.match(r"\s*(?:[-*+]|\d+\.)\s+", line)
            follow = " " * len(item.group()) if item else indent
            fits = width * 4 // 5 if line.isupper() else width
            lines += textwrap.wrap(line, fits, subsequent_indent=follow)
    return lines


# ponytail: Markdown, prepared by keep_lines, until pdf_oxide's HTML+CSS pipeline
# wraps text (https://github.com/yfedoseev/pdf_oxide/issues/2082).
def markdown_to_pdf(markdown: str, *, title: str | None = None, width: int = WIDTH) -> bytes:
    pdf = get_pdf_engine().Pdf.from_markdown("\n".join(keep_lines(markdown, width)), title=title)
    return bytes(pdf.to_bytes())
