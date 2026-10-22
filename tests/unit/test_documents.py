"""Markdown made into a PDF by pdf_oxide, its layout helped."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("pdf_oxide")

from django_ai_sdk.files.documents import keep_lines, markdown_to_pdf  # noqa: E402
from django_ai_sdk.files.previews import get_pdf_engine  # noqa: E402

MARKDOWN = """# Vertaling

Een alinea met **vet** en *schuin*: é ë ĳ € “quotes”, Łódź &amp; meer.

| Code | Omschrijving |
|---|---|
| A1 | Eerste rij |

- een
- twee
"""


def _document(markdown: str) -> Any:
    return get_pdf_engine().PdfDocument.from_bytes(markdown_to_pdf(markdown, title="T"))


def _text(document: Any, page: int = 0) -> str:
    return " ".join(document.extract_text(page).split())


def test_keeps_the_text() -> None:
    text = _text(_document(MARKDOWN))

    for words in (
        "Vertaling",
        "Een alinea met vet en schuin",
        "é ë ĳ € “quotes”, Łódź & meer",
        "Code · Omschrijving",
        "A1 · Eerste rij",
        "twee",
    ):
        assert words in text


def test_long_text_wraps_inside_the_page() -> None:
    words = _document("Een alinea die over meerdere regels loopt. " * 40).extract_words(0)

    assert max(x + width for x, _, width, _ in (w.bbox for w in words)) < 612
    assert len({round(w.bbox[1]) for w in words}) > 5


def test_long_text_runs_over_pages() -> None:
    assert _document("Een alinea tekst.\n\n" * 200).page_count() > 1


def test_a_list_item_wraps_under_its_text() -> None:
    lines = keep_lines("- " + "woord " * 30, 40)

    assert lines[0].startswith("- woord")
    assert all(line.startswith("  woord") for line in lines[1:])


def test_code_and_headings_are_left_as_they_are() -> None:
    code = "```\n" + "x = 1  # " + "lang " * 30 + "\n```"

    assert keep_lines(f"# {'Kop ' * 30}\n{code}", 40) == [f"# {'Kop ' * 30}", *code.split("\n")]


def test_empty_lines_are_one_between_blocks_and_none_on_top() -> None:
    assert keep_lines("\n\n\n# Kop\n\n\n\nTekst\n\n", 80) == ["# Kop", "", "Tekst", ""]
