from __future__ import annotations

import csv
import io
import re
from functools import cache
from io import BytesIO
from typing import Any

from django_ai_sdk.logger import get_logger

logger = get_logger(__name__)

__all__ = [
    "IMAGE_EXTENSIONS",
    "MAX_QUOTES",
    "PAGE_DPI",
    "PREVIEW_EXTENSIONS",
    "QUOTE_LETTERS",
    "get_pdf_engine",
    "letters",
    "passage_areas",
    "quote_page",
    "render_page",
    "to_pdf",
]

QUOTE_LETTERS = (12, 80)
MAX_QUOTES = 8
QUOTE_CHARS = 400
PAGE_DPI = 110
HIGHLIGHT = (255, 255, 0)

# a word's box: x, y (from the bottom of the page), width, height, in points
Box = tuple[float, float, float, float]


@cache
def get_pdf_engine() -> Any:
    """With its logging on, converting an Office file deadlocks."""
    import pdf_oxide

    pdf_oxide.disable_logging()
    return pdf_oxide


def csv_to_xlsx(data: bytes) -> bytes:
    from openpyxl import Workbook

    workbook = Workbook()
    sheet = workbook.active
    for row in csv.reader(io.StringIO(data.decode("utf-8-sig", errors="replace"))):
        sheet.append(row)
    out = BytesIO()
    workbook.save(out)
    return out.getvalue()


def get_office(kind: str) -> Any:
    return lambda data: getattr(get_pdf_engine().OfficeConverter, f"from_{kind}_bytes")(data)


def get_markdown_pdf(text: str) -> bytes:
    from django_ai_sdk.files.documents import markdown_to_pdf

    return markdown_to_pdf(text)


def get_text(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


PDF_CONVERTERS_BY_EXTENSION = {
    "pdf": lambda data: data,
    "docx": get_office("docx"),
    "pptx": get_office("pptx"),
    "xlsx": get_office("xlsx"),
    "csv": lambda data: get_office("xlsx")(csv_to_xlsx(data)),
    "md": lambda data: get_markdown_pdf(get_text(data)),
    "txt": lambda data: get_pdf_engine().Pdf.from_text(get_text(data)),
    "json": lambda data: get_pdf_engine().Pdf.from_text(get_text(data)),
}

PREVIEW_EXTENSIONS = frozenset(PDF_CONVERTERS_BY_EXTENSION)
IMAGE_EXTENSIONS = frozenset({"png", "jpg", "jpeg", "webp", "gif"})


def to_pdf(data: bytes, extension: str) -> bytes | None:
    """The file as a PDF, or None if the extension is not supported."""
    convert = PDF_CONVERTERS_BY_EXTENSION.get(extension.lower().lstrip("."))
    if convert is None:
        return None
    pdf = convert(data)
    return pdf if isinstance(pdf, bytes) else pdf.to_bytes()


def letters(text: str) -> str:
    return re.sub(r"[\W_]+", "", (text or "").lower())


def clean_quotes(quotes: list[str]) -> list[str]:
    return [quote[:QUOTE_CHARS] for quote in quotes[:MAX_QUOTES]]


def quote_page(pdf: bytes, quotes: list[str]) -> int | None:
    shortest, longest = QUOTE_LETTERS
    needles = {
        needle
        for quote in clean_quotes(quotes)
        if len(needle := letters(quote)[:longest]) >= shortest
    }
    if not needles:
        return None
    best, best_hits = None, 0
    with get_pdf_engine().PdfDocument.from_bytes(pdf) as document:
        for number in range(1, document.page_count() + 1):
            text = letters(document.extract_text(number - 1))
            hits = sum(needle in text for needle in needles)
            if hits > best_hits:
                best, best_hits = number, hits
    return best


def passage_areas(words: list, quote: str) -> list[Box]:
    from django_ai_sdk.adapters.citations.grounding import get_locate_passage

    text, ranges = "", []
    for word in words:
        if text:
            text += " "
        ranges.append((len(text), len(text) + len(word.text)))
        text += word.text
    if not (found := get_locate_passage(text, quote)):
        return []
    start, end, _ = found
    return [
        word.bbox
        for word, (first, last) in zip(words, ranges, strict=True)
        if first < end and last > start
    ]


def render_page(pdf: bytes, page: int, quotes: list[str]) -> tuple[bytes, str] | None:
    from PIL import Image, ImageChops, ImageDraw

    quotes = clean_quotes(quotes)
    with get_pdf_engine().PdfDocument.from_bytes(pdf) as document:
        if not 1 <= page <= document.page_count():
            return None
        pixmap = document.render_pixmap(page - 1, dpi=PAGE_DPI)
        words = document.extract_words(page - 1) if quotes else []
    size = (pixmap.width, pixmap.height)

    image = Image.frombuffer("RGBa", size, pixmap.data).convert("RGB")
    scale = PAGE_DPI / 72
    boxes = [
        (
            x * scale,
            image.height - (y + height) * scale,
            (x + width) * scale,
            image.height - y * scale,
        )
        for quote in quotes
        for x, y, width, height in passage_areas(words, quote)
    ]
    mark = ""
    if boxes:
        marker = Image.new("RGB", size, "white")
        draw = ImageDraw.Draw(marker)
        for box in boxes:
            draw.rectangle(box, fill=HIGHLIGHT)
        image = ImageChops.multiply(image, marker)
        left, top, right, bottom = zip(*boxes, strict=True)
        mark = ",".join(
            f"{value:.4f}"
            for value in (
                min(top) / image.height,
                min(left) / image.width,
                max(bottom) / image.height,
                max(right) / image.width,
            )
        )
    png = BytesIO()
    image.save(png, "PNG")
    return png.getvalue(), mark
