from __future__ import annotations

from io import BytesIO
from unittest.mock import MagicMock

import pytest
from django_ai_sdk.files.processors import AnyDocFileProcessor


def _make_docx(text: str = "Hello from docx") -> bytes:
    from docx import Document

    doc = Document()
    doc.add_paragraph(text)
    buf = BytesIO()
    doc.save(buf)
    return buf.getvalue()


class TestAnyDocFileProcessor:
    @pytest.fixture
    def processor(self):
        return AnyDocFileProcessor()

    async def test_is_valid_docx(self, processor, tmp_path):
        file = tmp_path / "test.docx"
        file.write_bytes(_make_docx())
        assert await processor.is_valid(str(file)) is True

    async def test_is_valid_pdf(self, processor, tmp_path):
        file = tmp_path / "test.pdf"
        file.write_bytes(b"%PDF-1.4 fake pdf")
        assert await processor.is_valid(str(file)) is True

    async def test_is_valid_odt(self, processor, tmp_path):
        file = tmp_path / "test.odt"
        file.write_bytes(b"fake odt")
        assert await processor.is_valid(str(file)) is True

    async def test_is_valid_csv(self, processor, tmp_path):
        file = tmp_path / "test.csv"
        file.write_text("a,b,c")
        assert await processor.is_valid(str(file)) is True

    async def test_is_valid_epub(self, processor, tmp_path):
        file = tmp_path / "test.epub"
        file.write_bytes(b"fake epub")
        assert await processor.is_valid(str(file)) is True

    async def test_is_valid_rtf(self, processor, tmp_path):
        file = tmp_path / "test.rtf"
        file.write_bytes(b"{\\rtf1 fake}")
        assert await processor.is_valid(str(file)) is True

    async def test_is_valid_rejects_unsupported(self, processor, tmp_path):
        file = tmp_path / "test.png"
        file.write_bytes(b"\x89PNG\r\n\x1a\n")
        assert await processor.is_valid(str(file)) is False

    async def test_is_valid_case_insensitive(self, processor, tmp_path):
        file = tmp_path / "TEST.DOCX"
        file.write_bytes(_make_docx())
        assert await processor.is_valid(str(file)) is True

    async def test_is_valid_with_path_attr(self, processor, tmp_path):
        file = tmp_path / "test.docx"
        file.write_bytes(_make_docx())
        file_like = MagicMock()
        file_like.name = "test.docx"
        file_like.path = str(file)
        assert await processor.is_valid(file_like) is True

    async def test_is_valid_rejects_no_name(self, processor):
        file_like = MagicMock(spec=[])
        assert await processor.is_valid(file_like) is False

    async def test_run_from_path(self, processor, tmp_path):
        file = tmp_path / "test.docx"
        file.write_bytes(_make_docx("Hello from anydoc"))
        result = await processor.run(str(file))
        assert result is not None
        assert "Hello from anydoc" in result

    async def test_run_from_path_object(self, processor, tmp_path):
        file = tmp_path / "test.docx"
        file.write_bytes(_make_docx("Path object"))
        result = await processor.run(file)
        assert result is not None
        assert "Path object" in result

    async def test_run_from_fieldfile_like(self, processor, tmp_path):
        src = tmp_path / "source.docx"
        src.write_bytes(_make_docx("FieldFile content"))
        file_like = MagicMock()
        file_like.name = "source.docx"
        file_like.path = str(src)
        result = await processor.run(file_like)
        assert result is not None
        assert "FieldFile content" in result

    async def test_run_returns_none_on_convert_error(self, processor, tmp_path, monkeypatch):
        import anydoc

        file = tmp_path / "test.docx"
        file.write_bytes(b"not a real docx")

        def raise_error(data: bytes, fmt: str | None = None) -> str:
            raise anydoc.MalformedError("malformed")

        monkeypatch.setattr(anydoc, "to_markdown_bytes", raise_error)
        result = await processor.run(str(file))
        assert result is None

    async def test_run_returns_none_on_needs_ocr(self, processor, tmp_path, monkeypatch):
        import anydoc

        file = tmp_path / "scan.pdf"
        file.write_bytes(b"fake scanned pdf")

        def raise_error(data: bytes, fmt: str | None = None) -> str:
            raise anydoc.NeedsOcrError("needs ocr")

        monkeypatch.setattr(anydoc, "to_markdown_bytes", raise_error)
        result = await processor.run(str(file))
        assert result is None

    async def test_run_returns_none_on_encrypted(self, processor, tmp_path, monkeypatch):
        import anydoc

        file = tmp_path / "secret.pdf"
        file.write_bytes(b"fake encrypted pdf")

        def raise_error(data: bytes, fmt: str | None = None) -> str:
            raise anydoc.EncryptedError("encrypted")

        monkeypatch.setattr(anydoc, "to_markdown_bytes", raise_error)
        result = await processor.run(str(file))
        assert result is None

    async def test_step_is_set(self, processor):
        assert processor.step == "anydoc"
