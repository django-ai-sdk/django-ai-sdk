from unittest.mock import MagicMock

import pytest

from django_ai_sdk.files.processors import TextFileProcessor


class TestTextFileProcessor:
    @pytest.fixture
    def processor(self):
        return TextFileProcessor()

    @pytest.fixture
    def mock_puremagic_text(self, monkeypatch):
        monkeypatch.setattr(
            "django_ai_sdk.files.processors.puremagic.from_string",
            lambda stream, mime=True: "text/plain",
        )

    async def test_is_valid_for_txt_file(self, processor, tmp_path, mock_puremagic_text):
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Hello, world!")
        assert await processor.is_valid(str(txt_file)) is True

    async def test_is_valid_for_md_file(self, processor, tmp_path, mock_puremagic_text):
        md_file = tmp_path / "test.md"
        md_file.write_text("# Markdown Header")
        assert await processor.is_valid(str(md_file)) is True

    async def test_is_valid_returns_false_for_non_text(self, processor, tmp_path):
        png_file = tmp_path / "test.png"
        png_file.write_bytes(b"\x89PNG\r\n\x1a\n")
        assert await processor.is_valid(str(png_file)) is False

    async def test_run_reads_file_path(self, processor, tmp_path):
        txt_file = tmp_path / "test.txt"
        expected_content = "Hello, world!\nThis is a test."
        txt_file.write_text(expected_content)
        result = await processor.run(str(txt_file))
        assert result == expected_content

    async def test_run_reads_path_object(self, processor, tmp_path):
        txt_file = tmp_path / "test.txt"
        expected_content = "Path object test content."
        txt_file.write_text(expected_content)
        result = await processor.run(txt_file)
        assert result == expected_content

    async def test_run_reads_file_with_path_attr(self, processor, tmp_path):
        expected_content = "FieldFile-like test content."
        src = tmp_path / "source.txt"
        src.write_text(expected_content)

        file_like = MagicMock()
        file_like.path = str(src)
        result = await processor.run(file_like)
        assert result == expected_content

    async def test_run_raises_on_binary_content(self, processor, tmp_path):
        png_file = tmp_path / "test.png"
        png_file.write_bytes(b"\x89PNG\r\n\x1a\n")
        with pytest.raises(UnicodeDecodeError):
            await processor.run(str(png_file))

    async def test_is_valid_with_path_attr(self, processor, tmp_path, mock_puremagic_text):
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Hello, world!")

        file_like = MagicMock()
        file_like.path = str(txt_file)
        assert await processor.is_valid(file_like) is True
