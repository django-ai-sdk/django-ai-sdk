"""
Unit tests for TextFileProcessor.
"""

from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from django.core.files.uploadedfile import InMemoryUploadedFile, TemporaryUploadedFile

from django_ai_sdk.files.processors import TextFileProcessor


class TestTextFileProcessor:
    """Test suite for TextFileProcessor."""

    @pytest.fixture
    def processor(self):
        """Create TextFileProcessor instance."""
        return TextFileProcessor()

    def test_is_valid_for_txt_file(self, processor, tmp_path):
        """Test that .txt files are recognized as valid."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Hello, world!")
        assert processor.is_valid(str(txt_file)) is True

    def test_is_valid_for_md_file(self, processor, tmp_path):
        """Test that .md files are recognized as valid."""
        md_file = tmp_path / "test.md"
        md_file.write_text("# Markdown Header")
        assert processor.is_valid(str(md_file)) is True

    def test_is_valid_for_bytesio_txt(self, processor):
        """Test that BytesIO with text/plain content is valid."""
        content = b"Hello, world!"
        file_obj = BytesIO(content)
        assert processor.is_valid(file_obj) is True

    def test_is_valid_for_bytesio_md(self, processor):
        """Test that BytesIO with markdown content is valid."""
        content = b"# Markdown Header"
        file_obj = BytesIO(content)
        assert processor.is_valid(file_obj) is True

    def test_is_valid_returns_false_for_non_text(self, processor, monkeypatch):
        """Test that non-text files are rejected."""
        # Mock magic.from_buffer to return a non-text MIME type
        def mock_from_buffer(buffer, mime=True):
            return "image/png"
        monkeypatch.setattr("magic.from_buffer", mock_from_buffer)

        content = b"\x89PNG\r\n\x1a\n"
        file_obj = BytesIO(content)
        assert processor.is_valid(file_obj) is False

    def test_run_reads_file_path(self, processor, tmp_path):
        """Test reading content from a file path."""
        txt_file = tmp_path / "test.txt"
        expected_content = "Hello, world!\nThis is a test."
        txt_file.write_text(expected_content)
        result = processor.run(str(txt_file))
        assert result == expected_content

    def test_run_reads_path_object(self, processor, tmp_path):
        """Test reading content from a Path object."""
        txt_file = tmp_path / "test.txt"
        expected_content = "Path object test content."
        txt_file.write_text(expected_content)
        result = processor.run(txt_file)
        assert result == expected_content

    def test_run_reads_bytesio(self, processor):
        """Test reading content from BytesIO."""
        expected_content = "BytesIO test content."
        file_obj = BytesIO(expected_content.encode("utf-8"))
        result = processor.run(file_obj)
        assert result == expected_content

    def test_run_reads_in_memory_uploaded_file(self, processor):
        """Test reading content from InMemoryUploadedFile."""
        expected_content = "InMemoryUploadedFile test content."
        file_obj = BytesIO(expected_content.encode("utf-8"))
        uploaded_file = InMemoryUploadedFile(
            file=file_obj,
            field_name="file",
            name="test.txt",
            content_type="text/plain",
            size=len(expected_content),
            charset="utf-8",
        )
        result = processor.run(uploaded_file)
        assert result == expected_content

    def test_run_reads_temporary_uploaded_file(self, processor, tmp_path):
        """Test reading content from TemporaryUploadedFile."""
        expected_content = "TemporaryUploadedFile test content."
        temp_file = tmp_path / "tmp_test.txt"
        temp_file.write_text(expected_content)

        # Create a mock TemporaryUploadedFile
        uploaded_file = MagicMock(spec=TemporaryUploadedFile)
        uploaded_file.temporary_file_path.return_value = str(temp_file)
        uploaded_file.read.return_value = expected_content.encode("utf-8")

        # For is_valid, it needs to work with magic.from_file
        assert processor.is_valid(uploaded_file) is True

        result = processor.run(uploaded_file)
        assert result == expected_content

    def test_run_raises_on_binary_content(self, processor):
        """Test that run raises UnicodeDecodeError for binary content."""
        # run() is only called after is_valid() passes; if a caller ignores
        # is_valid() and feeds binary bytes, the decode error should propagate
        file_obj = BytesIO(b"\x89PNG\r\n\x1a\n")
        with pytest.raises(UnicodeDecodeError):
            processor.run(file_obj)
