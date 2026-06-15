from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import IO, ClassVar, Protocol

import magic
from django.core.files.base import File
from django.core.files.uploadedfile import InMemoryUploadedFile, TemporaryUploadedFile

type FileSource = str | Path | File | IO[bytes]


class FileProcessor(Protocol):
    ALLOWED_MIME_TYPES: ClassVar[tuple[str, ...]] = ()

    def is_valid(self, file: FileSource) -> bool:
        pass

    def run(self, file: FileSource) -> str | None:
        pass


class BaseFileProcessor:
    """Shared MIME validation for concrete file processors."""

    ALLOWED_MIME_TYPES: ClassVar[tuple[str, ...]] = ()

    def is_valid(
        self,
        file: str | Path | BytesIO | InMemoryUploadedFile | TemporaryUploadedFile,
    ) -> bool:
        if isinstance(file, (str, Path)):
            mime_type = magic.from_file(file, mime=True)
        elif isinstance(file, TemporaryUploadedFile):
            mime_type = magic.from_file(file.temporary_file_path(), mime=True)
        else:
            mime_type = magic.from_buffer(file.read(), mime=True)
            file.seek(0)
        return mime_type in self.ALLOWED_MIME_TYPES

    def run(self, file: str | Path | File) -> str | None:
        raise NotImplementedError


class TextFileProcessor(BaseFileProcessor):
    ALLOWED_MIME_TYPES: ClassVar[tuple[str, ...]] = (
        "text/plain",
        "text/markdown",
        "text/x-markdown",
    )

    def run(self, file: str | Path | File) -> str | None:
        if isinstance(file, (str, Path)):
            with open(file, encoding="utf-8") as f:
                return f.read()
        file.seek(0)
        content = file.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8")
        return content


class CSVFileProcessor(BaseFileProcessor):
    """Returns raw CSV string. Use CSVParseTransform to convert to list[dict]."""

    ALLOWED_MIME_TYPES: ClassVar[tuple[str, ...]] = (
        "text/csv",
        "text/plain",
    )

    def run(self, file: str | Path | File) -> str | None:
        if isinstance(file, (str, Path)):
            with open(file, encoding="utf-8") as f:
                return f.read()
        file.seek(0)
        content = file.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8")
        return content


class JSONFileProcessor(BaseFileProcessor):
    """Returns raw JSON string. Use JSONParseTransform to convert to dict/list."""

    ALLOWED_MIME_TYPES: ClassVar[tuple[str, ...]] = (
        "application/json",
        "text/json",
    )

    def run(self, file: str | Path | File) -> str | None:
        if isinstance(file, (str, Path)):
            with open(file, encoding="utf-8") as f:
                return f.read()
        file.seek(0)
        content = file.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8")
        return content
