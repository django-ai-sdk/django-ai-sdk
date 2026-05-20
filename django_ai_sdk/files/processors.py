from io import BytesIO
from pathlib import Path
from typing import Protocol, TypeVar

import magic
from django.core.files.base import File
from django.core.files.uploadedfile import InMemoryUploadedFile, TemporaryUploadedFile
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class FileProcessor(Protocol):
    ALLOWED_MIME_TYPES: list[str] = []

    def is_valid(
        self, file: str | Path | BytesIO | InMemoryUploadedFile | TemporaryUploadedFile
    ) -> bool:
        """Check if file is valid"""
        if isinstance(file, (str, Path)):
            mime_type = magic.from_file(file, mime=True)
        elif isinstance(file, TemporaryUploadedFile):
            mime_type = magic.from_file(file.temporary_file_path(), mime=True)
        else:
            mime_type = magic.from_buffer(file.read(), mime=True)
            file.seek(0)

        return mime_type in self.ALLOWED_MIME_TYPES

    def run(
        self,
        file: str | Path | File,
        *,
        response_format: type[T] | None = None,
    ) -> T | str | None:
        pass


class ContentProcessor(Protocol):
    def run(
        self,
        content: str,
        *,
        response_format: type[T] | None = None,
    ) -> T | str | None: ...
