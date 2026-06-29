from __future__ import annotations

import asyncio
import mimetypes
from functools import lru_cache
from pathlib import Path
from typing import IO, ClassVar, Protocol

import aiofiles
import filetype
from django.conf import settings
from django.core.files.base import File

type FileSource = str | Path | File | IO[bytes]


@lru_cache
def get_allowed_files() -> dict[str, str]:
    return getattr(settings, "AI_SDK_ALLOWED_FILES", {})


def get_file_name(file: FileSource) -> str | None:
    """Get filename"""
    if isinstance(file, (str, Path)):
        return str(file)
    elif hasattr(file, "name"):
        return str(file.name)

    return None


def get_mime_type(file: FileSource) -> str | None:
    name = get_file_name(file)

    # check for user defined types
    if name:
        ext = Path(name).suffix.lower()
        allowed_files = get_allowed_files()
        if ext in allowed_files:
            return allowed_files[ext]

    # check for magic bytes
    mime_type = filetype.guess_mime(file)
    if mime_type:
        return mime_type

    # check with std mimetypes package
    if name:
        mime, _ = mimetypes.guess_type(name)
        return mime

    return None


class FileProcessor(Protocol):
    ALLOWED_MIME_TYPES: ClassVar[tuple[str, ...]] = ()

    async def is_valid(self, file: FileSource) -> bool:
        pass

    async def run(self, file: FileSource) -> str | None:
        pass


class BaseFileProcessor:
    """Shared MIME validation for concrete file processors."""

    ALLOWED_MIME_TYPES: ClassVar[tuple[str, ...]] = ()

    async def is_valid(self, file: FileSource) -> bool:

        mime_type = get_mime_type(file)
        return mime_type in self.ALLOWED_MIME_TYPES

    async def run(self, file: FileSource) -> str | None:
        raise NotImplementedError


class TextFileProcessor(BaseFileProcessor):
    ALLOWED_MIME_TYPES: ClassVar[tuple[str, ...]] = (
        "text/plain",
        "text/markdown",
        "text/x-markdown",
    )

    async def run(self, file: FileSource) -> str | None:
        if isinstance(file, (str, Path)):
            async with aiofiles.open(file, encoding="utf-8") as f:
                return await f.read()
        await asyncio.to_thread(file.seek, 0)
        content = await asyncio.to_thread(file.read)
        if isinstance(content, bytes):
            content = content.decode("utf-8")
        return content


class CSVFileProcessor(TextFileProcessor):
    """Returns raw CSV string. Use CSVParseTransform to convert to list[dict]."""

    ALLOWED_MIME_TYPES: ClassVar[tuple[str, ...]] = (
        "text/csv",
        "text/plain",
    )


class JSONFileProcessor(TextFileProcessor):
    """Returns raw JSON string. Use JSONParseTransform to convert to dict/list."""

    ALLOWED_MIME_TYPES: ClassVar[tuple[str, ...]] = (
        "application/json",
        "text/json",
    )
