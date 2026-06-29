from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import IO, ClassVar, Protocol

import aiofiles
import puremagic
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


async def read_aio_bytes(
    file: FileSource,
) -> bytes | None:
    if hasattr(file, "path"):
        async with aiofiles.open(file.path, mode="rb") as f:
            return await f.read()

    elif isinstance(file, (str, Path)):
        async with aiofiles.open(file, mode="rb") as f:
            return await f.read()

    return None


async def read_aio_text(
    file: FileSource,
    encoding: str | None = None,
) -> str | None:
    if hasattr(file, "path"):
        async with aiofiles.open(file.path, encoding=encoding) as f:
            return await f.read()

    elif isinstance(file, (str, Path)):
        async with aiofiles.open(file, encoding=encoding) as f:
            return await f.read()

    return None


async def get_mime_type(file: FileSource) -> str | None:
    stream = await read_aio_bytes(file)

    if stream is not None:
        try:
            return puremagic.from_string(stream, mime=True)
        except puremagic.PureError:
            pass

    name = get_file_name(file)
    if name:
        extension = puremagic.ext_from_filename(name)
        allowed_files = get_allowed_files()
        if extension in allowed_files:
            return allowed_files[extension]

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
        mime_type = await get_mime_type(file)
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
        return await read_aio_text(file, encoding="utf-8")


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
