from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING

from django.conf import DEFAULT_STORAGE_ALIAS
from django.core.files.storage import InvalidStorageError, Storage, storages
from django.utils.module_loading import import_string

from django_ai_sdk.files.processors import TextFileProcessor, get_allowed_files
from django_ai_sdk.utils import resolve_setting

if TYPE_CHECKING:
    import io

    from django.core.files.base import File as DjangoFile

    from django_ai_sdk.files.pipeline import FilePipeline


def compute_file_hash(file: bytes | io.IOBase | DjangoFile) -> str:
    """SHA-256 hex digest of file content.

    Resets the IO pointer to 0 after reading so the caller can reuse it.
    """
    hasher = hashlib.sha256()
    if isinstance(file, bytes):
        hasher.update(file)
    else:
        for chunk in iter(lambda: file.read(65536), b""):
            hasher.update(chunk)
        file.seek(0)
    return hasher.hexdigest()


@dataclass
class UploadSettings:
    max_upload_size: int
    allowed_mime_types: list[str] = field(default_factory=lambda: ["*/*"])


@lru_cache(maxsize=1)
def get_upload_settings() -> UploadSettings:

    max_size = resolve_setting("AI_SDK_MAX_UPLOAD_SIZE", 10 * 1024 * 1024)

    allowed: set[str] = set()
    allowed.update(get_allowed_files().values())

    pipeline_setting = resolve_setting("AI_SDK_MEMORY_FILE_PIPELINE")
    if pipeline_setting:
        paths = [pipeline_setting] if isinstance(pipeline_setting, str) else pipeline_setting
        for path in paths:
            try:
                pipeline = import_string(path)()
                allowed.update(pipeline.file_processor.ALLOWED_MIME_TYPES)
            except Exception:
                pass

    return UploadSettings(
        max_upload_size=max_size,
        allowed_mime_types=sorted(allowed) if allowed else ["*/*"],
    )


AI_SDK_STORAGE_ALIAS = "django_ai_sdk"


def get_entry_document_storage() -> Storage:
    try:
        return storages[AI_SDK_STORAGE_ALIAS]
    except InvalidStorageError:
        return storages[DEFAULT_STORAGE_ALIAS]


DEFAULT_ENTRY_DOCUMENT_UPLOAD_TO = "memories/documents/"


def get_entry_document_upload_to() -> str:
    return resolve_setting("AI_SDK_FILE_UPLOAD_TO", DEFAULT_ENTRY_DOCUMENT_UPLOAD_TO)


async def get_default_file_pipeline(file: object | None = None) -> FilePipeline:
    """When a list is given, the first pipeline whose accepts(file) is True is used.
    Falls back to TextFileProcessor with no transforms when no match found.
    """
    from django_ai_sdk.files.pipeline import FilePipeline

    pipelines = await get_default_file_pipelines(file)
    return pipelines[0] if pipelines else FilePipeline(TextFileProcessor())


async def get_default_file_pipelines(file: object | None = None) -> list[FilePipeline]:
    setting = resolve_setting("AI_SDK_MEMORY_FILE_PIPELINE")
    if not setting:
        return []
    paths = [setting] if isinstance(setting, str) else setting
    pipelines: list[FilePipeline] = []
    for path in paths:
        pipeline = import_string(path)()
        if file is None or await pipeline.accepts(file):
            pipelines.append(pipeline)
    return pipelines
