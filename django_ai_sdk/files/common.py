from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING

from django.conf import settings
from django.utils.module_loading import import_string

from django_ai_sdk.files.processors import TextFileProcessor, get_allowed_files

if TYPE_CHECKING:
    import io

    from django.core.files.base import File as DjangoFile

    from django_ai_sdk.files.pipeline import FilePipeline


def compute_file_hash(file: bytes | io.IOBase | DjangoFile) -> str:
    """SHA-256 hex digest of file content.

    Accepts bytes or any binary IO object (UploadedFile, BytesIO, open()).
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
    """File upload constraints for the frontend."""

    max_upload_size: int
    allowed_mime_types: list[str] = field(default_factory=lambda: ["*/*"])


@lru_cache(maxsize=1)
def get_upload_settings() -> UploadSettings:
    """Collect upload limits from Django settings and file processors."""

    max_size = getattr(settings, "AI_SDK_MAX_UPLOAD_SIZE", 10 * 1024 * 1024)

    allowed: set[str] = set()
    allowed.update(get_allowed_files().values())

    pipeline_setting = getattr(settings, "AI_SDK_MEMORY_FILE_PIPELINE", None)
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


async def get_default_file_pipeline(file: object | None = None) -> FilePipeline:
    """Return a default FilePipeline for uploads without assistant context.

    Configurable via AI_SDK_MEMORY_FILE_PIPELINE setting:
      - single dotted path: "myapp.pipelines.get_my_pipeline"
      - list of dotted paths: ["myapp.pipelines.ocr_pipeline", "myapp.pipelines.text_pipeline"]

    Each path must be a zero-argument callable returning a FilePipeline.
    When a list is given, the first pipeline whose accepts(file) is True is used.
    Falls back to TextFileProcessor with no transforms when no match found.
    """
    from django_ai_sdk.files.pipeline import FilePipeline

    setting = getattr(settings, "AI_SDK_MEMORY_FILE_PIPELINE", None)
    if setting:
        paths = [setting] if isinstance(setting, str) else setting
        for path in paths:
            pipeline = import_string(path)()
            if file is None or await pipeline.accepts(file):
                return pipeline
    return FilePipeline(TextFileProcessor())
