from __future__ import annotations

from typing import TYPE_CHECKING

from django.conf import settings
from django.utils.module_loading import import_string

from django_ai_sdk.files.processors import TextFileProcessor

if TYPE_CHECKING:
    from django_ai_sdk.files.pipeline import FilePipeline


def get_default_file_pipeline(file: object | None = None) -> FilePipeline:
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
            if file is None or pipeline.accepts(file):
                return pipeline
    return FilePipeline(TextFileProcessor())
