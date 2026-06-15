from __future__ import annotations

from typing import TYPE_CHECKING

from django.conf import settings
from django.utils.module_loading import import_string

from django_ai_sdk.files.processors import TextFileProcessor

if TYPE_CHECKING:
    from django_ai_sdk.files.pipeline import FilePipeline


def get_default_file_pipeline(file=None) -> FilePipeline:
    """Return a default FilePipeline for uploads without assistant context.

    Configurable via AI_SDK_MEMORY_FILE_PIPELINE setting (dotted path to a
    zero-argument callable that returns a FilePipeline).

    Fallback: TextFileProcessor with no transforms — Entry.data = {}.
    """
    from django_ai_sdk.files.pipeline import FilePipeline

    path = getattr(settings, "AI_SDK_MEMORY_FILE_PIPELINE", None)
    if path:
        return import_string(path)()
    return FilePipeline(TextFileProcessor())
