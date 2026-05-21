from django.conf import settings
from django.utils.module_loading import import_string

from django_ai_sdk.files.handlers import ContentHandler, FileHandler


def get_default_content_handler() -> ContentHandler:
    """Load default handler for unmanager services"""
    try:
        return (
            import_string(settings.AI_SDK_MEMORY_CONTENT_HANDLER)() or DefaultMemoryContentHandler()
        )

    except Exception:
        return DefaultMemoryContentHandler()


def get_default_file_handler() -> FileHandler:
    """Load default handler for unamanged services"""
    try:
        return import_string(settings.AI_SDK_MEMORY_FILE_HANDLER)() or DefaultMemoryFileHandler()
    except Exception:
        return DefaultMemoryFileHandler()


class DefaultMemoryFileHandler(FileHandler):
    file_processors = []


class DefaultMemoryContentHandler(ContentHandler):
    file_processors = []
