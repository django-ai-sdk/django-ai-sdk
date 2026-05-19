from pathlib import Path

from django_ai_sdk.files.common import get_default_content_handler, get_default_file_handler
from django_ai_sdk.files.handlers import ContentHandler, FileHandler


class FileService:
    @staticmethod
    async def process(
        file: str | Path,
        *,
        handler: FileHandler | None = None,
    ) -> str | None:

        # attach file handler
        current_handler = get_default_file_handler() if not handler else handler

        # processing from each file processor
        for processor in current_handler.get_file_processors():
            return processor.run(file)

    @staticmethod
    async def extract(
        content: str,
        *,
        handler: ContentHandler | None = None,
    ) -> str | None:

        # attach content handler
        current_handler = get_default_content_handler() if not handler else handler

        # processing from each content processor
        for processor in current_handler.get_content_processors():
            return processor.run(content)
