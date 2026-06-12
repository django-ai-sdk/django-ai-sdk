from pathlib import Path

from django.core.files.base import File


class FileService:
    @staticmethod
    async def process(file: str | Path | File) -> str | None:
        """Process file using the default pipeline. Returns content or None."""
        from django_ai_sdk.files.common import get_default_file_pipeline

        result = await get_default_file_pipeline().run(file)
        return result.content if result else None
