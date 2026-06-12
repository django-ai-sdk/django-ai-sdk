from io import BytesIO
from pathlib import Path

import magic
from django.core.files.base import File
from django.core.files.uploadedfile import InMemoryUploadedFile, TemporaryUploadedFile



class FileProcessor(Protocol):

    def is_valid(
    ) -> bool:
        if isinstance(file, (str, Path)):
            mime_type = magic.from_file(file, mime=True)
        elif isinstance(file, TemporaryUploadedFile):
            mime_type = magic.from_file(file.temporary_file_path(), mime=True)
        else:
            mime_type = magic.from_buffer(file.read(), mime=True)
            file.seek(0)
        return mime_type in self.ALLOWED_MIME_TYPES



        "text/plain",
        "text/markdown",
        "text/x-markdown",

        if isinstance(file, (str, Path)):
            with open(file, encoding="utf-8") as f:
                return f.read()

        file.seek(0)
        content = file.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8")
        return content
