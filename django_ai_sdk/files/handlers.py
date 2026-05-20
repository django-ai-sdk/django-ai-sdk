from typing import Protocol, TypeVar

from pydantic import BaseModel

from django_ai_sdk.files.processors import ContentProcessor, FileProcessor

T = TypeVar("T", bound=BaseModel)


class FileHandler(Protocol):
    file_processors: list[type[FileProcessor]] = []

    def get_file_processors(self) -> list[FileProcessor]:
        return [processor() for processor in self.file_processors]


class ContentHandler(Protocol):
    content_processors: list[type[ContentProcessor]] = []

    def get_content_processors(self) -> list[ContentProcessor]:
        return [processor() for processor in self.content_processors]
