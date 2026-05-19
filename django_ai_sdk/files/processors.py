from pathlib import Path
from typing import Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class FileProcessor(Protocol):
    def run(
        self,
        file: str | Path,
        *,
        response_format: type[T] | None = None,
    ) -> T | str: ...


class ContentProcessor(Protocol):
    def run(
        self,
        content: str,
        *,
        response_format: type[T] | None = None,
    ) -> T | str: ...
