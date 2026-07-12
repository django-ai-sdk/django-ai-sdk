from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from django_ai_sdk.assistant import Assistant
    from django_ai_sdk.files.processors import FileProcessor
    from django_ai_sdk.files.transforms import BaseTransform

OnStep = Callable[[str], Awaitable[None]]


@dataclass
class PipelineResult:
    content: str
    data: dict | list = field(default_factory=dict)


def parse_data(data: Any) -> dict | list:
    """Normalize transform output for storage"""
    if hasattr(data, "model_dump"):
        return data.model_dump()
    if isinstance(data, (dict, list)):
        return data
    return {}


class FilePipeline:
    """Composable file processing pipeline.

    Selects a file processor and chains transforms in sequence.
    Each assistant can declare its own pipelines; the first one
    whose processor accepts the uploaded file is used.

    Example::

        FilePipeline(
            CSVFileProcessor(),
            transforms=[CSVTransform(), LLMExtractTransform(response_format=MySchema)],
        )
    """

    def __init__(
        self,
        file_processor: FileProcessor,
        transforms: list[BaseTransform] | None = None,
    ) -> None:
        # SAFEGUARD: Accept both a class and an instance
        self.file_processor = (
            file_processor() if isinstance(file_processor, type) else file_processor
        )
        self.transforms: list[BaseTransform] = transforms or []

    async def accepts(self, file: Any) -> bool:
        return await self.file_processor.is_valid(file)

    async def run(
        self,
        file: Any,
        *,
        assistant: Assistant | None = None,
        on_step: OnStep | None = None,
    ) -> PipelineResult | None:
        """Run processor then all transforms in sequence.
        Processor runs in a thread to avoid blocking the event loop.

        ``on_step``, if given, is awaited with the processor's/each transform's
        ``step`` name right before it runs — steps with no ``step`` (None)
        are skipped. Lets callers report fine-grained progress (e.g. "ocr",
        "extracting") instead of just pipeline-level pending/done.
        """
        if not await self.accepts(file):
            return None

        if on_step and getattr(self.file_processor, "step", None):
            await on_step(self.file_processor.step)
        data: Any = await self.file_processor.run(file)
        if data is None:
            return None

        content = data

        for transform in self.transforms:
            if on_step and getattr(transform, "step", None):
                await on_step(transform.step)
            data = await transform.run(data, assistant=assistant)

        return PipelineResult(content=content, data=parse_data(data))
