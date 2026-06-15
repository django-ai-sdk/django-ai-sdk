from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from django_ai_sdk.assistant import Assistant
    from django_ai_sdk.files.processors import FileProcessor
    from django_ai_sdk.files.transforms import BaseTransform


@dataclass
class PipelineResult:
    content: str
    data: dict = field(default_factory=dict)


class FilePipeline:
    """Composable file processing pipeline.

    Selects a file processor and chains transforms in sequence.
    Each assistant can declare its own pipelines; the first one
    whose processor accepts the uploaded file is used.

    Example::

        FilePipeline(
            CSVFileProcessor(),
            transforms=[CSVParseTransform(), LLMExtractTransform(response_format=MySchema)],
        )
    """

    def __init__(
        self,
        file_processor: FileProcessor,
        transforms: list[BaseTransform] | None = None,
    ) -> None:
        self.file_processor = file_processor
        self.transforms: list[BaseTransform] = transforms or []

    def accepts(self, file: Any) -> bool:
        return self.file_processor.is_valid(file)

    async def run(self, file: Any, *, assistant: Assistant | None = None) -> PipelineResult | None:
        """Run processor then all transforms in sequence.

        Calls accepts() first — returns None for unsupported file types.
        Passes assistant to transforms that need LLM access.
        Processor runs in a thread to avoid blocking the event loop.
        """
        if not self.accepts(file):
            return None

        data: Any = await asyncio.to_thread(self.file_processor.run, file)
        if data is None:
            return None

        for transform in self.transforms:
            data = await transform.run(data, assistant=assistant)

        content = (
            data if isinstance(data, str) else json.dumps(data, default=str, ensure_ascii=False)
        )
        if not isinstance(data, str) and hasattr(data, "model_dump"):
            structured = data.model_dump()
        elif isinstance(data, dict):
            structured = data
        else:
            structured = {}

        return PipelineResult(content=content, data=structured)
