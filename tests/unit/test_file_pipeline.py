from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from django_ai_sdk.files.pipeline import FilePipeline, parse_data
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# parse_data
# ---------------------------------------------------------------------------


class _FakePydantic(BaseModel):
    summary: str
    keywords: list[str] = []


class TestParseData:
    def test_pydantic_model(self):
        obj = _FakePydantic(summary="hello", keywords=["a", "b"])
        result = parse_data(obj)
        assert result == {"summary": "hello", "keywords": ["a", "b"]}

    def test_dict(self):
        result = parse_data({"key": "val"})
        assert result == {"key": "val"}

    def test_list(self):
        result = parse_data([{"row": 1}, {"row": 2}])
        assert result == [{"row": 1}, {"row": 2}]

    def test_str_returns_empty_dict(self):
        result = parse_data("some text")
        assert result == {}

    def test_none_returns_empty_dict(self):
        result = parse_data(None)
        assert result == {}

    def test_tuple_returns_empty_dict(self):
        result = parse_data((1, 2))
        assert result == {}

    def test_int_returns_empty_dict(self):
        result = parse_data(42)
        assert result == {}


# ---------------------------------------------------------------------------
# FilePipeline.run — content / data split
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_processor():
    p = MagicMock()
    p.run = AsyncMock(return_value="raw file content")
    p.is_valid = AsyncMock(return_value=True)
    return p


class TestFilePipelineRun:
    async def test_no_transforms_content_is_processor_output_data_empty(self, mock_processor):
        pipeline = FilePipeline(mock_processor)
        result = await pipeline.run("dummy")

        assert result is not None
        assert result.content == "raw file content"
        assert result.data == {}

    async def test_content_preserved_through_transforms(self, mock_processor):
        transform = MagicMock()
        transform.run = AsyncMock(
            return_value=_FakePydantic(summary="extracted", keywords=["test"])
        )

        pipeline = FilePipeline(mock_processor, transforms=[transform])
        result = await pipeline.run("dummy")

        assert result is not None
        assert result.content == "raw file content"
        assert result.data == {"summary": "extracted", "keywords": ["test"]}

    async def test_content_with_list_transform(self, mock_processor):
        transform = MagicMock()
        transform.run = AsyncMock(return_value=[{"row": 1}, {"row": 2}])

        pipeline = FilePipeline(mock_processor, transforms=[transform])
        result = await pipeline.run("dummy")

        assert result is not None
        assert result.content == "raw file content"
        assert result.data == [{"row": 1}, {"row": 2}]

    async def test_accepts_returns_none(self, mock_processor):
        mock_processor.is_valid = AsyncMock(return_value=False)
        pipeline = FilePipeline(mock_processor)
        result = await pipeline.run("dummy")
        assert result is None

    async def test_processor_returns_none(self, mock_processor):
        mock_processor.run = AsyncMock(return_value=None)
        pipeline = FilePipeline(mock_processor)
        result = await pipeline.run("dummy")
        assert result is None
