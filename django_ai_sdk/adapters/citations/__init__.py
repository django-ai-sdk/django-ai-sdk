from __future__ import annotations

from .formatter import (
    NumberedSource,
    SourceFormatter,
    SourcesFormatter,
    from_document,
    from_web_result,
)
from .registry import SourceRegistry, current_registry, source_key
from .utils import collect_sources

__all__ = [
    "NumberedSource",
    "SourceFormatter",
    "SourceRegistry",
    "SourcesFormatter",
    "collect_sources",
    "current_registry",
    "from_document",
    "from_web_result",
    "source_key",
]
