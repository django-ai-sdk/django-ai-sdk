from abc import ABC, abstractmethod

from haystack import Pipeline
from haystack.tools import ComponentTool


class HaystackRAGBase(ABC):
    """Abstract base class for Haystack RAG implementations."""

    _is_warmed_up: bool = False

    @abstractmethod
    def warmup(self) -> None:
        """
        Warm up the RAG by building the indexed document store (expensive).

        After warmup, subsequent build_pipeline() calls will use the cached store.
        """
        pass

    @property
    def needs_warmup(self) -> bool:
        """Check if warmup is needed."""
        return not self._is_warmed_up

    @abstractmethod
    def build_pipeline(self) -> Pipeline:
        """
        Build and return the RAG pipeline (query side, cheap).

        Returns:
            A Haystack Pipeline configured for RAG.
        """
        pass

    @abstractmethod
    def as_tool(self) -> ComponentTool:
        """
        Return the RAG pipeline wrapped as a ComponentTool.

        Returns:
            A ComponentTool wrapping the RAG pipeline.
        """
        pass
