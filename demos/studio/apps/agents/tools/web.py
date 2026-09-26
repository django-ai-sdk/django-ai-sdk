from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, cast

import trafilatura
from django_ai_sdk.logger import get_logger
from haystack.tools import Tool
from haystack_integrations.components.websearch.ddgs import DDGSWebSearch
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

if TYPE_CHECKING:
    from haystack import Document

logger = get_logger(__name__)

web_search = DDGSWebSearch(top_k=5)


class WebSearchError(Exception):
    """Raised when web search fails after retries."""


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=1, exp_base=2, max=10),
    reraise=True,
)
def _search(query: str) -> list[Document]:
    """Run the search engine."""
    return cast("list[Document]", web_search.run(query=query)["documents"])


def search_web(query: Annotated[str, "The research query"]) -> list[dict[str, str]]:
    """Search the web for the given query and return structured results."""
    try:
        docs = _search(query)
    except Exception as exc:
        logger.warning(f"search_web failed for {query!r}: {exc}")
        raise WebSearchError("Web search is unavailable.") from exc

    return [
        {
            "title": doc.meta.get("title", ""),
            "content": (doc.content or "")[:500],
            "url": doc.meta.get("url", ""),
        }
        for doc in docs
    ]


def fetch_page(url: Annotated[str, "The full URL to fetch"]) -> str:
    """Fetch and extract the main text content of a webpage."""
    try:
        downloaded = trafilatura.fetch_url(url)
    except Exception:
        return "This page could not be fetched."

    if not downloaded:
        return "Could not download this URL."

    text = trafilatura.extract(downloaded) or downloaded
    return text[:6000]


def search_web_tool(**kwargs: object) -> Tool:
    """Web search tool factory."""
    return Tool(
        name="search_web",
        description="Search the web for information on a topic. Returns summarized search results.",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The research query to search for",
                },
            },
            "required": ["query"],
        },
        function=search_web,
    )


def fetch_page_tool(**kwargs: object) -> Tool:
    """Webpage fetch tool factory."""
    return Tool(
        name="fetch_page",
        description="Fetch and extract the main text content of a webpage URL.",
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The full URL to fetch"},
            },
            "required": ["url"],
        },
        function=fetch_page,
    )
