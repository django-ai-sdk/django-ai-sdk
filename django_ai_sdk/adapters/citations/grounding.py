from __future__ import annotations

import asyncio
import difflib
import json
import re
from collections import Counter
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

from django.core.exceptions import ImproperlyConfigured
from django.utils.module_loading import import_string

from django_ai_sdk.common import prompt
from django_ai_sdk.logger import get_logger
from django_ai_sdk.utils import resolve_setting

if TYPE_CHECKING:
    from django_ai_sdk.adapters.base import Run
    from django_ai_sdk.agents.base import Agent
    from django_ai_sdk.artifacts.schemas import Citation, CitationsData

    from .registry import SourceRegistry

logger = get_logger(__name__)

# how a passage was found in its source
Match = Literal["exact", "normalized", "fuzzy", "partial"]
# pieces of a passage too short to locate on their own
MIN_PIECE = 20
# characters of each source the citation agent reads
SOURCE_CHARS = 6000
# below this similarity a quote counts as not found in the answer.
FUZZY_RATIO = 0.85
# least characters of answer per citation call
BLOCK_CHARS = 300

CITATION_PROMPT = prompt("""\
    You get numbered sources and an answer, or a part of one, written from them.
    List the parts of the answer that a source supports. For each part, copy it from the
    answer exactly (a clause or a sentence, as written). For each source that supports it,
    give the source's number and copy the passage of that source that supports it, exactly
    as it is written in the source (one or two sentences, or the table row or field).
    Skip parts that no source supports. Do not change the answer or the sources.
    Reply with only this JSON, nothing else:
    {"citations": [{"quote": "<exact text from the answer>",
                    "support": [{"source": 1, "text": "<exact text from source 1>"}]}]}
""")


def get_citation_agent(agent: Agent) -> Agent | Run:
    """Get citation agent from settings, or the agent itself if none"""
    from django_ai_sdk.agents.base import Agent  # noqa: PLC0415
    from django_ai_sdk.agents.utils import task_runner  # noqa: PLC0415

    path = resolve_setting("AI_SDK_CITATION_AGENT", None)
    if not path:
        return task_runner() or agent
    agent_class = import_string(path)
    if not (isinstance(agent_class, type) and issubclass(agent_class, Agent)):
        raise ImproperlyConfigured(f"AI_SDK_CITATION_AGENT must name an Agent subclass: {path!r}")
    return agent_class()


def render(answer: str, registry: SourceRegistry) -> str:
    """The citation agent's input: the numbered sources, then the answer.

    Sources first: a turn's calls then share a prefix the provider can cache.
    """
    sources = "\n\n".join(
        f"[{s.index}] {s.title}\n{s.content[:SOURCE_CHARS]}" for s in registry.all
    )
    return f"Sources:\n{sources}\n\nAnswer:\n{answer}"


def parse_pairs(reply: str) -> list[tuple[str, list[tuple[int, str]]]]:
    """parse JSON from the citation agent's reply"""

    # FIX: this was probably wrapped due to vendor structured output formatting issues.
    text = reply.strip()
    if text.startswith("```"):  # a ```json wrapper around the object
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    start, end = text.find("{"), text.rfind("}")
    try:
        data = json.loads(text[start : end + 1]) if start != -1 else {}
    except json.JSONDecodeError:
        return []
    pairs = []
    for item in data.get("citations") or [] if isinstance(data, dict) else []:
        if not isinstance(item, dict) or not isinstance(item.get("quote"), str):
            continue
        support = [
            (s["source"], s.get("text") if isinstance(s.get("text"), str) else "")
            for s in item.get("support") or []
            if isinstance(s, dict) and isinstance(s.get("source"), int)
        ]
        if item["quote"].strip() and support:
            pairs.append((item["quote"].strip(), support))
    return pairs


_IGNORED = re.compile(r"[\s*_`#>|]+")  # whitespace
_PLAIN = str.maketrans(
    {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u00a0": " ",
        "\u202f": " ",
    }
)


def _normalized(text: str) -> tuple[str, list[int]]:
    """`text` lowercased without markdown marks"""
    text = text.translate(_PLAIN)  # offset plain markers
    out: list[str] = []
    offsets: list[int] = []
    for match in re.finditer(r"[\s*_`#>|]+|[^\s*_`#>|]+", text):
        if _IGNORED.fullmatch(match.group()):
            if out and out[-1] != " ":
                out.append(" ")
                offsets.append(match.start())
            continue
        for i, char in enumerate(match.group().lower()):
            out.append(char)
            offsets.append(match.start() + i)
    return "".join(out), offsets


def _span(offsets: list[int], start: int, end: int, text: str) -> tuple[int, int]:
    """Raw offsets for the normalized range, trimmed of whitespace."""
    raw_start, raw_end = offsets[start], offsets[end - 1] + 1
    while raw_start < raw_end and text[raw_start].isspace():
        raw_start += 1
    while raw_end > raw_start and text[raw_end - 1].isspace():
        raw_end -= 1
    return raw_start, raw_end


def locate(text: str, quote: str, start: int = 0) -> tuple[int, int] | None:
    """Find `quote` is in `text`, as raw `(start, end)` offsets, or None."""
    found = match(text, quote, start)
    return found and found[:2]


def match(text: str, quote: str, start: int = 0) -> tuple[int, int, Match] | None:
    """Where `quote` is in `text`, as raw `(start, end)` offsets, or None."""
    for begin in (start, 0):
        if (i := text.find(quote, begin)) != -1:
            return i, i + len(quote), "exact"

    norm_text, offsets = _normalized(text)
    norm_quote = _normalized(quote)[0].strip()
    if not norm_quote or not norm_text:
        return None
    norm_start = next((i for i, o in enumerate(offsets) if o >= start), 0)
    for begin in (norm_start, 0):
        if (i := norm_text.find(norm_quote, begin)) != -1:
            return *_span(offsets, i, i + len(norm_quote), text), "normalized"

    best: tuple[float, int] = (0.0, -1)
    size = len(norm_quote)
    for i in range(0, max(1, len(norm_text) - size + 1)):
        if i and norm_text[i - 1] != " ":
            continue  # only windows starting at a word
        ratio = difflib.SequenceMatcher(None, norm_quote, norm_text[i : i + size]).ratio()
        if ratio > best[0]:
            best = (ratio, i)
    if best[0] < FUZZY_RATIO:
        return None
    end = min(best[1] + size, len(norm_text))
    while end < len(norm_text) and norm_text[end] != " ":  # finish the last word
        end += 1
    return *_span(offsets, best[1], end, text), "fuzzy"


async def ground(answer: str, registry: SourceRegistry, agent: Agent | Run) -> CitationsData | None:
    """The answer's citations and the sources to store"""
    if not registry.all or not answer.strip():
        return None
    return await Grounder(registry, agent).finish(answer)


class Grounder:
    """Grounds an answer block by block while it streams"""

    def __init__(
        self,
        registry: SourceRegistry,
        agent: Agent | Run,
        on_block: Callable[[], None] | None = None,
    ) -> None:
        self.registry, self.agent, self.on_block = registry, agent, on_block
        self.text = ""  # the answer so far
        self.cursor = 0  # where the next block starts
        self.tasks: list[asyncio.Task[list[Citation]]] = []

    def feed(self, delta: str) -> None:
        """Add streamed answer text; ground each block once it is complete."""
        self.text += delta
        if "\n" in delta and self.registry.all:  # a block ends at a blank line
            self._start(self._block_end())

    async def finish(self, answer: str, timeout: float | None = None) -> CitationsData:
        """The whole answer's citations"""

        async for _ in self.wait(answer, timeout):
            pass
        return self.citations()

    async def wait(self, answer: str, timeout: float | None = None) -> AsyncIterator[None]:
        """Ground the rest of `answer`"""
        self.text = answer
        self._start(len(answer))
        loop = asyncio.get_running_loop()
        deadline = None if timeout is None else loop.time() + timeout
        pending = {t for t in self.tasks if not t.done()}
        while pending:
            left = None if deadline is None else max(0.0, deadline - loop.time())
            done, pending = await asyncio.wait(
                pending, timeout=left, return_when=asyncio.FIRST_COMPLETED
            )
            if not done:
                logger.warning(
                    "Citations took longer than {}s; {} blocks have none",
                    timeout,
                    len(pending),
                )
                break
            if pending:
                yield
        self.cancel()

    def citations(self) -> CitationsData:
        """The citations of the blocks grounded so far."""
        from django_ai_sdk.artifacts.schemas import CitationsData  # noqa: PLC0415

        citations = sorted(
            (c for t in self.tasks if t.done() and not t.cancelled() for c in t.result()),
            key=lambda c: c.start,
        )
        return CitationsData(citations=citations, sources=_stored_sources(citations, self.registry))

    def cancel(self) -> None:
        for task in self.tasks:
            task.cancel()

    def _block_end(self) -> int:
        """End of the complete blocks since the cursor"""
        end = self.text.rfind("\n\n", self.cursor + BLOCK_CHARS)
        if end == -1 or self.text.count("```", 0, end) % 2:
            return self.cursor
        return end + 2

    def _start(self, end: int) -> None:
        if not self.text[self.cursor : end].strip():
            return
        start, self.cursor = self.cursor, end
        task = asyncio.create_task(self._cite(self.text, start, end))
        if self.on_block:
            task.add_done_callback(lambda _: self.on_block and self.on_block())
        self.tasks.append(task)

    async def _cite(self, answer: str, start: int, end: int) -> list[Citation]:
        try:
            cited = await cite(answer[start:end], self.registry, self.agent)
        except Exception:
            logger.exception("Placing citations failed for a block")
            return []
        return [
            c.model_copy(update={"start": c.start + start, "end": c.end + start}) for c in cited
        ]


async def cite(answer: str, registry: SourceRegistry, agent: Agent | Run) -> list[Citation]:
    """Citations of `answer`, offsets into it, from one citation agent call."""
    from django_ai_sdk.artifacts.schemas import Citation, Evidence  # noqa: PLC0415
    from django_ai_sdk.common import ChatMessage  # noqa: PLC0415

    reply = await agent.run(
        [ChatMessage(role="user", content=render(answer, registry))],
        system_prompt=CITATION_PROMPT,
        response_format=None,
    )
    citations: list[Citation] = []
    outcomes: Counter[str] = Counter()  # per supporting passage
    cursor = 0
    for quote, support in parse_pairs(reply if isinstance(reply, str) else ""):
        evidence = get_evidence(support, registry, Evidence, outcomes)
        span = locate(answer, quote, cursor)
        if not evidence or span is None:
            outcomes["rejected"] += 1
            logger.debug("Citation rejected: {!r} → {}", quote[:80], [n for n, _ in support])
            continue
        citations.append(
            Citation(
                start=span[0],
                end=span[1],
                quote=answer[span[0] : span[1]],
                evidence=evidence,
            )
        )
        cursor = span[1]
    logger.info("Citations placed: {}, passages {}", len(citations), dict(outcomes))
    return citations


def uncited(registry: SourceRegistry) -> CitationsData:
    """The turn's own sources without citations"""
    from django_ai_sdk.artifacts.schemas import CitationsData  # noqa: PLC0415

    return CitationsData(citations=[], sources=_stored_sources([], registry))


def get_evidence(
    support: list[tuple[int, str]],
    registry: SourceRegistry,
    evidence_class: Any,
    outcomes: Counter[str],
) -> list[Any]:
    """Single Evidence per source whose passage is found in that source."""
    found: dict[str, Any] = {}
    for number, passage in support:
        source = registry.get(number)
        content = source.content[:SOURCE_CHARS] if source else ""
        located = get_locate_passage(content, passage) if source and passage else None
        outcomes[located[2] if located else "missing"] += 1
        if source and located and source.key not in found:
            start, end, how = located
            found[source.key] = evidence_class(
                key=source.key,
                start=start,
                end=end,
                text=content[start:end],
                matcher=how,
            )
    return list(found.values())


def get_locate_passage(content: str, passage: str) -> tuple[int, int, Match] | None:
    """Where `passage` is in `content`."""

    if found := match(content, passage):
        return found
    pieces = re.split(r"(?<=[.!?;:])\s+|\n+|\.{3}|…", passage)
    spans: list[tuple[int, int]] = []
    for piece in (p.strip() for p in pieces):
        cursor = spans[-1][1] if spans else 0
        if len(piece) >= MIN_PIECE and (span := locate(content, piece, cursor)):
            if span[0] >= cursor:  # in order
                spans.append(span)
    if not spans:
        return None
    start, end = spans[0][0], spans[-1][1]
    if end - start > 2 * len(passage):  # stitched from far apart: keep the longest piece
        start, end = max(spans, key=lambda s: s[1] - s[0])
    return start, end, "partial"


def _stored_sources(citations: list[Any], registry: SourceRegistry) -> list[Any]:
    """The turn's own sources and every cited one"""
    from django_ai_sdk.artifacts.schemas import CitedSource  # noqa: PLC0415

    numbers: dict[str, int] = {}
    for citation in citations:
        for evidence in citation.evidence:
            numbers.setdefault(evidence.key, len(numbers) + 1)
    return [
        CitedSource(
            key=s.key,
            number=numbers.get(s.key, 0),
            title=s.title,
            kind=s.kind,
            content=s.content[:SOURCE_CHARS],
            url=s.url,
            document_id=s.doc_id,
            chunk_id=s.chunk_id,
            cited=s.key in numbers,
        )
        for s in registry.all
        if s.key in numbers or s.key not in registry.seeded  # earlier turns' only when cited
    ]
