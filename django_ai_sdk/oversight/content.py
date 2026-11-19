"""A turn's content: which text is kept, how it is masked, and how it is read back.

Anything that can quote the turn (judges' reasons, judge-written entity names,
scores keyed by source URL) is stored with the content, so Capture governs it and
forgetting or expiry destroys it.
"""

from __future__ import annotations

import re
from functools import cache
from itertools import batched
from typing import TYPE_CHECKING, Any

from django.db import transaction
from django.utils import timezone

from django_ai_sdk.oversight.lenses import PIILens, SecretsLens
from django_ai_sdk.oversight.patterns import PII_PATTERNS, mask_all, mask_json
from django_ai_sdk.oversight.types import (
    Answer,
    Capture,
    HistoryItem,
    Source,
    Subject,
    ToolCall,
    TurnContext,
)
from django_ai_sdk.utils import map_strings, serialize

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from django.db.models import QuerySet

    from django_ai_sdk.oversight.guard import TurnGuard
    from django_ai_sdk.oversight.guard.judging import Answered
    from django_ai_sdk.oversight.lenses import Lens
    from django_ai_sdk.oversight.models import Turn, TurnContent


# Content sections that are added after close.
SIDE_SECTIONS = frozenset({"texts"})

# Stands in for each judge-written entity name, keeping `.is_true` and counts right.
FOUND = "[found]"

# Shorter found text is not masked, since it would hit ordinary words.
MIN_LITERAL = 3
FORGET_BATCH = 500


def as_source(item: Any) -> Source:
    if isinstance(item, Source):
        return item
    if isinstance(item, dict):
        return Source.model_validate({k: v for k, v in item.items() if k in Source.model_fields})
    return Source(
        **{
            name: str(getattr(item, name, "") or "")
            for name in Source.model_fields
            if name != "seeded"
        }
    )


def capture_content(
    guard: TurnGuard, capture: Capture, keep: Masker | None
) -> dict[str, Any] | None:
    """The turn's content to keep at `capture`; `keep` masks every kept string."""
    context = guard.context
    if capture is Capture.NONE:
        return None
    base: dict[str, Any] = {
        "agent": {"id": context.agent_id, "name": context.agent_name},
        # Lens spans point into the original text, which only FULL keeps.
        "capture": str(capture),
    }
    if capture is Capture.METADATA:
        # No text at all. Even a title or URL can reveal who the turn was about.
        return {
            **base,
            "input_chars": len(guard.original_input or context.input),
            "output_chars": len(guard.original_output or context.output),
            "sources": [{"kind": s.kind, "seeded": s.seeded} for s in context.sources],
            "tool_calls": [{"name": c.name, "error": c.error} for c in context.tool_calls],
        }
    assert keep is not None
    full = capture is Capture.FULL
    return {
        **base,
        "input": guard.original_input if full else keep.text(context.input),
        "output": (guard.original_output or context.output) if full else keep.text(context.output),
        "shown_input": context.input if full and context.input != guard.original_input else None,
        "shown_output": context.output
        if full and guard.original_output and context.output != guard.original_output
        else None,
        "history": keep([h.model_dump() for h in context.history]),
        "sources": keep([s.model_dump() for s in context.sources]),
        "tool_calls": keep([c.model_dump(mode="json") for c in context.tool_calls]),
    }


class Masker:
    """Masks every string MASKED capture keeps: lens-found text, then built-in detectors.

    `active=False` leaves text as is, for FULL capture.
    """

    def __init__(self, found: Mapping[str, str] | None = None, *, active: bool = True) -> None:
        self.active = active
        self.found = {
            literal: label
            for literal, label in (found or {}).items()
            if len(literal.strip()) >= MIN_LITERAL
        }
        ordered = sorted(self.found, key=len, reverse=True)
        self._pattern = re.compile("|".join(map(re.escape, ordered))) if ordered else None

    def text(self, text: str) -> str:
        if not self.active or not text:
            return text
        # Found literals first: a detector could otherwise cut one apart and leave the rest.
        if self._pattern is not None:
            text = self._pattern.sub(
                lambda m: f"[{(self.found[m.group()] or 'masked').upper()}]", text
            )
        spans = [span for lens in _detectors() for span in lens.finder.find(text, lens.name)]
        return mask_all(text, [(spans, "[{label}]")]) if spans else text

    def __call__(self, value: Any) -> Any:
        return mask_json(value, self.text) if self.active else value

    def keys(self, value: dict[str, float]) -> dict[str, float]:
        """A source's key can be its URL."""
        masked: dict[str, float] = {}
        for key, score in value.items():
            name = self.text(key)
            masked[name] = max(score, masked.get(name, score))
        return masked


def capture_masker(capture: Capture | str, found: Mapping[str, str] | None = None) -> Masker | None:
    """None when `capture` keeps no text."""
    if capture == Capture.FULL:
        return Masker(active=False)
    if capture == Capture.MASKED:
        return Masker(found)
    return None


def turn_literals(guard: TurnGuard) -> dict[str, str]:
    context = guard.context
    said_output = guard.original_output or context.output
    texts = {
        # Gate lenses read the question before the gate masked it.
        Subject.QUESTION: guard.original_input or context.input,
        Subject.ANSWER: said_output,
        Subject.EXCHANGE: f"{context.input}\n\n{said_output}",
        Subject.SOURCES: context.text(Subject.SOURCES),
    }
    return found_literals(guard.answered, texts)


def unmasked_literals(masked: str, shown: str) -> dict[str, str]:
    """Map each text that masking replaced in `shown` to its label.

    shortcut: knows only the "[LABEL]" masks the SDK writes; a Mask with a custom
    template is not recovered. Store the found texts per turn if that is needed.
    """
    parts = _MASK.split(masked)
    if len(parts) == 1:
        return {}
    pattern = "".join(re.escape(p) if i % 2 == 0 else "(.+?)" for i, p in enumerate(parts))
    match = re.fullmatch(pattern, shown, re.DOTALL)
    if match is None:
        return {}
    labels = parts[1::2]
    return {
        text: label.lower()
        for text, label in zip(match.groups(), labels, strict=True)
        if text != f"[{label}]"
    }


_MASK = re.compile(r"\[([A-Z][A-Z0-9_]*)\]")


def found_literals(answered: Iterable[Answered], texts: Mapping[Subject, str]) -> dict[str, str]:
    """Map each text an entities lens found to its label.

    Only entities count: they find names and addresses, which the record must not
    keep; other lenses point at ordinary words that stay readable.
    """
    found: dict[str, str] = {}
    for item in answered:
        if item.lens.answer.kind != "entities":
            continue
        text = texts.get(item.lens.subject, "")
        for span in item.answer.spans:
            if 0 <= span.start < span.end <= len(text):
                found.setdefault(text[span.start : span.end], span.label or item.lens.name)
    return found


@cache
def _detectors() -> tuple[Any, ...]:
    # No `not_after`: a number that only its label kept unmasked is not kept on record.
    numbers = PIILens(name="captured", kinds=list(PII_PATTERNS), not_after={})
    return numbers, SecretsLens(name="captured")


def stored_value(lens: Lens, answer: Answer) -> Any:
    """The value as stored and shown: judge-written entity names become FOUND."""
    value = serialize(answer.value)
    if lens.answer.kind != "entities" or not value:
        return value
    items = value if isinstance(value, list) else [value]
    labels = set(lens.labels)
    if not labels and lens.reports_kinds(answer):
        return items
    return [item if item in labels else FOUND for item in items]


def without_nul(value: Any) -> Any:
    return map_strings(value, lambda text: text.replace("\x00", ""), keys=True)


def load_content(turn: Turn) -> dict[str, Any] | None:
    row = turn.content
    return None if row is None or expired(row) else row.stored


def expired(row: TurnContent) -> bool:
    return row.expires_at is not None and row.expires_at <= timezone.now()


def add_to_content(turn: Turn, *, texts: dict[str, Any] | None = None) -> bool:
    """Add answer texts, locking the row so concurrent writers both land.

    False without writing when the content is gone, expired, or forgotten.
    """
    from django_ai_sdk.oversight.models import TurnContent

    if turn.content_id is None:
        return False
    with transaction.atomic():
        row = TurnContent.objects.select_for_update().filter(pk=turn.content_id).first()
        content = row.stored if row is not None and not expired(row) else None
        if row is None or content is None:
            return False
        content = {
            **content,
            "texts": {**(content.get("texts") or {}), **(texts or {})},
        }
        content = {k: v for k, v in content.items() if v != {} or k not in SIDE_SECTIONS}
        TurnContent.objects.filter(pk=row.pk).update(stored=content)
    return True


def forget_thread(thread_id: Any) -> None:
    from django_ai_sdk.oversight.models import Turn

    forget_turns(Turn.objects.filter(thread_id=str(thread_id)))


def forget_turns(turns: QuerySet[Turn]) -> None:
    """Delete the turns' content and reviewers' notes; the turns stay. Run in a transaction.

    Locking first means a turn closing now, or later after it was swept as interrupted,
    either finds itself `content_forgotten` or has stored its content already, which is
    deleted here.
    """
    from django_ai_sdk.oversight.models import ReviewItem, Turn, TurnContent

    locked = list(
        turns.select_for_update().order_by("pk").values_list("pk", "content_id", "status")
    )
    ids = [pk for pk, _, _ in locked]
    # SET_NULL clears the turns' content link.
    TurnContent.objects.filter(pk__in=[c for _, c, _ in locked if c is not None]).delete()
    ReviewItem.objects.filter(turn_id__in=ids).exclude(note="").update(note="")
    # An interrupted turn may still close: it stores nothing either.
    unclosed = [pk for pk, _, status in locked if status != Turn.Status.CLOSED]
    if unclosed:
        Turn.objects.filter(pk__in=unclosed).update(content_forgotten=True)


def forget_actor(actor_id: str) -> int:
    """Clear `actor_id` from its turns, forgetting the content of threadless ones.

    Threaded turns keep content until their thread is deleted. Run in a transaction;
    safe to rerun (`oversight_forget_actor`). Returns how many turns named the actor.
    """
    from django_ai_sdk.oversight.models import Turn

    if not actor_id:
        return 0
    turns = Turn.objects.filter(actor_id=actor_id)
    threadless = turns.filter(thread_id="").values_list("pk", flat=True).iterator(FORGET_BATCH)
    for batch in batched(threadless, FORGET_BATCH):
        forget_turns(Turn.objects.filter(pk__in=batch))
    return turns.update(actor_id="")


def context_from_turn(turn: Turn, content: dict[str, Any] | None) -> TurnContext:
    content = content or {}
    return TurnContext(
        turn_id=str(turn.id),
        thread_id=turn.thread_id,
        channel=turn.channel,
        agent_id=turn.agent_id,
        agent_name=turn.agent_name,
        input=content.get("shown_input") or content.get("input") or "",
        output=content.get("shown_output") or content.get("output") or "",
        history=[HistoryItem.model_validate(h) for h in content.get("history") or []],
        sources=[
            Source.model_validate(s) for s in content.get("sources") or [] if isinstance(s, dict)
        ],
        tool_calls=[
            ToolCall.model_validate(c) for c in content.get("tool_calls") or [] if c.get("name")
        ],
    )
