"""A turn's content: which text is kept, how it is masked, and how it is read back.

Anything that can quote the turn is stored with its content. That includes
judges' reasons, names a judge wrote itself for an `Entities` lens, per-item scores keyed
by source URL. Capture decides what is kept, and
forgetting the turn (`forget_turns`) or its expiry destroys it.
"""

from __future__ import annotations

import re
from functools import cache
from typing import TYPE_CHECKING, Any

from django.utils import timezone

from django_ai_sdk.oversight.patterns import mask_all, mask_json
from django_ai_sdk.oversight.types import (
    Answer,
    Capture,
    HistoryItem,
    Source,
    Subject,
    ToolCall,
    TurnContext,
)
from django_ai_sdk.utils import map_strings

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from django.db.models import QuerySet

    from django_ai_sdk.oversight.guard import TurnGuard
    from django_ai_sdk.oversight.guard.asking import Answered
    from django_ai_sdk.oversight.lenses import Lens
    from django_ai_sdk.oversight.models import Turn, TurnContent


# Content sections that are added after close.
SIDE_SECTIONS = frozenset({"texts"})


# Stands in for each name a judge wrote itself for an `Entities` lens. One per name keeps
# `.is_true` and counts right. The name itself is kept only in the content.
FOUND = "[found]"


# Found text shorter than this is not masked, since it would hit ordinary words.
MIN_LITERAL = 3


def as_source(item: Any) -> Source:
    if isinstance(item, Source):
        return item
    if isinstance(item, dict):
        return Source.model_validate({k: v for k, v in item.items() if k in Source.model_fields})
    # Any other object, such as a NumberedSource: read its text fields by name.
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
    """Build the turn's content to keep at `capture`. None for Capture.NONE.

    `keep` is `masker_for_capture(capture, ...)`. It runs over every kept string,
    including the history, sources and tool calls.
    """
    view = guard.context
    if capture is Capture.NONE:
        return None
    base: dict[str, Any] = {
        "agent": {"id": view.agent_id, "name": view.agent_name},
        # Lens spans point into the original text, which only FULL keeps.
        "capture": str(capture),
    }
    if capture is Capture.METADATA:
        # No text at all. Even a title or URL can reveal who the turn was about.
        return {
            **base,
            "input_chars": len(guard.original_input or view.input),
            "output_chars": len(guard.original_output or view.output),
            "sources": [{"kind": s.kind, "seeded": s.seeded} for s in view.sources],
            "tool_calls": [{"name": c.name, "error": c.error} for c in view.tool_calls],
        }
    assert keep is not None
    full = capture is Capture.FULL
    return {
        **base,
        "input": guard.original_input if full else keep.text(view.input),
        "output": (guard.original_output or view.output) if full else keep.text(view.output),
        # What was shown or stored, when it differs from the original.
        "shown_input": view.input if full and view.input != guard.original_input else None,
        "shown_output": view.output
        if full and guard.original_output and view.output != guard.original_output
        else None,
        "history": keep([h.model_dump() for h in view.history]),
        "sources": keep([s.model_dump() for s in view.sources]),
        "tool_calls": keep([c.model_dump(mode="json") for c in view.tool_calls]),
    }


class Masker:
    """Masks every string that MASKED capture keeps.

    First the built-in detectors run (all `PII` kinds and `Secrets`). Then every
    occurrence of text a lens found is replaced by its label, such as `[EMAIL]`,
    longest first. `active=False` leaves text as is, for FULL capture.
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
        return _mask_detected(text)

    def __call__(self, value: Any) -> Any:
        """Mask every string in a JSON value."""
        return mask_json(value, self.text) if self.active else value

    def keys(self, value: dict[str, float]) -> dict[str, float]:
        """Mask the keys of `value`, since a source's key can be its URL."""
        masked: dict[str, float] = {}
        for key, score in value.items():
            name = self.text(key)
            masked[name] = max(score, masked.get(name, score))
        return masked


def masker_for_capture(
    capture: Capture | str, found: Mapping[str, str] | None = None
) -> Masker | None:
    """Return the masker for text kept at `capture`. None when no text is kept.

    FULL keeps text as is. MASKED masks it, using `found` from the turn's lenses.
    NONE and METADATA keep no text.
    """
    if capture == Capture.FULL:
        return Masker(active=False)
    if capture == Capture.MASKED:
        return Masker(found)
    return None


def turn_found(guard: TurnGuard) -> dict[str, str]:
    """Map each text this turn's entity lenses found to its label (see `found_literals`)."""
    view = guard.context
    said_output = guard.original_output or view.output
    texts = {
        # Gate lenses read the question before the gate masked it.
        Subject.INPUT: guard.original_input or view.input,
        Subject.OUTPUT: said_output,
        Subject.EXCHANGE: f"{view.input}\n\n{said_output}",
        Subject.SOURCES: "\n\n".join(s.content for s in view.fresh_sources()),
    }
    return found_literals(guard.answered, texts)


def unmasked_literals(masked: str, shown: str) -> dict[str, str]:
    """Map each text that masking replaced in `shown` to store `masked` to its label.

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
    """Map each text an `Entities` lens found to its label.

    Only entities count. They find names and addresses, which the record must not
    keep. Topic and keyword lenses point at ordinary words that stay readable.
    `texts` holds the text each subject's lenses read. Spans outside it, such as
    a tool call's, are skipped.
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


def _mask_detected(text: str) -> str:
    """Mask personal data (every kind, all countries) and secrets in `text`."""
    if not text:
        return text
    spans = [span for lens in _detectors() for span in lens.find(text)]
    return mask_all(text, [(spans, "[{label}]")]) if spans else text


@cache
def _detectors() -> tuple[Any, ...]:
    from django_ai_sdk.oversight.patterns import PII_PATTERNS
    from django_ai_sdk.oversight.presets import PIILens, SecretsLens

    return PIILens(name="captured", kinds=list(PII_PATTERNS)), SecretsLens(name="captured")


def stored_value(lens: Lens, answer: Answer) -> Any:
    """Return the answer's value as stored in the record and shown to the client.

    For an `Entities` lens, declared labels and the kinds code found stay as they
    are. Any other name becomes FOUND, and the name itself is kept only in the content.
    """
    from django_ai_sdk.utils import serialize

    value = serialize(answer.value)
    if lens.answer.kind != "entities" or not value:
        return value
    items = value if isinstance(value, list) else [value]
    labels = set(lens.labels)
    if not labels and lens.reports_kinds(answer):
        return items
    return [item if item in labels else FOUND for item in items]


def without_nul(value: Any) -> Any:
    """Remove NUL characters from every string in a JSON value, keys included."""
    return map_strings(value, lambda text: text.replace("\x00", ""), keys=True)


def load_content(turn: Turn) -> dict[str, Any] | None:
    """Load the turn's content. None when it has none or it expired (prune deletes it later)."""
    row = turn.content
    return None if row is None or _expired(row) else row.stored


def _expired(row: TurnContent) -> bool:
    return row.expires_at is not None and row.expires_at <= timezone.now()


def load_contents(turns: list[Turn]) -> dict[Any, dict[str, Any] | None]:
    """Like `load_content` for many turns, keyed by turn id.

    Content past its expiry and not yet pruned loads as None, as `content_state` says.
    """
    now = timezone.now()
    return {
        t.id: None
        if t.content is None or (t.content.expires_at and t.content.expires_at <= now)
        else t.content.stored
        for t in turns
    }


def add_to_content(turn: Turn, *, texts: dict[str, Any] | None = None) -> bool:
    """Add answer texts to the turn's content.

    The row is locked so two writers at once both land.
    Returns False without writing when the content is gone, expired, or was
    forgotten since.
    """
    from django.db import transaction

    from django_ai_sdk.oversight.models import TurnContent

    if turn.content_id is None:
        return False
    with transaction.atomic():
        row = TurnContent.objects.select_for_update().filter(pk=turn.content_id).first()
        content = row.stored if row is not None and not _expired(row) else None
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
    """Forget the content of every turn in the thread (`forget_turns`)."""
    from django_ai_sdk.oversight.models import Turn

    forget_turns(Turn.objects.filter(thread_id=str(thread_id)))


def forget_turns(turns: QuerySet[Turn]) -> None:
    """Delete the turns' content and blank reviewers' notes on them. The turns stay.

    Run it in a transaction. The turns are locked first: a turn closing now either
    waits and then finds itself flagged `content_forgotten`, or has stored its
    content already and it is deleted here.
    """
    from django_ai_sdk.oversight.models import ReviewItem, Turn, TurnContent

    locked = list(
        turns.select_for_update().order_by("pk").values_list("pk", "content_id", "status")
    )
    ids = [pk for pk, _, _ in locked]
    # SET_NULL clears the turns' content link.
    TurnContent.objects.filter(pk__in=[c for _, c, _ in locked if c is not None]).delete()
    ReviewItem.objects.filter(turn_id__in=ids).exclude(note="").update(note="")
    open_ids = [pk for pk, _, status in locked if status == Turn.Status.OPEN]
    if open_ids:
        Turn.objects.filter(pk__in=open_ids).update(content_forgotten=True)


def context_from_turn(turn: Turn, content: dict[str, Any] | None) -> TurnContext:
    """Rebuild a TurnContext from the record, for reflect and review queues."""
    content = content or {}
    return TurnContext(
        turn_id=str(turn.id),
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


__all__ = [
    "FOUND",
    "SIDE_SECTIONS",
    "Masker",
    "add_to_content",
    "as_source",
    "capture_content",
    "context_from_turn",
    "forget_thread",
    "forget_turns",
    "found_literals",
    "load_content",
    "load_contents",
    "masker_for_capture",
    "stored_value",
    "turn_found",
    "without_nul",
]
