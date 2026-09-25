from __future__ import annotations

from typing import TYPE_CHECKING

from .parser import normalize_citations

if TYPE_CHECKING:
    from .registry import CitationRegistry


class StreamingCitationBuffer:
    """Normalizes citations in streamed text, holding back a possibly split tag.

    An unclosed ``<``, ``[`` or ``【`` near the end of the text is held until the
    next chunk completes it, or until it grows past HOLD chars.
    """

    HOLD = 64  # Fits a bundled tag like <source id="10, 11, 12, 13" />.
    OPENERS = "<[【"  # Start of a citation form that may be split.
    CLOSERS = ">]】"  # Any of these after the opener means it completed.

    def __init__(self, registry: CitationRegistry) -> None:
        self.registry = registry
        self.cited: set[int] = set()
        self._pending = ""
        self._code = ""  # Code delimiter open in already-emitted text.

    def feed(self, text: str) -> str:
        self._pending += text
        start = max(self._pending.rfind(opener) for opener in self.OPENERS)
        tail = self._pending[start:] if start != -1 else ""
        held = tail and len(tail) <= self.HOLD and not any(c in tail for c in self.CLOSERS)
        cut = start if held else len(self._pending)
        # A fence can be split across chunks ("``" + "`"), so hold trailing backticks too.
        cut = len(self._pending[:cut].rstrip("`"))
        out, self._pending = self._pending[:cut], self._pending[cut:]
        return self._normalize(out)

    def flush(self) -> str:
        out, self._pending = self._pending, ""
        return self._normalize(out)

    def _normalize(self, text: str) -> str:
        if not text:
            return ""
        # Sources register before the model cites them
        text, cited, self._code = normalize_citations(text, self.registry.valid_ids, self._code)
        self.cited.update(cited)
        return text
