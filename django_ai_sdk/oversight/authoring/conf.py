"""The AI_SDK_OVERSIGHT_AUTHORING setting. docs/content/oversight-authoring.md lists the keys."""

from __future__ import annotations

from django_ai_sdk.checks import AppSettings, names

# What a lens may configure inline, and the arguments each takes on top of a lens's
# own. None of them take code or a regex, because authored config runs on every turn
# and a bad regex could backtrack for seconds. A regex or a model needs an alias.
INLINE_ARGS: dict[str, frozenset[str]] = {
    "keywords": frozenset({"keywords"}),
    "pii": frozenset({"kinds"}),
    "secrets": frozenset(),
    "refusal": frozenset(),
    "length": frozenset({"max_chars"}),
    "language": frozenset(),
}


setting = AppSettings(
    "AI_SDK_OVERSIGHT_AUTHORING",
    {
        # Policies every authored policy must extend, such as ["security"].
        "MUST_EXTEND": ([], names, 'Policy names declared in code, such as ["security"].'),
    },
)


__all__ = ["INLINE_ARGS", "setting"]
