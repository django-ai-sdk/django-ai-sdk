"""Lens health for the system checks: which judges a lens reaches, and which never run."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from django_ai_sdk.oversight.judges.base import (
    Judge,
    alias_defined,
    alias_of,
    is_optional,
    resolve,
    spec_label,
)
from django_ai_sdk.oversight.judges.patterns import Patterns
from django_ai_sdk.oversight.lenses import JudgeLens

if TYPE_CHECKING:
    from django_ai_sdk.oversight.lenses import Lens


def reachable_judges(lens: JudgeLens) -> list[tuple[Any, Judge | None]]:
    """The judges a lens may ask, each built (None when it does not build).

    They stop at a `Patterns` judge that is sure even when it finds nothing: the
    chain always ends there, so no judge after it is ever asked.
    """
    reached: list[tuple[Any, Judge | None]] = []
    for spec in lens.usable_judges():
        try:
            judge = resolve(spec)
        except Exception:  # noqa: BLE001 - E004 and E005 report it
            judge = None
        reached.append((spec, judge))
        if isinstance(judge, Patterns) and judge.miss_confidence >= lens.min_confidence:
            break
    return reached


def lens_health(lens: Lens) -> dict[str, Any]:
    """Report whether `lens` can run on this deployment, and which parts never will.

    - runnable: at least one of its steps builds and has nothing `missing()`.
    - skipped: optional judge aliases that are not defined.
    - unjudged: optional aliases were skipped, and no judge but patterns is left.
    - unreachable: judges after a `Patterns` judge that is sure when it finds nothing.
    - undefined: required judge aliases that are not defined.
    """
    health: dict[str, Any] = {
        "skipped": [],
        "undefined": [],
        "unjudged": False,
        "unreachable": [],
    }
    if not isinstance(lens, JudgeLens):
        return {**health, "runnable": True}
    reached = reachable_judges(lens)
    built = [judge for _, judge in reached if judge is not None]
    skipped = [
        spec for spec in lens.judges if is_optional(spec) and not alias_defined(alias_of(spec))
    ]
    return {
        **health,
        "runnable": any(not judge.missing() for judge in built),
        "skipped": skipped,
        "undefined": [
            spec
            for spec in lens.judges
            if isinstance(spec, str) and not is_optional(spec) and not alias_defined(spec)
        ],
        "unjudged": bool(skipped) and all(isinstance(judge, Patterns) for judge in built),
        "unreachable": [spec_label(spec) for spec in lens.usable_judges()[len(reached) :]],
    }
