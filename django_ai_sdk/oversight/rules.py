"""Rules act on what lenses answer.

    Rule(
        "block_injection",
        when=L("prompt_injection").is_true & L("prompt_injection").confident(0.9),
        then=Block("I can't help with that request."),
        mode=Mode.ENFORCE,
    )

A rule starts in report mode. It reports what it would have done on every turn,
so you learn its false-positive rate before it can block anyone.

`min_mode` is a floor set in code. A runtime overlay cannot set the rule below it,
and cannot change the condition or actions of a rule that has a floor.
"""

from __future__ import annotations

from typing import Any

from django.core.exceptions import ImproperlyConfigured

from django_ai_sdk.oversight.actions import ACTION_ORDER, Action, action_from_dict
from django_ai_sdk.oversight.conditions import Condition, condition_from_dict
from django_ai_sdk.oversight.types import Mode, rank


class Rule:
    """When `when` matches, do `then`. Reports only until its mode is raised."""

    def __init__(
        self,
        name: str,
        *,
        when: Condition | dict[str, Any],
        then: Action | list[Action],
        mode: Mode | str = Mode.REPORT,
        message: str = "",
        notice: str = "",
        description: str = "",
        min_mode: Mode | str = Mode.OFF,
    ) -> None:
        self.name = name
        self.when = when if isinstance(when, Condition) else condition_from_dict(when)
        self.then = [then] if isinstance(then, Action) else list(then)
        self.mode = Mode(mode)
        # An overlay may not set the mode below this.
        self.min_mode = Mode(min_mode)
        if below(self.mode, self.min_mode):
            raise ImproperlyConfigured(
                f"Rule {name!r}: mode {self.mode} is below its min_mode {self.min_mode}"
            )
        # Shown to the user when the rule refuses (a block, or a tool it took away).
        # Overrides Block's message; the overlay may tune it.
        self.message = message
        # Shown to the user whenever the rule acts, e.g. "The fraud team is alerted."
        # Leave empty to queue or alert without telling the user.
        self.notice = notice
        self.description = description

    def __repr__(self) -> str:
        return f"<Rule {self.name!r} {self.mode}>"

    def lenses(self) -> set[str]:
        return self.when.lenses()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "when": self.when.to_dict(),
            "then": [action.to_dict() for action in self.then],
            "mode": str(self.mode),
            "message": self.message,
            "notice": self.notice,
            "description": self.description,
            "min_mode": str(self.min_mode),
        }

    @classmethod
    def from_dict(cls, node: dict[str, Any]) -> Rule:
        return cls(
            node["name"],
            when=condition_from_dict(node["when"]),
            then=[action_from_dict(a) for a in node.get("then") or []],
            mode=node.get("mode", Mode.REPORT),
            message=node.get("message", ""),
            notice=node.get("notice", ""),
            description=node.get("description", ""),
            min_mode=node.get("min_mode", Mode.OFF),
        )

    def ordered_actions(self) -> list[Action]:
        return sorted(self.then, key=lambda action: rank(ACTION_ORDER, action.kind))


# Modes from least to most protective.
MODE_ORDER = (Mode.OFF, Mode.REPORT, Mode.ENFORCE)


def below(mode: Mode | str, floor: Mode | str) -> bool:
    """Whether `mode` protects less than `floor` (off < report < enforce)."""
    return rank(MODE_ORDER, Mode(mode)) < rank(MODE_ORDER, Mode(floor))


__all__ = ["MODE_ORDER", "Rule", "below"]
