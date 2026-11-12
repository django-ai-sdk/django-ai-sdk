"""Rules act on what lenses answer.

    Rule(
        "block_injection",
        when=L("prompt_injection").is_true & L("prompt_injection").confident(0.9),
        then=Block("I can't help with that request."),
        mode=Mode.ENFORCE,
    )

A rule starts in report mode: it reports what it would have done, on every turn,
so its false-positive rate is known before it can block anyone.
"""

from __future__ import annotations

from typing import Any

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
    ) -> None:
        self.name = name
        self.when = when if isinstance(when, Condition) else condition_from_dict(when)
        self.then = [then] if isinstance(then, Action) else list(then)
        self.mode = Mode(mode)
        # Shown to the user when the rule refuses (a block, or a tool it took away).
        # Overrides Block's message.
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
        }

    @classmethod
    def from_dict(cls, node: dict[str, Any]) -> Rule:
        return cls(
            node["name"],
            when=node["when"],
            then=[action_from_dict(a) for a in node.get("then") or []],
            mode=node.get("mode", Mode.REPORT),
            message=node.get("message", ""),
            notice=node.get("notice", ""),
            description=node.get("description", ""),
        )

    def ordered_actions(self) -> list[Action]:
        return sorted(self.then, key=lambda action: rank(ACTION_ORDER, action.kind))


__all__ = ["Rule"]
