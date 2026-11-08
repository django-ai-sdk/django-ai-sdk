"""What a rule does when it matches. Lenses only answer questions. Rules act.

`outcome` is what the turn becomes when the action is enforced. `effective` lists
the stages where the action can still change the turn. At other stages it is only
recorded.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from django.core.exceptions import ImproperlyConfigured
from django.utils.translation import gettext_lazy as _

from django_ai_sdk.oversight.types import Outcome, Severity, Stage

if TYPE_CHECKING:
    from collections.abc import Mapping

    from django.utils.functional import Promise


class Action:
    """Base class for what a matched rule does."""

    kind: ClassVar[str] = ""
    outcome: ClassVar[Outcome] = Outcome.FLAGGED
    # Stages where the action can still change the turn. Elsewhere it is recorded.
    effective: ClassVar[tuple[Stage, ...]] = tuple(Stage)

    def params(self) -> dict[str, Any]:
        return {}

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, **self.params()}

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.params()}>"


def _severity(value: str) -> str:
    try:
        return str(Severity(value))
    except ValueError:
        raise ValueError(
            f"Unknown severity {value!r}; one of {[str(s) for s in Severity]}"
        ) from None


class Alert(Action):
    """Mark the turn for attention at `severity`."""

    kind = "alert"

    def __init__(self, severity: str = "warning") -> None:
        self.severity = _severity(severity)

    def params(self) -> dict[str, Any]:
        return {"severity": self.severity}


class Mask(Action):
    """Mask the spans that `lenses` found. By default, every lens the rule reads.

    At the gate, the question is masked before the model or the thread sees it.
    At finish, the stored answer is masked and the client replaces its copy when
    the stream ends. In reflect, the stored message is masked afterwards.
    """

    kind = "mask"
    outcome = Outcome.MASKED
    effective = (Stage.GATE, Stage.FINISH, Stage.REFLECT)

    def __init__(self, lenses: list[str] | None = None, template: str = "[{label}]") -> None:
        try:
            template.format(label="X")
        except (KeyError, IndexError, ValueError) as exc:
            raise ValueError(f"Mask template {template!r} must only use {{label}}: {exc}") from exc
        self.lenses = list(lenses or [])
        self.template = template

    def params(self) -> dict[str, Any]:
        return {"lenses": self.lenses, "template": self.template}


class RestrictTools(Action):
    """Remove the tools in `names` for this turn. An empty list removes all tools."""

    kind = "restrict_tools"
    outcome = Outcome.RESTRICTED
    effective = (Stage.GATE, Stage.TOOL)

    def __init__(self, names: list[str] | None = None) -> None:
        self.names = list(names or [])

    def params(self) -> dict[str, Any]:
        return {"names": self.names}


class Block(Action):
    """Refuse the turn. The question never reaches the model, or the answer never reaches the user."""

    kind = "block"
    outcome = Outcome.BLOCKED
    effective = (Stage.GATE, Stage.TOOL, Stage.FINISH)

    # A lazy translation. Call str() on it where it is stored or sent.
    DEFAULT_MESSAGE = _("This request can't be answered here.")

    def __init__(self, message: str | Promise = "") -> None:
        self.message = message or self.DEFAULT_MESSAGE

    def params(self) -> dict[str, Any]:
        return {"message": str(self.message)}


class Queue(Action):
    """Add the turn to review queue `queue` at `priority`."""

    kind = "queue"

    def __init__(self, queue: str = "default", priority: int = 0) -> None:
        if isinstance(priority, bool) or not isinstance(priority, int):
            raise TypeError(f"Queue priority is a whole number, not {priority!r}")
        self.queue = queue
        self.priority = priority

    def params(self) -> dict[str, Any]:
        return {"queue": self.queue, "priority": self.priority}


class Notify(Action):
    """Send the match to the notifiers named in `to`, or to all of `NOTIFIERS`.

    Like `Alert`, it also marks the turn at `severity`, so it needs no Alert beside it.
    """

    kind = "notify"

    def __init__(self, to: list[str] | None = None, severity: str = "warning") -> None:
        self.to = list(to or [])
        self.severity = _severity(severity)

    def params(self) -> dict[str, Any]:
        return {"to": self.to, "severity": self.severity}


ACTION_KINDS: dict[str, type[Action]] = {
    cls.kind: cls for cls in (Alert, Mask, RestrictTools, Block, Queue, Notify)
}

# Actions are enforced in this order. After a block the rest do not matter.
# A rule's notice describes its first action in this order that the user can see.
ACTION_ORDER = ("block", "restrict_tools", "mask", "queue", "notify", "alert")
# Actions the user can see. The client is never told about a queue, alert or
# notification (SEC-20) unless the rule's notice says so.
USER_FACING = frozenset({"block", "restrict_tools", "mask"})


def build_from_kind[T](
    registry: Mapping[str, T], node: Mapping[str, Any], what: str
) -> tuple[T, dict[str, Any]]:
    """Look up `node["kind"]` in `registry`. Returns the entry and the rest of `node`."""
    rest = dict(node)
    kind = rest.pop("kind", "")
    if kind not in registry:
        raise ImproperlyConfigured(f"Unknown {what} {kind!r}; one of {sorted(registry)}")
    return registry[kind], rest


def action_from_dict(node: Mapping[str, Any]) -> Action:
    """Build an action from its JSON form, {"kind": "block", ...params}."""
    cls, params = build_from_kind(ACTION_KINDS, node, "action")
    return cls(**params)


__all__ = [
    "ACTION_KINDS",
    "ACTION_ORDER",
    "USER_FACING",
    "Action",
    "Alert",
    "Block",
    "Mask",
    "Notify",
    "Queue",
    "RestrictTools",
    "action_from_dict",
    "build_from_kind",
]
