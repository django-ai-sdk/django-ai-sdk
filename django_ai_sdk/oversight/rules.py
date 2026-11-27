from __future__ import annotations

from collections.abc import Iterator, Mapping, Set
from typing import TYPE_CHECKING, Any, ClassVar

from django.core.exceptions import ImproperlyConfigured
from django.utils.translation import gettext_lazy as _

from django_ai_sdk.oversight.types import Mode, Outcome, Severity, Stage, rank

if TYPE_CHECKING:
    from django.utils.functional import Promise

    from django_ai_sdk.oversight.types import Answer

OPS: dict[str, Any] = {
    "is_true": lambda v, _: v is True or (isinstance(v, list) and bool(v)),
    "is_false": lambda v, _: v is False or v == [],
    "eq": lambda v, x: v == x,
    "ne": lambda v, x: v != x,
    "gt": lambda v, x: _num(v) is not None and _num(v) > x,
    "ge": lambda v, x: _num(v) is not None and _num(v) >= x,
    "lt": lambda v, x: _num(v) is not None and _num(v) < x,
    "le": lambda v, x: _num(v) is not None and _num(v) <= x,
    "in": lambda v, x: v in x if not isinstance(v, list) else bool(set(v) & set(x)),
    "contains": lambda v, x: isinstance(v, list | str) and x in v,
}
META_OPS = {"missing", "present", "confident", "unsure", "disputed"}


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    return float(value) if isinstance(value, int | float) else None


class Condition:
    """Raises ValueError for a bad node."""

    __slots__ = ("node",)

    def __init__(self, node: Mapping[str, Any]) -> None:
        self.node = _validated(node)

    def __and__(self, other: Condition) -> Condition:
        return Condition({"all": _parts(self, "all") + _parts(other, "all")})

    def __or__(self, other: Condition) -> Condition:
        return Condition({"any": _parts(self, "any") + _parts(other, "any")})

    def __invert__(self) -> Condition:
        return Condition({"not": self.node})

    def __repr__(self) -> str:
        return f"<Condition {self.node}>"

    def to_dict(self) -> dict[str, Any]:
        return self.node

    def lenses(self, op: str = "") -> set[str]:
        """The lenses the condition reads; with `op`, only those it tests with it."""
        return {leaf["lens"] for leaf in _leaves(self.node) if not op or leaf["op"] == op}

    def matches(
        self, answers: Mapping[str, Answer], unknown: Set[str] = frozenset()
    ) -> bool | None:
        """Return True, False, or None when a lens in `unknown` leaves it undecided."""
        return _node_matches(self.node, answers, unknown)


def _parts(condition: Condition, key: str) -> list[dict[str, Any]]:
    return list(condition.node[key]) if key in condition.node else [condition.node]


def _children(node: Mapping[str, Any]) -> list[dict[str, Any]]:
    if "not" in node:
        return [node["not"]]
    return node.get("all") or node.get("any") or []


def _leaves(node: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    if "lens" in node:
        yield node
    for child in _children(node):
        yield from _leaves(child)


def _node_matches(
    node: Mapping[str, Any], answers: Mapping[str, Answer], unknown: Set[str]
) -> bool | None:
    if "all" in node or "any" in node:
        decisive = "any" in node
        results = [_node_matches(child, answers, unknown) for child in _children(node)]
        if decisive in results:
            return decisive
        return None if None in results else not decisive
    if "not" in node:
        result = _node_matches(node["not"], answers, unknown)
        return None if result is None else not result
    lens, op, operand = node["lens"], node["op"], node.get("value")
    answer = answers.get(lens)
    usable = answer is not None and answer.ok
    if not usable and lens in unknown and (op not in {"missing", "present"} or answer is None):
        return None
    if op == "missing":
        return not usable
    if op == "present":
        return usable
    if not usable:
        return False
    if op == "confident":
        return answer.confidence >= node["value"]
    if op == "unsure":
        return answer.confidence < node["value"]
    if op == "disputed":
        return answer.disputed
    try:
        return bool(OPS[op](answer.value, operand))
    except TypeError:
        return False


def _validated(node: Any) -> dict[str, Any]:
    if not isinstance(node, Mapping):
        raise ValueError(f"{node!r} is not a condition")
    for key in ("all", "any"):
        if key in node:
            return {key: [_validated(child) for child in node[key]]}
    if "not" in node:
        return {"not": _validated(node["not"])}
    if "lens" not in node or "op" not in node:
        raise ValueError(f"{dict(node)!r} is not a condition")
    lens, op, value = str(node["lens"]), str(node["op"]), node.get("value")
    if op not in OPS and op not in META_OPS:
        raise ValueError(f"Unknown op {op!r}")
    if op in {"confident", "unsure"}:
        if _num(value) is None or not 0 <= value <= 1:
            raise ValueError(f"{op!r} needs a confidence from 0 to 1, not {value!r}")
    elif op in {"gt", "ge", "lt", "le"} and _num(value) is None:
        raise ValueError(f"{op!r} needs a number, not {value!r}")
    elif op == "in" and not isinstance(value, list | tuple):
        raise ValueError(f"'in' needs a list of values, not {value!r}")
    leaf: dict[str, Any] = {"lens": lens, "op": op}
    if value is not None:
        leaf["value"] = list(value) if op == "in" else value
    return leaf


class L:
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, name: str) -> None:
        self.name = name

    def _test(self, op: str, value: Any = None) -> Condition:
        return Condition({"lens": self.name, "op": op, "value": value})

    @property
    def is_true(self) -> Condition:
        return self._test("is_true")

    @property
    def is_false(self) -> Condition:
        return self._test("is_false")

    @property
    def missing(self) -> Condition:
        return self._test("missing")

    @property
    def present(self) -> Condition:
        return self._test("present")

    @property
    def disputed(self) -> Condition:
        return self._test("disputed")

    def __eq__(self, value: object) -> Condition:  # ty: ignore[incompatible-method-override]
        return self._test("eq", value)

    def __ne__(self, value: object) -> Condition:  # ty: ignore[incompatible-method-override]
        return self._test("ne", value)

    def __gt__(self, value: float) -> Condition:
        return self._test("gt", value)

    def __ge__(self, value: float) -> Condition:
        return self._test("ge", value)

    def __lt__(self, value: float) -> Condition:
        return self._test("lt", value)

    def __le__(self, value: float) -> Condition:
        return self._test("le", value)

    def in_(self, values: Any) -> Condition:
        return self._test("in", list(values))

    def contains(self, value: str) -> Condition:
        return self._test("contains", value)

    def confident(self, at_least: float) -> Condition:
        return self._test("confident", at_least)

    def unsure(self, below: float) -> Condition:
        return self._test("unsure", below)


class Action:
    kind: ClassVar[str] = ""
    outcome: ClassVar[Outcome] = Outcome.FLAGGED
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
    kind = "alert"

    def __init__(self, severity: str = "warning") -> None:
        self.severity = _severity(severity)

    def params(self) -> dict[str, Any]:
        return {"severity": self.severity}


class Mask(Action):
    """Mask the spans `lenses` found (default: every lens the rule reads)."""

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
    """An empty list removes all tools."""

    kind = "restrict_tools"
    outcome = Outcome.RESTRICTED
    effective = (Stage.GATE, Stage.TOOL)

    def __init__(self, names: list[str] | None = None) -> None:
        self.names = list(names or [])

    def params(self) -> dict[str, Any]:
        return {"names": self.names}


class Block(Action):
    kind = "block"
    outcome = Outcome.BLOCKED
    effective = (Stage.GATE, Stage.TOOL, Stage.FINISH)

    DEFAULT_MESSAGE = _("This request can't be answered here.")

    def __init__(self, message: str | Promise = "") -> None:
        self.message = message or self.DEFAULT_MESSAGE

    def params(self) -> dict[str, Any]:
        return {"message": str(self.message)}


class Queue(Action):
    kind = "queue"

    def __init__(self, queue: str = "default") -> None:
        self.queue = queue

    def params(self) -> dict[str, Any]:
        return {"queue": self.queue}


class Notify(Action):
    """Send the match to notifiers `to` (default all of NOTIFIERS); also alerts at `severity`."""

    kind = "notify"

    def __init__(self, to: list[str] | None = None, severity: str = "warning") -> None:
        self.to = list(to or [])
        self.severity = _severity(severity)

    def params(self) -> dict[str, Any]:
        return {"to": self.to, "severity": self.severity}


ACTION_KINDS: dict[str, type[Action]] = {
    cls.kind: cls for cls in (Alert, Mask, RestrictTools, Block, Queue, Notify)
}

ACTION_ORDER = ("block", "restrict_tools", "mask", "queue", "notify", "alert")
USER_FACING = frozenset({"block", "restrict_tools", "mask"})


def build_from_kind[T](
    registry: Mapping[str, T], node: Mapping[str, Any], what: str
) -> tuple[T, dict[str, Any]]:
    rest = dict(node)
    kind = rest.pop("kind", "")
    if kind not in registry:
        raise ImproperlyConfigured(f"Unknown {what} {kind!r}; one of {sorted(registry)}")
    return registry[kind], rest


def action_from_dict(node: Mapping[str, Any]) -> Action:
    cls, params = build_from_kind(ACTION_KINDS, node, "action")
    return cls(**params)


class Rule:
    """Reports only until its mode is raised."""

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
        self.when = when if isinstance(when, Condition) else Condition(when)
        self.then = [then] if isinstance(then, Action) else list(then)
        self.mode = Mode(mode)
        self.message = message
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
