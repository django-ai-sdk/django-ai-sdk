"""Conditions over lens answers. Written in Python, stored and edited as JSON.

    L("prompt_injection").is_true & (L("toxicity") > 0.8)
    L("pii").contains("bsn") | L("topic").in_({"hr", "legal"})
    ~L("grounded").is_true & L("sources").confident(0.7)

A condition wraps the JSON node it is stored as, so a rule from code and a rule
from a dict behave the same. Every comparison on a lens without a usable answer
is false. Use `L(name).missing` to test for that. `is_true` is a yes, or for an
Entities lens, that it found something; `is_false` the opposite.

A lens in `unknown` (its check or judge failed this turn) is unknown, not
false: a comparison on it is neither true nor false, so `~L("grounded").is_true`
does not match just because "grounded" failed. Only `missing` and `present` read
a failed lens like any other, through its error answer.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Set
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from django_ai_sdk.oversight.types import Answer

# op -> test(answer value, operand)
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
# Ops that read the answer itself, not its value.
META_OPS = {"missing", "present", "confident", "unsure"}


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    return float(value) if isinstance(value, int | float) else None


class Condition:
    """A test over a turn's answers, keyed by lens name. Raises ValueError for a bad node."""

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

    def lenses(self) -> set[str]:
        return set(_lenses(self.node))

    def decide(self, answers: Mapping[str, Answer], unknown: Set[str] = frozenset()) -> bool | None:
        """Return True, False, or None when a lens in `unknown` leaves it undecided."""
        return _decide(self.node, answers, unknown)


def _parts(condition: Condition, key: str) -> list[dict[str, Any]]:
    return list(condition.node[key]) if key in condition.node else [condition.node]


def _children(node: Mapping[str, Any]) -> list[dict[str, Any]]:
    if "not" in node:
        return [node["not"]]
    return node.get("all") or node.get("any") or []


def _lenses(node: Mapping[str, Any]) -> Iterator[str]:
    if "lens" in node:
        yield node["lens"]
    for child in _children(node):
        yield from _lenses(child)


def _decide(
    node: Mapping[str, Any], answers: Mapping[str, Answer], unknown: Set[str]
) -> bool | None:
    if "all" in node or "any" in node:
        # Kleene logic: one False decides an `all`, one True an `any`.
        decisive = "any" in node
        results = [_decide(child, answers, unknown) for child in _children(node)]
        if decisive in results:
            return decisive
        return None if None in results else not decisive
    if "not" in node:
        result = _decide(node["not"], answers, unknown)
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
    try:
        return bool(OPS[op](answer.value, operand))
    except TypeError:
        return False


def _validated(node: Any) -> dict[str, Any]:
    """Return `node` as a clean JSON node. Rejects a bad op or operand now, not on every turn."""
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
    """A lens by name. Its operators build conditions."""

    # __eq__ builds a condition, so an L cannot be hashed.
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

    # Returns a condition instead of a bool, so the signature differs from object's.
    def __eq__(self, value: object) -> Condition:  # ty: ignore[invalid-method-override]
        return self._test("eq", value)

    def __ne__(self, value: object) -> Condition:  # ty: ignore[invalid-method-override]
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


__all__ = ["Condition", "L"]
