"""Conditions over lens answers. Written in Python, stored and edited as JSON.

    L("prompt_injection").is_true & (L("toxicity") > 0.8)
    L("pii").contains("bsn") | L("topic").in_({"hr", "legal"})
    ~L("grounded").is_true & L("sources").confident(0.7)

Both forms use the same evaluator, so a rule from code and a rule from a dict
behave the same. Every comparison on a lens without a usable answer is
false. Use `L(name).missing` to test for that. `is_true` is a yes, or for an
Entities lens, that it found something; `is_false` the opposite.

A failed lens (its check or judge raised or timed out) is unknown, not false. A
comparison on it is neither true nor false, so `~L("grounded").is_true` does not
match just because "grounded" failed. Only `missing` and `present` treat a failed
lens like any other. A lens that answers at a later stage is unknown to every test,
so `L("gate").is_true | L("reflect").is_true` matches at the gate once "gate" is.
"""

from __future__ import annotations

from collections.abc import Mapping, Set
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
# Ops that check whether a lens answered. They treat a failed lens like any other.
PRESENCE_OPS = {"missing", "present"}


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    return float(value) if isinstance(value, int | float) else None


class Condition:
    """A test over a turn's answers, keyed by lens name."""

    def evaluate(self, answers: Mapping[str, Answer], failed: Set[str] = frozenset()) -> bool:
        """Whether the condition is known to match.

        `failed` names the lenses that failed this turn. Comparisons on them are
        unknown, and an unknown condition does not match.
        """
        return self.judge(answers, failed) is True

    def judge(
        self, answers: Mapping[str, Answer], failed: Set[str], pending: Set[str] = frozenset()
    ) -> bool | None:
        """Return True, False, or None when a failed or `pending` lens makes it unknown.

        `pending` names lenses that answer at a later stage. Every test on one is
        unknown, `missing` too, so only what is known already can decide.
        """
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    def lenses(self) -> set[str]:
        raise NotImplementedError

    def __and__(self, other: Condition) -> Condition:
        return All([*_flatten(self, All), *_flatten(other, All)])

    def __or__(self, other: Condition) -> Condition:
        return AnyOf([*_flatten(self, AnyOf), *_flatten(other, AnyOf)])

    def __invert__(self) -> Condition:
        return Not(self)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.to_dict()}>"


def _flatten(condition: Condition, kind: type[_Group]) -> list[Condition]:
    return list(condition.items) if isinstance(condition, kind) else [condition]


class Compare(Condition):
    """One test on one lens's answer."""

    def __init__(self, lens: str, op: str, value: Any = None) -> None:
        if op not in OPS and op not in META_OPS:
            raise ValueError(f"Unknown op {op!r}")
        # Reject a bad operand now instead of failing on every turn.
        if op in {"confident", "unsure"}:
            if _num(value) is None or not 0 <= float(value) <= 1:
                raise ValueError(f"{op!r} needs a confidence from 0 to 1, not {value!r}")
        elif op in {"gt", "ge", "lt", "le"} and _num(value) is None:
            raise ValueError(f"{op!r} needs a number, not {value!r}")
        elif op == "in" and not isinstance(value, list | tuple | set | frozenset):
            raise ValueError(f"'in' needs a list of values, not {value!r}")
        self.lens = lens
        self.op = op
        # A number, a list, a label or None, depending on `op`.
        self.value: Any = sorted(value) if isinstance(value, set | frozenset) else value

    def judge(
        self, answers: Mapping[str, Answer], failed: Set[str], pending: Set[str] = frozenset()
    ) -> bool | None:
        if self.lens in pending:
            return None
        answer = answers.get(self.lens)
        usable = answer is not None and answer.ok
        if not usable and self.lens in failed and self.op not in PRESENCE_OPS:
            return None
        if self.op == "missing":
            return not usable
        if self.op == "present":
            return usable
        if not usable:
            return False
        if self.op == "confident":
            return answer.confidence >= float(self.value)
        if self.op == "unsure":
            return answer.confidence < float(self.value)
        try:
            return bool(OPS[self.op](answer.value, self.value))
        except TypeError:
            return False

    def to_dict(self) -> dict[str, Any]:
        node: dict[str, Any] = {"lens": self.lens, "op": self.op}
        if self.value is not None:
            node["value"] = self.value
        return node

    def lenses(self) -> set[str]:
        return {self.lens}


class _Group(Condition):
    """A list of conditions, stored under `key` in the JSON form."""

    key = ""

    def __init__(self, items: list[Condition]) -> None:
        self.items = items

    def to_dict(self) -> dict[str, Any]:
        return {self.key: [item.to_dict() for item in self.items]}

    def lenses(self) -> set[str]:
        return set().union(*(item.lenses() for item in self.items))


class All(_Group):
    key = "all"

    def judge(
        self, answers: Mapping[str, Answer], failed: Set[str], pending: Set[str] = frozenset()
    ) -> bool | None:
        results = [item.judge(answers, failed, pending) for item in self.items]
        if False in results:
            return False
        return None if None in results else True


class AnyOf(_Group):
    # Not `Any`, which would shadow typing.Any in this module.
    key = "any"

    def judge(
        self, answers: Mapping[str, Answer], failed: Set[str], pending: Set[str] = frozenset()
    ) -> bool | None:
        results = [item.judge(answers, failed, pending) for item in self.items]
        if True in results:
            return True
        return None if None in results else False


class Not(Condition):
    def __init__(self, item: Condition) -> None:
        self.item = item

    def judge(
        self, answers: Mapping[str, Answer], failed: Set[str], pending: Set[str] = frozenset()
    ) -> bool | None:
        result = self.item.judge(answers, failed, pending)
        return None if result is None else not result

    def to_dict(self) -> dict[str, Any]:
        return {"not": self.item.to_dict()}

    def lenses(self) -> set[str]:
        return self.item.lenses()


class Always(Condition):
    """Matches every turn, e.g. for a rule that always alerts."""

    def judge(
        self, answers: Mapping[str, Answer], failed: Set[str], pending: Set[str] = frozenset()
    ) -> bool | None:
        return True

    def to_dict(self) -> dict[str, Any]:
        return {"always": True}

    def lenses(self) -> set[str]:
        return set()


class L:
    """A lens by name. Its operators build Compare conditions."""

    # __eq__ builds a condition, so an L cannot be hashed.
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, name: str) -> None:
        self.name = name

    @property
    def is_true(self) -> Compare:
        return Compare(self.name, "is_true")

    @property
    def is_false(self) -> Compare:
        return Compare(self.name, "is_false")

    @property
    def missing(self) -> Compare:
        return Compare(self.name, "missing")

    @property
    def present(self) -> Compare:
        return Compare(self.name, "present")

    # Returns a condition instead of a bool, so the signature differs from object's.
    def __eq__(self, value: object) -> Compare:  # ty: ignore[invalid-method-override]
        return Compare(self.name, "eq", value)

    def __ne__(self, value: object) -> Compare:  # ty: ignore[invalid-method-override]
        return Compare(self.name, "ne", value)

    def __gt__(self, value: float) -> Compare:
        return Compare(self.name, "gt", value)

    def __ge__(self, value: float) -> Compare:
        return Compare(self.name, "ge", value)

    def __lt__(self, value: float) -> Compare:
        return Compare(self.name, "lt", value)

    def __le__(self, value: float) -> Compare:
        return Compare(self.name, "le", value)

    def in_(self, values: Any) -> Compare:
        return Compare(self.name, "in", list(values))

    def contains(self, value: str) -> Compare:
        return Compare(self.name, "contains", value)

    def confident(self, at_least: float) -> Compare:
        return Compare(self.name, "confident", at_least)

    def unsure(self, below: float) -> Compare:
        return Compare(self.name, "unsure", below)


def condition_from_dict(node: Mapping[str, Any] | bool) -> Condition:
    """Build a condition from its JSON form. Raises ValueError for anything else."""
    if node is True or (isinstance(node, Mapping) and node.get("always")):
        return Always()
    if not isinstance(node, Mapping):
        raise ValueError(f"{node!r} is not a condition")
    if "all" in node:
        return All([condition_from_dict(item) for item in node["all"]])
    if "any" in node:
        return AnyOf([condition_from_dict(item) for item in node["any"]])
    if "not" in node:
        return Not(condition_from_dict(node["not"]))
    if "lens" in node and "op" in node:
        return Compare(str(node["lens"]), str(node["op"]), node.get("value"))
    raise ValueError(f"{dict(node)!r} is not a condition")


__all__ = [
    "All",
    "Always",
    "AnyOf",
    "Compare",
    "Condition",
    "L",
    "Not",
    "condition_from_dict",
]
