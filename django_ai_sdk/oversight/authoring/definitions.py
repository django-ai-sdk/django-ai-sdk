"""Turn a saved definition into a Policy class and validate it.

An authored policy is a normal `Policy` subclass with `origin = "authored"`, so it is
registered, built and run like a code policy. It gets the same checks as code, plus
these. A definition may not:

- use the name of a code policy, or a name that is not a slug
- extend another authored policy: only code policies and packs
- declare a lens or rule named like one it inherits, or extend parents where a later
  one replaces an earlier one's: a lens or rule declared in code never changes at runtime
- configure a lens beyond `INLINE_ARGS` (no regex, code or import path): keywords,
  presets, judge aliases and registered validators only. Models, classifiers and
  regexes must be aliases set by the deployment.
- skip a policy listed in `MUST_EXTEND`
- exceed MAX_EXTENDS parents, MAX_LENSES lenses, MAX_RULES rules or MAX_BYTES of JSON
- keep more than a parent: a higher `capture`, a longer `retention`, or `fail_mode`
  "open" over a parent that fails closed
- build into what the system checks flag (`policy_errors`): the checks never see
  an authored policy, so it is held to them here
"""

from __future__ import annotations

import json
import re
from typing import Any

from django.core.exceptions import ImproperlyConfigured

from django_ai_sdk.oversight.authoring.conf import INLINE_ARGS, setting
from django_ai_sdk.oversight.checks.policies import policy_errors
from django_ai_sdk.oversight.judges.base import alias_defined, alias_of, is_optional
from django_ai_sdk.oversight.lenses import OPTIONS, Lens, lens_from_dict
from django_ai_sdk.oversight.policy import Policy, ResolvedPolicy, build, policy_lenses
from django_ai_sdk.oversight.registry import get_invalid_policies, get_policy_class, validate_policy
from django_ai_sdk.oversight.resolve import ancestors
from django_ai_sdk.oversight.rules import Rule
from django_ai_sdk.oversight.types import Capture, rank

NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,99}$")
MAX_EXTENDS = 20
MAX_LENSES = 50
MAX_RULES = 100
MAX_BYTES = 256 * 1024
RULE_KEYS = frozenset({"name", "when", "then", "mode", "message", "notice", "description"})
POLICY_OPTIONS = frozenset({"capture", "stream_mode", "fail_mode", "retention"})
# What any authored lens may set, and what each kind adds. Nothing here takes a
# regex or names code: validators are names a developer registered.
LENS_KEYS = frozenset(
    {"kind", "preset", "name", "label", "subject", "stages", "sample", "enabled"}
    | {"needs_grounding", "answer", *OPTIONS}
)
KIND_KEYS = {
    "pattern": frozenset(),
    "check": frozenset({"validators"}),
    "judge": frozenset({"judges", "min_confidence", "question", "guidance", "reads_question"}),
}
ORIGIN = "authored"


class DefinitionError(ValueError):
    """A definition that cannot be saved. `problems` lists every reason."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("; ".join(problems))


def is_authored(cls: type[Policy] | None) -> bool:
    return cls is not None and getattr(cls, "origin", "") == ORIGIN


def _judge_spec(spec: Any) -> Any:
    """Return `spec` if a lens may use it: a known alias, or Patterns of keywords."""
    if isinstance(spec, str):
        # A trailing "?" marks the alias optional. The lens skips it when it is not set up.
        if not alias_of(spec) or (not is_optional(spec) and not alias_defined(spec)):
            raise ValueError(f"no judge alias {alias_of(spec)!r} in AI_SDK_OVERSIGHT['JUDGES']")
        return spec
    if isinstance(spec, dict) and spec.get("class") == "Patterns" and "keywords" in INLINE_ARGS:
        extra = sorted(set(spec) - {"class", "name"} - INLINE_ARGS["keywords"])
        if extra:
            raise ValueError(f"Patterns takes only keywords here, not {extra}")
        return dict(spec)
    raise ValueError(f"judge {spec!r} is an alias, or Patterns with keywords")


def _lens(config: Any) -> Lens:
    """Build an authored lens. Raises ValueError for anything outside the safe set."""
    if not isinstance(config, dict):
        raise ValueError("a lens is an object")
    config = dict(config)
    name, kind, preset = config.get("name"), config.get("kind"), config.get("preset", "")
    if preset or (kind == "pattern" and "keywords" in config):
        what = preset or "keywords"
        if what not in INLINE_ARGS:
            raise ValueError(
                f"lens {name!r}: {what!r} cannot be used here; one of {sorted(INLINE_ARGS)}"
            )
        allowed = LENS_KEYS | INLINE_ARGS[what]
    else:
        allowed = LENS_KEYS | KIND_KEYS.get(str(kind), frozenset())
    extra = sorted(set(config) - allowed)
    if extra:
        raise ValueError(f"lens {name!r} does not take {extra} here")
    if isinstance(config.get("judges"), list):
        config["judges"] = [_judge_spec(spec) for spec in config["judges"]]
    try:
        return lens_from_dict(config)
    except (ImproperlyConfigured, TypeError) as exc:
        raise ValueError(f"lens {name!r}: {exc}") from None


def _rule(data: Any) -> Rule:
    if not isinstance(data, dict):
        raise ValueError("a rule is an object")
    extra = sorted(set(data) - RULE_KEYS)
    if extra:
        raise ValueError(f"rule {data.get('name')!r} does not take {extra}")
    if not isinstance(data.get("name"), str) or not data["name"].strip():
        raise ValueError("every rule has a name")
    if not isinstance(data.get("then"), list):
        raise ValueError(f"rule {data.get('name')!r}: `then` is a list of actions")
    try:
        return Rule.from_dict(data)
    except (ImproperlyConfigured, KeyError, TypeError) as exc:
        raise ValueError(f"rule {data.get('name')!r}: {exc}") from None


def build_policy_class(name: str, definition: dict[str, Any]) -> type[Policy]:
    """Build the Policy subclass for `definition`. Raises ValueError on a bad part."""
    options = dict(definition.get("options") or {})
    unknown = sorted(set(options) - POLICY_OPTIONS)
    if unknown:
        raise ValueError(f"unknown options {unknown}; one of {sorted(POLICY_OPTIONS)}")
    attrs: dict[str, Any] = {
        "name": name,
        "label": str(definition.get("label") or ""),
        "description": str(definition.get("description") or ""),
        "extends": [str(parent) for parent in definition.get("extends") or []],
        "lenses": [_lens(config) for config in definition.get("lenses") or []],
        "rules": [_rule(data) for data in definition.get("rules") or []],
        "origin": ORIGIN,
        **{key: options[key] for key in POLICY_OPTIONS if options.get(key) not in (None, "")},
    }
    return type(f"Authored_{name.replace('-', '_')}", (Policy,), attrs)


def check_definition(name: str, definition: dict[str, Any]) -> tuple[type[Policy], ResolvedPolicy]:
    """Validate `definition` and return its class and resolved policy.

    Raises DefinitionError with the problems found. Shape problems stop it early.
    Inheritance and build problems are reported together. Validation runs only when
    both pass.
    """
    extends = [str(p) for p in definition.get("extends") or []]
    _fail_on(_shape_problems(name, definition, extends))
    parents, problems = _parents(extends)
    try:
        cls = build_policy_class(name, definition)
    except ValueError as exc:
        raise DefinitionError([*problems, str(exc)]) from None
    _fail_on(problems)
    _fail_on(_inherited_names(parents, cls))
    try:
        validate_policy(cls)
        resolved = build(name, policy_class=cls)
    except (ImproperlyConfigured, LookupError, ValueError) as exc:
        raise DefinitionError([str(exc)]) from None
    _fail_on(_changed_from_parents(parents, resolved))
    _fail_on(_looser_than_parents(parents, resolved))
    _fail_on(policy_errors(name, resolved))
    return cls, resolved


def _shape_problems(name: str, definition: dict[str, Any], extends: list[str]) -> list[str]:
    """Problems with the name, size and parents, found without building anything."""
    problems: list[str] = []
    if not NAME.match(name or ""):
        problems.append("the name is lowercase letters, digits, '-' and '_', up to 100")
    existing = get_policy_class(name)
    if existing is not None and not is_authored(existing):
        problems.append(f"{name!r} is a policy declared in code: pick another name")
    elif name in get_invalid_policies():
        # Code declared it and it failed its checks: its agents must stay refused.
        problems.append(f"{name!r} is a policy declared in code that does not load")
    if len(json.dumps(definition, default=str)) > MAX_BYTES:
        problems.append(f"the definition is larger than {MAX_BYTES // 1024} KiB")
    if len(extends) > MAX_EXTENDS:
        problems.append(f"at most {MAX_EXTENDS} policies to extend")
    if len(definition.get("lenses") or []) > MAX_LENSES:
        problems.append(f"at most {MAX_LENSES} lenses")
    if len(definition.get("rules") or []) > MAX_RULES:
        problems.append(f"at most {MAX_RULES} rules")
    missing = [p for p in extends if get_policy_class(p) is None]
    if missing:
        problems.append(f"no policy {missing} to extend")
    authored = [p for p in extends if is_authored(get_policy_class(p))]
    if authored:
        problems.append(f"{authored} are authored: extend only policies declared in code and packs")
    if name in extends:
        problems.append("a policy cannot extend itself")
    return problems


def _parents(extends: list[str]) -> tuple[dict[str, ResolvedPolicy], list[str]]:
    """Build each parent, and list a missing MUST_EXTEND or a parent that does not load."""
    problems: list[str] = []
    inherited = {*extends, *(ancestor for parent in extends for ancestor in ancestors(parent))}
    required = [p for p in setting("MUST_EXTEND") if p not in inherited]
    if required:
        problems.append(f"every authored policy extends {required} (MUST_EXTEND)")
    parents: dict[str, ResolvedPolicy] = {}
    for parent in extends:
        try:
            parents[parent] = build(parent)
        except Exception as exc:
            problems.append(f"policy {parent!r} does not load: {exc}")
    return parents, problems


def _inherited_names(parents: dict[str, ResolvedPolicy], cls: type[Policy]) -> list[str]:
    """Own lenses and rules may not take a name a parent has, a preset's default included."""
    own_lenses = {lens.name for lens in policy_lenses(cls)}
    own_rules = {rule.name for rule in cls.rules}
    return [
        f"{what} {item!r} of {parent!r} is replaced: name yours differently"
        for parent, built in parents.items()
        for what, own, inherited in (
            ("lens", own_lenses, built.lenses),
            ("rule", own_rules, built.rules),
        )
        for item in sorted(own & set(inherited))
    ]


def _changed_from_parents(
    parents: dict[str, ResolvedPolicy], resolved: ResolvedPolicy
) -> list[str]:
    """Every lens and rule a parent has must come out of the build unchanged.

    Own names are refused before the build, so this catches a later parent replacing
    an earlier one's.
    """
    problems: list[str] = []
    for parent, built in parents.items():
        for lens_name, lens in built.lenses.items():
            kept = resolved.lenses.get(lens_name)
            # Each build copies the parent's lens, so an unchanged one has equal fields.
            if kept is None or type(kept) is not type(lens) or vars(kept) != vars(lens):
                problems.append(
                    f"lens {lens_name!r} of {parent!r} is replaced: name yours differently"
                )
        for rule_name, rule in built.rules.items():
            kept_rule = resolved.rules.get(rule_name)
            if kept_rule is None or kept_rule.to_dict() != rule.to_dict():
                problems.append(
                    f"rule {rule_name!r} of {parent!r} is replaced: name yours differently"
                )
    return problems


def _looser_than_parents(parents: dict[str, ResolvedPolicy], resolved: ResolvedPolicy) -> list[str]:
    """A policy may keep less than each parent, never more.

    Each value is the built one, so a parent's retention includes the RETENTION setting
    for a kind it does not set, and the policy's own includes what another parent sets.
    """
    problems: list[str] = []
    for parent, built in parents.items():
        if rank(tuple(Capture), resolved.capture) > rank(tuple(Capture), built.capture):
            problems.append(
                f"capture {str(resolved.capture)!r} keeps more than {parent!r}, "
                f"which captures {str(built.capture)!r}"
            )
        for kind, days in sorted(resolved.retention.items()):
            if days > built.retention.get(kind, days):
                problems.append(
                    f"{kind} retention of {days} days is longer than the "
                    f"{built.retention[kind]} days {parent!r} keeps "
                    "(its own retention, or the RETENTION setting)"
                )
        # Open would let a failed lens pass the rules a parent that fails closed enforces.
        if resolved.fail_mode == "open" and built.fail_mode != "open":
            problems.append(f"fail_mode 'open' would loosen {parent!r}, which fails closed")
    return problems


def _fail_on(problems: list[str]) -> None:
    if problems:
        raise DefinitionError(problems)


__all__ = [
    "MAX_BYTES",
    "MAX_EXTENDS",
    "MAX_LENSES",
    "MAX_RULES",
    "ORIGIN",
    "DefinitionError",
    "build_policy_class",
    "check_definition",
    "is_authored",
]
