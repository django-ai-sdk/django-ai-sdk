"""Turn a saved definition into a Policy class and validate it.

An authored policy is a normal `Policy` subclass with `origin = "authored"`, so it is
registered, built and run like a code policy. It gets the same checks as code, plus
these. A definition may not:

- use the name of a code policy, or a name that is not a slug
- change any lens or rule a parent has, so a lens or rule declared in code never
  changes at runtime: every one of them must come out of the build as its parent
  built it
- configure a lens beyond `INLINE_ARGS` (no regex, code or import path): keywords,
  presets, judge aliases and registered validators only. Models, classifiers and
  regexes must be aliases set by the deployment.
- skip a policy listed in `MUST_EXTEND`
- exceed MAX_LENSES lenses, MAX_RULES rules or MAX_BYTES of JSON
- set `fail_mode` "open" over a parent that fails closed
- build into what the system checks flag (`policy_errors`): the checks never see
  an authored policy, so it is held to them here
"""

from __future__ import annotations

import json
import re
from typing import Any

from django.core.exceptions import ImproperlyConfigured

from django_ai_sdk.oversight.actions import action_from_dict
from django_ai_sdk.oversight.authoring.conf import INLINE_ARGS, setting
from django_ai_sdk.oversight.checks.policies import policy_errors
from django_ai_sdk.oversight.judges.base import alias_defined, alias_of, is_optional
from django_ai_sdk.oversight.lenses import OPTIONS, Lens, lens_from_dict
from django_ai_sdk.oversight.policy import Policy, ResolvedPolicy, build
from django_ai_sdk.oversight.registry import get_invalid_policies, get_policy_class, validate_policy
from django_ai_sdk.oversight.resolve import code_ancestors
from django_ai_sdk.oversight.rules import Rule

NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,99}$")
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


def allowed_inline() -> set[str]:
    """What a definition may configure itself here: "keywords" and preset names."""
    return set(INLINE_ARGS)


def _judge_spec(spec: Any) -> Any:
    """Return `spec` if a lens may use it: a known alias, or Patterns of keywords."""
    if isinstance(spec, str):
        # A trailing "?" marks the alias optional. The lens skips it when it is not set up.
        if not alias_of(spec) or (not is_optional(spec) and not alias_defined(spec)):
            raise ValueError(f"no judge alias {alias_of(spec)!r} in AI_SDK_OVERSIGHT['JUDGES']")
        return spec
    if (
        isinstance(spec, dict)
        and spec.get("class") == "Patterns"
        and "keywords" in allowed_inline()
    ):
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
        if what not in allowed_inline():
            raise ValueError(
                f"lens {name!r}: {what!r} cannot be used here; one of {sorted(allowed_inline())}"
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
    if not str(data.get("name") or "").strip():
        raise ValueError("every rule has a name")
    then = data.get("then")
    if not isinstance(then, list):
        raise ValueError(f"rule {data.get('name')!r}: `then` is a list of actions")
    try:
        return Rule(
            str(data.get("name") or ""),
            when=data.get("when") or {},
            then=[action_from_dict(dict(action)) for action in then],
            mode=data.get("mode", "report"),
            message=str(data.get("message") or ""),
            notice=str(data.get("notice") or ""),
            description=str(data.get("description") or ""),
        )
    except (ImproperlyConfigured, KeyError, TypeError) as exc:
        raise ValueError(f"rule {data.get('name')!r}: {exc}") from None


def in_force(definition: dict[str, Any], *, active: bool) -> dict[str, Any]:
    """The definition that runs: without its own lenses and rules when it is off.

    Its parents and options stay, so what it extends keeps running.
    """
    return definition if active else {**definition, "lenses": [], "rules": []}


def build_policy_class(name: str, definition: dict[str, Any]) -> type[Policy]:
    """Build the Policy subclass for `definition`. Raises ValueError on a bad part."""
    options = dict(definition.get("options") or {})
    unknown = sorted(set(options) - POLICY_OPTIONS)
    if unknown:
        raise ValueError(f"unknown options {unknown}; one of {sorted(POLICY_OPTIONS)}")
    attrs: dict[str, Any] = {
        "__module__": __name__,
        "__qualname__": f"Authored.{name}",
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
    parents, problems = _parents(name, extends)
    try:
        cls = build_policy_class(name, definition)
    except ValueError as exc:
        raise DefinitionError([*problems, str(exc)]) from None
    _fail_on(problems)
    try:
        validate_policy(cls)
        resolved = build(name, policy_class=cls)
    except (ImproperlyConfigured, LookupError, ValueError) as exc:
        raise DefinitionError([str(exc)]) from None
    _fail_on(_changed_from_parents(parents, resolved))
    _fail_on(
        [
            f"rule {own.name!r} reads lenses it does not have: {unknown}"
            for own in cls.rules
            if (unknown := sorted(own.lenses() - set(resolved.lenses)))
        ]
    )
    # Open would let a failed lens pass the rules it inherits from a policy that fails closed.
    if resolved.fail_mode == "open":
        closed = [parent for parent in extends if build(parent).fail_mode != "open"]
        _fail_on([f"fail_mode 'open' would loosen {closed}, which fail closed"] if closed else [])
    _fail_on(policy_errors(name, resolved))
    return cls, resolved


def _shape_problems(name: str, definition: dict[str, Any], extends: list[str]) -> list[str]:
    """Problems with the name, size and parents, found without building anything."""
    problems: list[str] = []
    if not NAME.match(name or ""):
        problems.append("the name is lowercase letters, digits, '-' and '_', up to 100")
    from django_ai_sdk.oversight.authoring.loader import loader

    existing = get_policy_class(name)
    if existing is not None and not is_authored(existing):
        problems.append(f"{name!r} is a policy declared in code: pick another name")
    elif name in get_invalid_policies() and name not in loader.broken:
        # Code declared it and it failed its checks: its agents must stay refused.
        problems.append(f"{name!r} is a policy declared in code that does not load")
    if len(json.dumps(definition, default=str)) > MAX_BYTES:
        problems.append(f"the definition is larger than {MAX_BYTES // 1024} KiB")
    if len(definition.get("lenses") or []) > MAX_LENSES:
        problems.append(f"at most {MAX_LENSES} lenses")
    if len(definition.get("rules") or []) > MAX_RULES:
        problems.append(f"at most {MAX_RULES} rules")
    missing = [p for p in extends if get_policy_class(p) is None]
    if missing:
        problems.append(f"no policy {missing} to extend")
    if name in extends:
        problems.append("a policy cannot extend itself")
    return problems


def _parents(name: str, extends: list[str]) -> tuple[dict[str, ResolvedPolicy], list[str]]:
    """Build each parent, and list a missing MUST_EXTEND or a parent that does not load."""
    problems: list[str] = []
    required = [p for p in setting("MUST_EXTEND") if p not in code_ancestors(name, parents=extends)]
    if required:
        problems.append(f"every authored policy extends {required} (MUST_EXTEND)")
    parents: dict[str, ResolvedPolicy] = {}
    for parent in extends:
        try:
            parents[parent] = build(parent)
        except Exception as exc:
            problems.append(f"policy {parent!r} does not load: {exc}")
    return parents, problems


def _changed_from_parents(
    parents: dict[str, ResolvedPolicy], resolved: ResolvedPolicy
) -> list[str]:
    """Every lens and rule a parent has must come out of the build unchanged.

    Compared after the build, so neither a lens or rule of the same name, nor a preset
    that takes its default name, nor a later parent can replace an earlier one's.
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


def _fail_on(problems: list[str]) -> None:
    if problems:
        raise DefinitionError(problems)


__all__ = [
    "MAX_BYTES",
    "MAX_LENSES",
    "MAX_RULES",
    "ORIGIN",
    "DefinitionError",
    "allowed_inline",
    "build_policy_class",
    "check_definition",
    "in_force",
    "is_authored",
]
