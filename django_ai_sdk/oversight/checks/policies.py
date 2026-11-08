"""Checks on each policy.

Covers lenses that never run or have no model judge left, gate rules a slow judge
can't meet, and answers masked at finish while they stream live.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from django.core.checks import Error, Warning

from django_ai_sdk.oversight.checks import PREFIX


def policy_checks(in_use: set[str]) -> list[Any]:
    """Check that every policy registers and builds, and its rules and lenses can run.

    Unanswered lenses are only reported for policies in `in_use`.
    """
    from django_ai_sdk.oversight.policy import build
    from django_ai_sdk.oversight.registry import (
        get_duplicate_policies,
        get_invalid_policies,
        get_policies,
    )

    messages: list[Any] = []
    # Collected across policies and reported as one warning.
    unanswered = Unanswered()
    for name, reason in get_invalid_policies().items():
        messages.append(
            Error(f"Oversight policy {name!r} was not registered: {reason}", id=f"{PREFIX}.E001")
        )
    for name, paths in get_duplicate_policies().items():
        messages.append(
            Error(
                f"Oversight policy name {name!r} is declared by {', '.join(paths)}: an agent "
                "naming it could run either. The first one registered keeps the name.",
                hint="Give each policy class a name of its own.",
                id=f"{PREFIX}.E010",
            )
        )
    for name in get_policies():
        try:
            resolved = build(name)
        except Exception as exc:
            messages.append(
                Error(f"Oversight policy {name!r} does not resolve: {exc}", id=f"{PREFIX}.E002")
            )
            continue
        for rule in resolved.rules.values():
            unknown = rule.lenses() - set(resolved.lenses)
            if unknown:
                messages.append(
                    Error(
                        f"Rule {rule.name!r} in policy {name!r} reads unknown lenses {sorted(unknown)}",
                        id=f"{PREFIX}.E003",
                    )
                )
        # Unanswered lenses only matter for policies in use.
        messages.extend(lens_checks(name, resolved, unanswered if name in in_use else None))
        if name in in_use:
            messages.extend(stream_mode_checks(name, resolved))
    if unanswered.lenses:
        messages.append(unanswered_warning(unanswered))
    return messages


def stream_mode_checks(name: str, resolved: Any) -> list[Any]:
    """Warn when a policy changes the answer at finish but streams it live."""
    if resolved.stream_mode != "live":
        return []
    acting = resolved.changes_answer_at_finish()
    if not acting:
        return []
    return [
        Warning(
            f"Policy {name!r} masks or withholds the answer at finish ({', '.join(acting)}), "
            "but streams it live: the client has the text before finish judges it.",
            hint="Set stream_mode 'auto' on the policy (or AI_SDK_OVERSIGHT['STREAM_MODE']), "
            "or use a client that applies the data-oversight 'replace'.",
            id=f"{PREFIX}.W011",
        )
    ]


def lens_checks(name: str, resolved: Any, unanswered: Unanswered | None) -> list[Any]:
    """Report lenses that never run on this deployment, or only run in part.

    Lenses with no model judge left are added to `unanswered` so the caller can
    report them in one warning.
    """
    from django_ai_sdk.oversight.checks.lenses import lens_health
    from django_ai_sdk.oversight.judges.base import alias_of, is_optional
    from django_ai_sdk.oversight.types import Mode

    messages: list[Any] = []
    dead: set[str] = set()
    # Lenses whose judges are all optional and undefined. That is a deliberate choice.
    optional: set[str] = set()
    for lens in resolved.lenses.values():
        health = lens_health(lens)
        messages.extend(
            Error(
                f"Lens {lens.name!r} in policy {name!r} names judge alias {alias!r}, "
                "which AI_SDK_OVERSIGHT['JUDGES'] does not define",
                id=f"{PREFIX}.E004",
            )
            for alias in health["undefined"]
        )
        if not lens.enabled:
            continue
        if not health["runnable"]:
            dead.add(lens.name)
            if all(is_optional(spec) for spec in lens.judges):
                optional.add(lens.name)
        if unanswered is not None and (not health["runnable"] or health["unjudged"]):
            shown = f"{lens.name} (patterns only)" if health["runnable"] else lens.name
            unanswered.lenses.setdefault(shown, set()).add(name)
            unanswered.aliases.update(alias_of(spec) for spec in health["skipped"])
            unanswered.cannot_run |= not health["skipped"]
        if health["unreachable"]:
            messages.append(
                Warning(
                    f"Lens {lens.name!r} in policy {name!r} never asks "
                    f"{', '.join(health['unreachable'])}: a Patterns judge before them is "
                    "already sure when it finds nothing.",
                    id=f"{PREFIX}.W004",
                )
            )
        if health["stages_missing_subject"]:
            stages = ", ".join(str(s) for s in health["stages_missing_subject"])
            messages.append(
                Warning(
                    f"Lens {lens.name!r} in policy {name!r} reads the {lens.subject} at "
                    f"{stages}, which never reads it: it never runs there.",
                    id=f"{PREFIX}.W006",
                )
            )
    for rule in resolved.rules.values():
        blind = sorted(rule.lenses() & dead)
        if rule.mode is not Mode.ENFORCE or not blind:
            continue
        if set(blind) <= optional:
            messages.append(
                Warning(
                    f"Enforced rule {rule.name!r} in policy {name!r} reads {blind}, whose "
                    "judges are all optional and none is defined here: it never acts.",
                    id=f"{PREFIX}.W008",
                )
            )
        else:
            messages.append(
                Error(
                    f"Enforced rule {rule.name!r} in policy {name!r} reads {blind}, which no "
                    "judge here can answer.",
                    id=f"{PREFIX}.E007",
                )
            )
    return messages


@dataclass
class Unanswered:
    """Judge lenses with no model judge left, across policies, for one warning."""

    # Lens (tagged "(patterns only)" when its patterns still answer it) to its policies.
    lenses: dict[str, set[str]] = field(default_factory=dict)
    # Optional aliases missing from AI_SDK_OVERSIGHT["JUDGES"].
    aliases: set[str] = field(default_factory=set)
    # Some judges aren't optional aliases but still can't build or run.
    cannot_run: bool = False


def unanswered_warning(unanswered: Unanswered) -> Any:
    """One warning listing every lens with no model judge left, grouped by policy."""
    by_policies: dict[tuple[str, ...], list[str]] = {}
    for lens, policies in sorted(unanswered.lenses.items()):
        by_policies.setdefault(tuple(sorted(policies)), []).append(lens)
    where = "; ".join(
        f"in {', '.join(policies)}: {', '.join(lenses)}"
        for policies, lenses in sorted(by_policies.items())
    )
    hints = []
    if unanswered.aliases:
        hints.append(
            f"Define {', '.join(repr(a) for a in sorted(unanswered.aliases))} in "
            "AI_SDK_OVERSIGHT['JUDGES'] to run them."
        )
    if unanswered.cannot_run:
        hints.append(
            f"Some name judges that cannot be built or run here: see {PREFIX}.E004, E005 and W005."
        )
    count = len(unanswered.lenses)
    return Warning(
        f"{count} oversight lens{'es have' if count != 1 else ' has'} no model judge here: "
        f"{where}. A lens without patterns is skipped on every turn.",
        hint=" ".join(hints),
        id=f"{PREFIX}.W003",
    )


def deadline_checks() -> list[Any]:
    """Find enforced gate and tool rules whose lenses may wait on a judge slower than the deadline.

    A judge is asked whenever the judges before it are unsure, and a pattern miss
    below `min_confidence` counts as unsure. At the deadline the stage abandons the
    judge and the lens fails. With FAIL_MODE "closed" the rule then refuses the turn
    or the tool call. With "open" it lets it through unjudged.
    """
    from django_ai_sdk.oversight.checks.agents import policies_in_use

    messages: list[Any] = []
    for name in sorted(policies_in_use()):
        resolved = build_or_none(name)
        if resolved is not None:  # Otherwise E002 reports it.
            messages.extend(deadline_errors(name, resolved))
    return messages


def deadline_errors(name: str, resolved: Any) -> list[Any]:
    """E013 for one built policy: see deadline_checks."""
    from django_ai_sdk.oversight.conf import deadline_ms
    from django_ai_sdk.oversight.guard.asking import FAIL_CLOSED_ACTIONS
    from django_ai_sdk.oversight.lenses import JudgeLens
    from django_ai_sdk.oversight.types import Mode, Stage

    messages: list[Any] = []
    for stage in (Stage.GATE, Stage.TOOL):
        deadline = deadline_ms(stage)
        if not deadline:
            continue
        slow = {
            lens.name: seconds
            for lens in resolved.active_lenses(stage)
            if isinstance(lens, JudgeLens) and (seconds := _slow_judge(lens, deadline)) is not None
        }
        for rule in resolved.active_rules():
            protective = [
                a for a in rule.then if a.kind in FAIL_CLOSED_ACTIONS and stage in a.effective
            ]
            reads = sorted(rule.lenses() & set(slow))
            if rule.mode is not Mode.ENFORCE or not protective or not reads:
                continue
            seconds = max(slow[lens] for lens in reads)
            messages.append(
                Error(
                    f"Enforced rule {rule.name!r} in policy {name!r} reads {reads} at "
                    f"{stage.value}, where a model judge is asked whenever the judges "
                    f"before it are unsure, with a timeout of {seconds:g} s against a "
                    f"{stage.value} deadline of {deadline:g} ms: on a slow answer the "
                    "lens fails, and the rule refuses (or, with FAIL_MODE 'open', lets "
                    "it through unjudged).",
                    hint="Patterns first do not avoid it: a pattern miss is unsure, so "
                    "the judge is still asked. Answer the lens with a LocalClassifier "
                    f"instead, raise AI_SDK_OVERSIGHT['DEADLINE_MS']['{stage.value}'] to "
                    "the judge's timeout, or move the lens to reflect.",
                    id=f"{PREFIX}.E013",
                )
            )
    return messages


def policy_errors(name: str, resolved: Any) -> list[str]:
    """What the system checks would flag in one built policy, as messages.

    Rules that can never act (E007), protective rules a slow judge fails (E013) and
    answers masked after they stream (W011). The checks run at deploy and see code
    policies only, so an authored policy and a rule switch are held to this on save.
    """
    found = [m for m in lens_checks(name, resolved, None) if m.is_serious()]
    found += [*stream_mode_checks(name, resolved), *deadline_errors(name, resolved)]
    return [m.msg for m in found]


def build_or_none(name: str) -> Any:
    """Build policy `name`, or None when it does not build (E002 reports why).

    Not cached: each check is registered on its own, and a policy registered at
    runtime would leave a cache stale.
    """
    from django_ai_sdk.oversight.policy import build

    try:
        return build(name)
    except Exception:
        return None


def _slow_judge(lens: Any, deadline: float) -> float | None:
    """Longest timeout among reachable external judges slower than `deadline`, or None.

    Any judge with `external` set counts, a user's own too. `LocalClassifier` doesn't.
    """
    from django_ai_sdk.oversight.checks.lenses import reachable_judges

    timeouts = [
        judge.timeout
        for _, judge in reachable_judges(lens)
        if judge is not None and judge.external and judge.timeout * 1000 > deadline
    ]
    return max(timeouts) if timeouts else None
