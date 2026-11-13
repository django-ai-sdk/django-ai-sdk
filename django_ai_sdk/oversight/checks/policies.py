"""Checks on each policy.

Covers enforced rules no judge can answer, gate rules a slow judge can't meet, and
answers masked at finish while they stream live.
"""

from __future__ import annotations

from typing import Any

from django.core.checks import Error, Warning

from django_ai_sdk.oversight.checks import PREFIX


def policy_checks(in_use: set[str]) -> list[Any]:
    """Check that every policy registers and builds, and its rules and lenses can run.

    Warnings about notifiers and streaming are only reported for policies in `in_use`.
    """
    from django_ai_sdk.oversight.notifiers import configured_notifiers
    from django_ai_sdk.oversight.policy import build
    from django_ai_sdk.oversight.registry import (
        get_duplicate_policies,
        get_invalid_policies,
        get_policies,
    )

    notifiers = {notifier.name for notifier in configured_notifiers()}
    messages: list[Any] = []
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
        messages.extend(notify_checks(name, resolved, notifiers, in_use=name in in_use))
        messages.extend(lens_checks(name, resolved))
        if name in in_use:
            messages.extend(stream_mode_checks(name, resolved))
    return messages


def notify_checks(name: str, resolved: Any, notifiers: set[str], *, in_use: bool) -> list[Any]:
    """Find Notify rules that would send to no one.

    A name in `to` that no notifier has is an error. An enforced Notify while no
    notifier is configured is a warning, for policies in use.
    """
    from django_ai_sdk.oversight.actions import Notify
    from django_ai_sdk.oversight.types import Mode

    messages: list[Any] = []
    for rule in resolved.active_rules():
        for action in rule.then:
            if not isinstance(action, Notify):
                continue
            unknown = sorted(set(action.to) - notifiers)
            if notifiers and unknown:
                messages.append(
                    Error(
                        f"Rule {rule.name!r} in policy {name!r} notifies {unknown}, which no "
                        "notifier in AI_SDK_OVERSIGHT['NOTIFIERS'] is named.",
                        hint=f"The notifiers are {sorted(notifiers)}.",
                        id=f"{PREFIX}.E014",
                    )
                )
            if not notifiers and in_use and rule.mode is Mode.ENFORCE:
                messages.append(
                    Warning(
                        f"Enforced rule {rule.name!r} in policy {name!r} notifies, but "
                        "AI_SDK_OVERSIGHT['NOTIFIERS'] is empty: nobody is told.",
                        id=f"{PREFIX}.W015",
                    )
                )
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


def lens_checks(name: str, resolved: Any) -> list[Any]:
    """Report undefined judge aliases, and enforced rules that read lenses no judge here answers.

    Such a rule is an error, or a warning when every judge of those lenses is an
    optional alias left undefined: that is a deliberate choice.
    """
    from django_ai_sdk.oversight.judges.base import alias_defined, is_optional
    from django_ai_sdk.oversight.lenses import JudgeLens
    from django_ai_sdk.oversight.types import Mode

    messages: list[Any] = []
    dead: set[str] = set()
    optional: set[str] = set()
    for lens in resolved.lenses.values():
        if not isinstance(lens, JudgeLens):
            continue
        messages.extend(
            Error(
                f"Lens {lens.name!r} in policy {name!r} names judge alias {alias!r}, "
                "which AI_SDK_OVERSIGHT['JUDGES'] does not define",
                id=f"{PREFIX}.E004",
            )
            for alias in lens.judges
            if isinstance(alias, str) and not is_optional(alias) and not alias_defined(alias)
        )
        if lens.enabled and all(judge.missing() for judge in built_judges(lens)):
            dead.add(lens.name)
            if all(is_optional(spec) for spec in lens.judges):
                optional.add(lens.name)
    for rule in resolved.rules.values():
        blind = sorted(rule.lenses() & dead)
        if rule.mode is not Mode.ENFORCE or not blind:
            continue
        if set(blind) <= optional:
            messages.append(
                Warning(
                    f"Enforced rule {rule.name!r} in policy {name!r} reads {blind}, whose "
                    "judges are all optional and none is defined here: it never acts.",
                    id=f"{PREFIX}.E007",
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


def built_judges(lens: Any) -> list[Any]:
    """The judges `lens` may ask that build here. E004 and E005 report the others."""
    from django_ai_sdk.oversight.judges.base import resolve

    judges = []
    for spec in lens.usable_judges():
        try:
            judges.append(resolve(spec))
        except Exception:
            continue
    return judges


def deadline_checks() -> list[Any]:
    """Find enforced gate and tool rules whose lenses may wait on a judge slower than the deadline.

    At the deadline the stage abandons the judge and the lens fails. With FAIL_MODE
    "closed" the rule then refuses the turn or the tool call. With "open" it lets it
    through unjudged.
    """
    from django_ai_sdk.oversight.checks.agents import policies_in_use

    messages: list[Any] = []
    for name in sorted(policies_in_use()):
        resolved = build_or_none(name)
        if resolved is not None:  # Otherwise E002 reports it.
            messages.extend(deadline_warnings(name, resolved))
    return messages


def deadline_warnings(name: str, resolved: Any) -> list[Any]:
    """E013 for one built policy: see deadline_checks.

    Any judge with `external` set counts, a user's own too. `LocalClassifier` doesn't.
    """
    from django_ai_sdk.oversight.conf import deadline_ms
    from django_ai_sdk.oversight.guard.asking import FAIL_CLOSED_ACTIONS
    from django_ai_sdk.oversight.lenses import JudgeLens
    from django_ai_sdk.oversight.types import Mode, Stage

    messages: list[Any] = []
    for stage in (Stage.GATE, Stage.TOOL):
        deadline = deadline_ms(stage)
        if not deadline:
            continue
        slow: dict[str, float] = {}
        for lens in resolved.active_lenses(stage):
            if not isinstance(lens, JudgeLens):
                continue
            timeouts = [
                judge.timeout
                for judge in built_judges(lens)
                if judge.external and judge.timeout * 1000 > deadline
            ]
            if timeouts:
                slow[lens.name] = max(timeouts)
        for rule in resolved.active_rules():
            protective = [
                a for a in rule.then if a.kind in FAIL_CLOSED_ACTIONS and stage in a.effective
            ]
            reads = sorted(rule.lenses() & set(slow))
            if rule.mode is not Mode.ENFORCE or not protective or not reads:
                continue
            seconds = max(slow[lens] for lens in reads)
            messages.append(
                Warning(
                    f"Enforced rule {rule.name!r} in policy {name!r} reads {reads} at "
                    f"{stage.value}, where it may ask a model judge "
                    f"with a timeout of {seconds:g} s against a "
                    f"{stage.value} deadline of {deadline:g} ms: on a slow answer the "
                    "lens fails, and the rule refuses (or, with FAIL_MODE 'open', lets "
                    "it through unjudged).",
                    hint="Answer the lens with a LocalClassifier "
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
    policies only, so an authored policy is held to this on save.
    """
    found = [m for m in lens_checks(name, resolved) if m.is_serious()]
    found += [*stream_mode_checks(name, resolved), *deadline_warnings(name, resolved)]
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
