"""System checks for oversight: a broken policy fails no request, so this is where it shows up."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Any

from django.core.checks import Error, Warning
from django.db import DatabaseError
from django.utils import timezone

from django_ai_sdk.oversight.agents import attribute_policy
from django_ai_sdk.oversight.conf import (
    DEFAULTS,
    FAIL_MODES,
    STREAM_MODES,
    UNGUARDED_MODES,
    configured,
    deadline_ms,
    is_enabled,
    is_retention,
    setting,
)
from django_ai_sdk.oversight.guard.judging import FAIL_CLOSED_ACTIONS
from django_ai_sdk.oversight.guard.masking import gate_masking_lenses
from django_ai_sdk.oversight.judges.base import Judge, alias_defined, build_judge
from django_ai_sdk.oversight.lenses import JudgeLens
from django_ai_sdk.oversight.notifiers import configured_notifiers
from django_ai_sdk.oversight.registry import (
    ancestors,
    build_policy,
    get_duplicate_policies,
    get_invalid_policies,
    get_policies,
)
from django_ai_sdk.oversight.rules import Mask, Notify
from django_ai_sdk.oversight.tasks import reflect_turn
from django_ai_sdk.oversight.types import INLINE_STAGES, Capture, Mode, Stage

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

PREFIX = "ai_sdk.oversight"


def check_oversight(app_configs: Any = None, **kwargs: Any) -> list[Any]:
    # E012: otherwise a misspelt key is silently ignored and a wrong value misread.
    found = settings_checks()
    if not isinstance(configured(), dict):
        return found
    in_use = policies_in_use()
    return [
        *policy_checks(in_use),
        *judge_checks(),
        *default_policy_checks(),
        *found,
        *agent_checks(),
    ]


def check_oversight_deploy(app_configs: Any = None, **kwargs: Any) -> list[Any]:
    """Production-only checks: unmasked content, unmeetable deadlines, no task worker."""
    if not isinstance(configured(), dict) or not is_enabled():
        return []
    return [*capture_checks(), *deadline_checks(), *task_backend_checks()]


def check_oversight_retention(
    app_configs: Any = None, databases: Any = None, **kwargs: Any
) -> list[Any]:
    """Reads only the databases it is given."""
    if not isinstance(configured(), dict) or not is_enabled():
        return []
    return [*retention_checks(databases), *sweep_checks(databases)]


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _is_positive_or_none(value: Any) -> bool:
    return value is None or (_is_number(value) and value > 0)


def _listed(choices: Iterable[str]) -> str:
    return ", ".join(repr(str(choice)) for choice in choices)


DEADLINE_STAGES = ("gate", "tool", "finish")
CLIENT_DETAILS = ("actions", "full")
CAPTURES = tuple(capture.value for capture in Capture)

# What each AI_SDK_OVERSIGHT key accepts: a test, and the words E012 shows when it fails.
ACCEPTS: dict[str, tuple[Callable[[Any], bool], str]] = {
    "ENABLED": (lambda value: isinstance(value, bool), "True or False"),
    "DEFAULT_POLICY": (lambda value: isinstance(value, str), 'a policy name, or "" for none'),
    "CAPTURE": (lambda value: value in CAPTURES, f"one of {_listed(CAPTURES)}"),
    "STREAM_MODE": (lambda value: value in STREAM_MODES, f"one of {_listed(STREAM_MODES)}"),
    "JUDGES": (
        lambda value: (
            isinstance(value, dict)
            and all(
                isinstance(alias, str) and isinstance(spec, str | dict | Judge)
                for alias, spec in value.items()
            )
        ),
        (
            'a dict by alias of {"class": ..., **kwargs} dicts, Judge instances or another'
            " alias's name, where \"class\" is a built-in judge's name or a dotted path, such as"
            ' {"judge": {"class": "LLMJudge"}}'
        ),
    ),
    "DEADLINE_MS": (
        lambda value: (
            isinstance(value, dict)
            and all(
                stage in DEADLINE_STAGES and _is_positive_or_none(ms) for stage, ms in value.items()
            )
        ),
        f"a dict of positive milliseconds or None by stage ({_listed(DEADLINE_STAGES)})",
    ),
    "MAX_INPUT_CHARS": (_is_positive_or_none, "a positive number of characters, or None"),
    "MAX_OTHER_TEXT_CHUNKS": (
        lambda value: type(value) is int and value >= 1,
        "a whole number of chunks of at least 1",
    ),
    "CLIENT_DETAIL": (lambda value: value in CLIENT_DETAILS, f"one of {_listed(CLIENT_DETAILS)}"),
    "FAIL_MODE": (lambda value: value in FAIL_MODES, f"one of {_listed(FAIL_MODES)}"),
    "RETENTION": (
        is_retention,
        'a dict of whole days of at least 1 by "content" and/or "metadata"',
    ),
    "NOTIFIERS": (
        lambda value: isinstance(value, list | tuple),
        (
            'a list of {"class": ..., **kwargs} dicts, Notifier instances or class names,'
            ' where a class name or "class" is a built-in notifier\'s name or a dotted path, such as'
            ' ["LogNotifier", {"class": "WebhookNotifier", "url": ...}]'
        ),
    ),
    "RUNTIME_CACHE_SECONDS": (
        lambda value: _is_number(value) and value >= 0,
        "a number of seconds of at least 0",
    ),
    "UNGUARDED_MODEL_CALLS": (
        lambda value: value in UNGUARDED_MODES,
        f"one of {_listed(UNGUARDED_MODES)}",
    ),
}


def settings_checks() -> list[Error]:
    user_settings = configured()
    if not isinstance(user_settings, dict):
        return [
            Error(
                f"AI_SDK_OVERSIGHT must be a dict, not a {type(user_settings).__name__}.",
                hint="Set a dict, or remove it.",
                id=f"{PREFIX}.E012",
            )
        ]
    errors = []
    for key, value in user_settings.items():
        if key not in ACCEPTS:
            errors.append(
                Error(
                    f"AI_SDK_OVERSIGHT has an unknown key {key!r}.",
                    hint=f"The keys are {', '.join(DEFAULTS)}.",
                    id=f"{PREFIX}.E012",
                )
            )
            continue
        is_accepted, accepted = ACCEPTS[key]
        if not is_accepted(value):
            errors.append(
                Error(f"AI_SDK_OVERSIGHT[{key!r}] must be {accepted}.", id=f"{PREFIX}.E012")
            )
    return errors


def policies_in_use() -> set[str]:
    """DEFAULT_POLICY, the policies agent classes name, and all their ancestors."""
    from django_ai_sdk.agents.registry import registered_agent_classes

    names = {str(setting("DEFAULT_POLICY") or "")}
    for cls in [*registered_agent_classes(), *agent_runtime_bases()]:
        try:
            names.add(agent_policy_name(cls))
        except Exception:
            continue  # E011 reports it.
    names.discard("")
    for name in list(names):
        names.update(ancestors(name))
    return names


def agent_runtime_bases() -> list[type]:
    from django_ai_sdk.agents.config import get_runtime_agent_bases

    try:
        return list(get_runtime_agent_bases())
    except Exception:
        return []


def agent_policy_name(cls: type) -> str:
    """ "" when the class names none or opts out. TypeError for a value naming no policy."""
    name, source = attribute_policy(cls)
    return name if source == "agent" else ""


def agent_checks() -> list[Any]:
    from django_ai_sdk.agents.registry import registered_agent_classes

    messages: list[Any] = []
    known = get_policies()
    for cls in registered_agent_classes():
        try:
            own = agent_policy_name(cls)
        except Exception as exc:
            messages.append(
                Error(
                    f"Agent {cls.__qualname__}.oversight names no policy: {exc}. Its turns "
                    "cannot be checked.",
                    hint="Set a Policy subclass, a registered policy's name, False or None.",
                    id=f"{PREFIX}.E011",
                )
            )
            continue
        if own and own not in known:
            messages.append(
                Error(
                    f"Agent {cls.__qualname__} names oversight policy {own!r}, which no app "
                    "registers: every turn it runs is blocked.",
                    hint="Check the name, and that the app declaring the policy is in "
                    "INSTALLED_APPS with the policy in its oversight.py.",
                    id=f"{PREFIX}.E009",
                )
            )
    return messages


def policy_checks(in_use: set[str]) -> list[Any]:
    """Notifier and streaming warnings only for policies in `in_use`."""
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
            resolved = build_policy(name)
        except Exception as exc:
            messages.append(
                Error(f"Oversight policy {name!r} does not build_judge: {exc}", id=f"{PREFIX}.E002")
            )
            continue
        messages.extend(notify_checks(name, resolved, notifiers, in_use=name in in_use))
        messages.extend(lens_checks(name, resolved))
        messages.extend(waiting_rule_checks(name, resolved))
        if name in in_use:
            messages.extend(stream_mode_checks(name, resolved))
    return messages


def notify_checks(name: str, resolved: Any, notifiers: set[str], *, in_use: bool) -> list[Any]:
    """Notify rules that would send to no one."""
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


def waiting_rule_checks(name: str, resolved: Any) -> list[Any]:
    """Enforced rules that wait for a lens judged too late for them to act."""
    messages: list[Any] = []
    gate_masking = {lens.name for lens in gate_masking_lenses(resolved)}
    for rule in resolved.active_rules():
        if rule.mode is not Mode.ENFORCE:
            continue
        lenses = [
            lens
            for n in sorted(rule.lenses())
            if (lens := resolved.lenses.get(n)) is not None and lens.enabled
        ]
        stopping = sorted({a.kind for a in rule.then} & {"block", "restrict_tools"})
        inline = [lens.name for lens in lenses if set(lens.stages) & set(INLINE_STAGES)]
        reflect_only = [lens.name for lens in lenses if set(lens.stages) == {Stage.REFLECT}]
        if stopping and inline and reflect_only:
            messages.append(
                Warning(
                    f"Enforced rule {rule.name!r} in policy {name!r} would "
                    f"{' or '.join(stopping)} on {inline}, but also reads {reflect_only}, "
                    f"judged only at reflect: when {inline} fail or leave it undecided, the "
                    "rule waits for reflect, where it can no longer act.",
                    hint="Judge those lenses inline too, or split the rule: one that blocks on "
                    "the inline lenses, and one for reflect that alerts or queues.",
                    id=f"{PREFIX}.W018",
                )
            )
        masks = [a for a in rule.then if isinstance(a, Mask)]
        masked = sorted(gate_masking & {n for a in masks for n in a.lenses or rule.lenses()})
        after_gate = [lens.name for lens in lenses if Stage.GATE not in lens.stages]
        if masked and after_gate:
            messages.append(
                Warning(
                    f"Enforced rule {rule.name!r} in policy {name!r} masks {masked} in the "
                    f"question, but also reads {after_gate}, judged after the gate: the rule "
                    "waits for them, so the model reads the question unmasked.",
                    hint="Split the rule: one that masks on the gate lenses, and one for the "
                    "others.",
                    id=f"{PREFIX}.W019",
                )
            )
    return messages


def stream_mode_checks(name: str, resolved: Any) -> list[Any]:
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
    """Undefined aliases, and enforced rules whose lenses no judge here answers.

    W008, not E007, when every such judge is an undefined optional alias: a deliberate choice.
    """
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
            if isinstance(alias, str) and not alias_defined(alias)
        )
        if lens.enabled and not any(_can_answer(judge, lens) for judge in built_judges(lens)):
            dead.add(lens.name)
            if not lens.usable_judges():
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


def _can_answer(judge: Any, lens: Any) -> bool:
    try:
        return not judge.missing() and judge.answers(lens)
    except Exception:
        return False


def built_judges(lens: Any) -> list[Any]:
    judges = []
    for spec in lens.usable_judges():
        try:
            judges.append(build_judge(spec))
        except Exception:
            continue
    return judges


def deadline_checks() -> list[Any]:
    """Enforced gate and tool rules whose judge may outlast the deadline.

    The lens then fails: under "closed" the rule blocks, under "open" it lets through unjudged.
    """
    messages: list[Any] = []
    for name in sorted(policies_in_use()):
        resolved = build_or_none(name)
        if resolved is not None:  # Otherwise E002 reports it.
            messages.extend(deadline_errors(name, resolved))
    return messages


def deadline_errors(name: str, resolved: Any) -> list[Any]:
    """Any judge with `external` set counts; `LocalClassifier` doesn't."""
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
                Error(
                    f"Enforced rule {rule.name!r} in policy {name!r} reads {reads} at "
                    f"{stage.value}, where it may ask a model judge "
                    f"with a timeout of {seconds:g} s against a "
                    f"{stage.value} deadline of {deadline:g} ms: on a slow answer the "
                    "lens fails, and the rule blocks (or, with FAIL_MODE 'open', lets "
                    "it through unjudged).",
                    hint="Answer the lens with a LocalClassifier "
                    f"instead, raise AI_SDK_OVERSIGHT['DEADLINE_MS']['{stage.value}'] to "
                    "the judge's timeout, or move the lens to reflect.",
                    id=f"{PREFIX}.E013",
                )
            )
    return messages


def policy_errors(name: str, resolved: Any) -> list[str]:
    """What the checks would flag in one built policy; authored policies are held to it on save."""
    found = [m for m in lens_checks(name, resolved) if m.is_serious()]
    found += [*stream_mode_checks(name, resolved), *deadline_errors(name, resolved)]
    return [m.msg for m in found]


def build_or_none(name: str) -> Any:
    """Not cached: a policy registered at runtime would leave a cache stale."""
    try:
        return build_policy(name)
    except Exception:
        return None


def judge_checks() -> list[Any]:
    messages: list[Any] = []
    for alias in setting("JUDGES") or {}:
        try:
            judge = build_judge(alias)
        except Exception as exc:
            messages.append(
                Error(f"Judge alias {alias!r} cannot be built: {exc}", id=f"{PREFIX}.E005")
            )
            continue
        if missing := judge.missing():
            messages.append(
                Warning(
                    f"Judge alias {alias!r} is {type(judge).__name__}, but {missing}: it abstains.",
                    id=f"{PREFIX}.W005",
                )
            )
    return messages


def default_policy_checks() -> list[Any]:
    default = setting("DEFAULT_POLICY")
    if default and default not in get_policies():
        return [
            Error(
                f"AI_SDK_OVERSIGHT['DEFAULT_POLICY'] names unknown policy {default!r}",
                id=f"{PREFIX}.E006",
            )
        ]
    return []


def capture_checks() -> list[Any]:
    if str(setting("CAPTURE")) != "full":
        return []
    return [
        Warning(
            "Oversight keeps every question and answer unmasked: "
            "AI_SDK_OVERSIGHT['CAPTURE'] is 'full'.",
            hint="Use CAPTURE 'masked' (the default).",
            id=f"{PREFIX}.W001",
        )
    ]


def task_backend_checks() -> list[Any]:
    from django_tasks.backends.dummy import DummyBackend
    from django_tasks.backends.immediate import ImmediateBackend

    try:
        backend = reflect_turn.get_backend()
    except Exception as exc:  # noqa: BLE001 - a backend that does not load is the finding
        problem = (
            f"are never queued: the task backend {reflect_turn.backend!r} cannot be loaded: {exc}"
        )
    else:
        if isinstance(backend, ImmediateBackend):
            problem = "run inline, inside the request that queues them"
        elif isinstance(backend, DummyBackend):
            problem = "are queued but never run"
        else:
            return []
        problem += f": TASKS uses {type(backend).__name__}"
    return [
        Warning(
            f"Oversight's reflect stage and notifications {problem}.",
            hint="Configure TASKS with a backend that has a worker, such as "
            "django_ai_sdk.durable.tasks.DBOSTaskBackend with ./manage.py ai_sdk_worker.",
            id=f"{PREFIX}.W014",
        )
    ]


PRUNE_DUE = timedelta(days=2)
# A crashed process leaves its turn open until oversight_sweep marks it interrupted.
SWEEP_DUE = timedelta(days=1)


def retention_checks(databases: Any) -> list[Any]:
    """Retention only stamps `expires_at`, so overdue rows mean nothing prunes."""
    from django_ai_sdk.oversight.models import Turn, TurnContent

    due = timezone.now() - PRUNE_DUE

    def overdue(alias: str) -> bool:
        return (
            Turn.objects.using(alias).filter(expires_at__lt=due).exists()
            or TurnContent.objects.using(alias).filter(expires_at__lt=due).exists()
        )

    return [
        Warning(
            f"Expired oversight rows in database {alias!r} are not deleted: "
            "oversight_prune does not run.",
            hint="Schedule `manage.py oversight_prune` daily; RETENTION only stamps "
            "expires_at on each row.",
            id=f"{PREFIX}.W007",
        )
        for alias, late in _per_database(databases, overdue)
        if late
    ]


def sweep_checks(databases: Any) -> list[Any]:
    from django_ai_sdk.oversight.models import Turn

    due = timezone.now() - SWEEP_DUE

    def stale(alias: str) -> bool:
        return Turn.objects.using(alias).filter(status="open", created_at__lt=due).exists()

    return [
        Warning(
            f"Oversight turns in database {alias!r} stay open for more than a day: "
            "oversight_sweep does not run.",
            hint="Schedule `manage.py oversight_sweep`, for example every 10 minutes. A turn "
            "whose run crashed or whose process died stays open until it is swept.",
            id=f"{PREFIX}.W017",
        )
        for alias, late in _per_database(databases, stale)
        if late
    ]


def _per_database[T](
    databases: Iterable[str] | None, read: Callable[[str], T]
) -> Iterator[tuple[str, T]]:
    """Skips databases whose tables are not migrated yet."""
    for alias in databases or ():
        try:
            found = read(alias)
        except DatabaseError:
            continue
        yield alias, found
