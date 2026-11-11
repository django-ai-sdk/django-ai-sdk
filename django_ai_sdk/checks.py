"""System checks for the core app, and the pieces the opt-in apps' checks share.

These catch misconfigurations that would not fail any request. Ids are
`ai_sdk.E001` and up. Each opt-in app has its own `PREFIX`, such as
`ai_sdk.oversight`, and reports a bad settings dict as `<PREFIX>.E012`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from django.core.checks import Error
from django.core.exceptions import ImproperlyConfigured

from django_ai_sdk.utils import resolve_setting

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    # Returns what is wrong with a value, or "" when it is fine.
    Validator = Callable[[Any], str]

PREFIX = "ai_sdk"
# The same check, a bad settings dict, has the same number in every app.
SETTINGS_CHECK = "E012"
TASKS_HINT = (
    "Configure TASKS with a backend that has a worker, such as "
    "django_tasks_db.DatabaseBackend with ./manage.py db_worker."
)
OVERSIGHT_APP = "django_ai_sdk.oversight"


def check_oversight_installed(app_configs: Any = None, **kwargs: Any) -> list[Any]:
    """Report policies that are set while the oversight app is not installed.

    Without the app, those agents run unguarded. An agent wants a policy when it
    sets `oversight` or overrides `get_oversight()`. A DEFAULT_POLICY in
    AI_SDK_OVERSIGHT applies to every agent.
    """
    from django.apps import apps
    from django.conf import settings

    if apps.is_installed(OVERSIGHT_APP):
        return []
    messages: list[Any] = []
    hint = f"Add {OVERSIGHT_APP!r} to INSTALLED_APPS, or remove the policy."
    configured = getattr(settings, "AI_SDK_OVERSIGHT", None)
    default = configured.get("DEFAULT_POLICY") if isinstance(configured, dict) else None
    if default:
        messages.append(
            Error(
                f"AI_SDK_OVERSIGHT['DEFAULT_POLICY'] is {default!r}, but {OVERSIGHT_APP} is "
                "not installed: no turn is checked.",
                hint=hint,
                id=f"{PREFIX}.E001",
            )
        )
    from django_ai_sdk.agents.registry import registered_agent_classes

    for cls in registered_agent_classes():
        if _asks_for_policy(cls):
            messages.append(
                Error(
                    f"Agent {cls.__qualname__} asks for an oversight policy, but "
                    f"{OVERSIGHT_APP} is not installed: its turns run unguarded.",
                    hint=hint,
                    id=f"{PREFIX}.E001",
                )
            )
    return messages


def _asks_for_policy(cls: type) -> bool:
    from django_ai_sdk.agents.base import Agent

    if getattr(cls, "oversight", None) not in (None, False):
        return True
    return getattr(cls, "get_oversight", None) is not Agent.get_oversight


def one_of(*choices: Any) -> Validator:
    def check(value: Any) -> str:
        if value in choices:
            return ""
        return f"is {value!r}, not one of {', '.join(repr(choice) for choice in choices)}"

    return check


def of_type(*types: type, label: str) -> Validator:
    """Accept an instance of `types`. A bool is not a number here."""

    def check(value: Any) -> str:
        if isinstance(value, types) and not (isinstance(value, bool) and bool not in types):
            return ""
        return f"is {value!r}, not {label}"

    return check


def number(minimum: float, *, above: bool = False, allow_none: bool = False) -> Validator:
    """Accept an int or float of at least `minimum` (more than it, with `above`)."""
    bound = f"above {minimum}" if above else f"of at least {minimum}"
    label = f"a number {bound}{' or None' if allow_none else ''}"

    def check(value: Any) -> str:
        if value is None and allow_none:
            return ""
        if isinstance(value, bool) or not isinstance(value, int | float):
            return f"is {value!r}, not {label}"
        if value < minimum or (above and value == minimum):
            return f"is {value!r}, not {label}"
        return ""

    return check


class AppSettings:
    """One settings dict, declared once as `{key: (default, validator, fix hint)}`.

    Call it with a key for `settings.<name>[key]`, or its default; a dict value merges
    one level deep into a dict default. A value that is not a dict raises
    ImproperlyConfigured, and the app's E012 check reports it.
    """

    def __init__(self, name: str, keys: dict[str, tuple[Any, Validator, str]]) -> None:
        self.name = name
        self.keys = keys
        self.defaults = {key: spec[0] for key, spec in keys.items()}

    def _configured(self) -> Any:
        return resolve_setting(self.name, None) or {}

    @property
    def broken(self) -> bool:
        """Whether the dict is set to something else. Checks that read it skip."""
        return not isinstance(self._configured(), dict)

    def __call__(self, key: str) -> Any:
        configured = self._configured()
        if not isinstance(configured, dict):
            raise ImproperlyConfigured(f"{self.name} is a {type(configured).__name__}, not a dict.")
        default = self.defaults[key]
        value = configured.get(key, default)
        if isinstance(value, dict) and isinstance(default, dict):
            return {**default, **value}
        return value

    def problems(self) -> list[tuple[str, str, str]]:
        """List (key, problem, fix). The key is "" when the setting is not a dict at all."""
        configured = self._configured()
        if not isinstance(configured, dict):
            kind = type(configured).__name__
            return [("", f"is a {kind}, not a dict", "Set a dict, or remove it.")]
        found: list[tuple[str, str, str]] = []
        for key, value in configured.items():
            if key not in self.keys:
                found.append(
                    (str(key), "is not a setting", f"The keys are {', '.join(self.keys)}.")
                )
            elif problem := self.keys[key][1](value):
                found.append((str(key), problem, self.keys[key][2]))
        return found

    def errors(self, prefix: str) -> list[Any]:
        """One `<prefix>.E012` per problem."""
        return [
            Error(
                f"{self.name}{f'[{key!r}]' if key else ''} {problem}.",
                hint=fix,
                id=f"{prefix}.{SETTINGS_CHECK}",
            )
            for key, problem, fix in self.problems()
        ]


def names(value: Any) -> str:
    """Accept a list or tuple of strings."""
    if isinstance(value, list | tuple) and all(isinstance(item, str) for item in value):
        return ""
    return f"is {value!r}, not a list of names"


def per_database[T](
    databases: Iterable[str] | None, read: Callable[[str], T]
) -> Iterator[tuple[str, T]]:
    """Yield (alias, read(alias)) for each database a check is given.

    A database whose tables are not migrated yet has nothing to check, so it is skipped.
    """
    from django.db import DatabaseError

    for alias in databases or ():
        try:
            found = read(alias)
        except DatabaseError:
            continue
        yield alias, found


def task_backend_problem(task: Any) -> str:
    """Why `task` would not run on a worker, or "" when its backend has one.

    Phrased to follow "<what> ...", such as "Oversight's reflect stage and notifications are queued but never run".
    """
    from django_tasks.backends.dummy import DummyBackend
    from django_tasks.backends.immediate import ImmediateBackend

    try:
        backend = task.get_backend()
    except Exception as exc:  # noqa: BLE001 - a backend that does not load is the finding
        return f"are never queued: the task backend {task.backend!r} cannot be loaded: {exc}"
    if isinstance(backend, ImmediateBackend):
        problem = "run inline, inside the request that queues them"
    elif isinstance(backend, DummyBackend):
        problem = "are queued but never run"
    else:
        return ""
    return f"{problem}: TASKS uses {type(backend).__name__}"


__all__ = [
    "SETTINGS_CHECK",
    "AppSettings",
    "TASKS_HINT",
    "check_oversight_installed",
    "number",
    "of_type",
    "names",
    "one_of",
    "per_database",
    "task_backend_problem",
]
