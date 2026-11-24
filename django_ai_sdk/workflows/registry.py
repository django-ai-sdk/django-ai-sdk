from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from django.core.exceptions import ImproperlyConfigured
from django.utils.text import slugify

from django_ai_sdk.workflows.schemas import AGENT_STEP

if TYPE_CHECKING:
    from django_ai_sdk.workflows.listeners import RunListener
    from django_ai_sdk.workflows.models import Workflow
    from django_ai_sdk.workflows.schemas import (
        FieldDefinition,
        ListenerDefinition,
        WorkflowDefinition,
    )

logger = logging.getLogger(__name__)

# Matches Workflow.slug, so a declared and a stored name are one key space.
NAME_MAX_LENGTH = 100

# Reserved key for unstored, unregistered runs, so channel policies can tell them apart.
ADHOC = "adhoc"

_registry: dict[str, WorkflowDefinition] = {}

# Rejected declarations, name -> reason. Read by the system check.
_invalid: dict[str, str] = {}


def register_workflow(definition: WorkflowDefinition) -> WorkflowDefinition:
    """Register a WorkflowDefinition unless it cannot run; return it unchanged either way."""
    try:
        validate_name(definition.name)
        validate_definition(definition)
    except ImproperlyConfigured as exc:
        _invalid[definition.name or "<unnamed>"] = str(exc)
        logger.warning("Workflow not registered: %s", exc)
        return definition

    _invalid.pop(definition.name, None)
    existing = _registry.get(definition.name)
    if existing is not None and existing != definition:
        logger.warning(
            "Workflow %r is declared twice with different definitions — only one is "
            "reachable under that name, and which one depends on app-loading order. "
            "Give one of them a different `name`.",
            definition.name,
        )
    _registry[definition.name] = definition
    return definition


def validate_name(name: str) -> None:
    """Raise ImproperlyConfigured unless `name` is slug form, like Workflow.slug."""
    if not name:
        raise ImproperlyConfigured(
            "A workflow must have a non-empty `name` — it is the registry key."
        )
    if name == ADHOC:
        raise ImproperlyConfigured(
            f"Workflow name {ADHOC!r} is reserved for runs of a definition that is "
            f"neither stored nor registered. Pick another name."
        )
    expected = slugify(name)[:NAME_MAX_LENGTH]
    if name != expected:
        raise ImproperlyConfigured(
            f"Workflow name {name!r} is not a slug, so a Workflow row could never "
            f"collide with it. Use {expected!r}."
        )


def validate_definition(definition: WorkflowDefinition) -> None:
    """Raise ImproperlyConfigured unless the definition is legal to store and compile."""
    from django_ai_sdk.workflows.definitions import get_step_registry
    from django_ai_sdk.workflows.listeners import get_listener_registry
    from django_ai_sdk.workflows.runner import validate_steps

    label = f"Workflow {definition.name!r}"
    validate_steps(definition.steps, label=label)
    _compiles(
        f"Inputs_{definition.name or 'workflow'}",
        definition.input_fields,
        f"{label} input_fields",
    )

    step_types = get_step_registry()
    listener_types = get_listener_registry()
    declared_inputs = set(definition.input_fields)

    for index, step in enumerate(definition.steps):
        where = f"{label} step {index} ({step.name!r})"
        if step.type != AGENT_STEP:
            if step.type not in step_types:
                raise ImproperlyConfigured(
                    f"{where} has type {step.type!r}, which is not in "
                    f"AI_SDK_WORKFLOWS['STEPS']. Registered: {sorted(step_types) or 'none'}."
                )
            try:
                step_types[step.type].validate_config(step.config)
            except (TypeError, ValueError) as exc:
                raise ImproperlyConfigured(
                    f"{where} has a config its step type refuses: {exc}"
                ) from exc
        history = sorted(name for name in step.history if name not in declared_inputs)
        if history:
            raise ImproperlyConfigured(
                f"{where} sends history {history}, which input_fields does not "
                f"declare. Declared: {sorted(declared_inputs) or 'none'}."
            )
        _compiles(f"Output_{step.name}", step.output_fields, f"{where} output_fields")
        _validate_listeners(step.listeners, where, listener_types)

    _validate_listeners(definition.listeners, label, listener_types)
    _validate_triggers(definition, label)


def _validate_triggers(definition: WorkflowDefinition, label: str) -> None:
    """Every cron parses in its zone, and every input a trigger fills is declared."""
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

    from django_ai_sdk.workflows.schemas import CronTrigger

    declared = set(definition.input_fields)
    for trigger in definition.triggers:
        where = f"{label} trigger {trigger.name!r}"
        if declared and (extra := sorted(set(trigger.inputs) - declared)):
            raise ImproperlyConfigured(
                f"{where} fills inputs {extra}, which input_fields does not declare. "
                f"Declared: {sorted(declared)}."
            )
        if not isinstance(trigger, CronTrigger):
            continue
        # Five fields, so nothing fires more than once a minute.
        if len(trigger.cron.split()) != 5 or not _cron_is_valid(trigger.cron):
            raise ImproperlyConfigured(
                f"{where} has cron {trigger.cron!r}, which is not a 5-field cron expression "
                "(minute hour day-of-month month day-of-week)."
            )
        try:
            ZoneInfo(trigger.timezone)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ImproperlyConfigured(
                f"{where} has timezone {trigger.timezone!r}, which is not an IANA zone such "
                "as 'Europe/Amsterdam'."
            ) from exc


def _cron_is_valid(expression: str) -> bool:
    try:
        from croniter import croniter
    except ImportError:  # pragma: no cover - croniter comes with the dbos extra
        return True
    return bool(croniter.is_valid(expression))


def _compiles(name: str, fields: dict[str, FieldDefinition], where: str) -> None:
    """Build the schema now: a field like `model_dump` passes name rules but kills the worker."""
    from django_ai_sdk.workflows.definitions import model_from_fields

    if not fields:
        return
    try:
        model_from_fields(name, fields)
    except Exception as exc:
        raise ImproperlyConfigured(f"{where} cannot compile: {exc}") from exc


def _validate_listeners(
    listeners: list[ListenerDefinition], where: str, listener_types: dict[str, type[RunListener]]
) -> None:
    """Every listener a definition names must be one the deployment exposed."""
    for listener in listeners:
        if listener.type not in listener_types:
            raise ImproperlyConfigured(
                f"{where} names listener {listener.type!r}, which is not in "
                f"AI_SDK_WORKFLOWS['LISTENERS']. Registered: {sorted(listener_types) or 'none'}."
            )
        try:
            listener_types[listener.type].validate_config(listener.config)
        except (TypeError, ValueError) as exc:
            raise ImproperlyConfigured(
                f"{where} listener {listener.type!r} has a config it refuses: {exc}"
            ) from exc


def is_registered(definition: WorkflowDefinition) -> bool:
    """Whether `definition` is the workflow registered under its name, unchanged."""
    return bool(definition.name) and _registry.get(definition.name) == definition


def get_declared_workflows() -> dict[str, WorkflowDefinition]:
    """Workflows declared in code, keyed by name."""
    return dict(_registry)


def get_invalid_workflows() -> dict[str, str]:
    """Declarations the registry refused, name -> reason. Read by the system check."""
    return dict(_invalid)


async def aget_workflow(name: str) -> WorkflowDefinition | None:
    """One workflow by name; registered wins over stored. A broken stored one raises."""
    declared, row = await afind(name)
    return declared if row is None else definition_of(row)


async def afind(name: str) -> tuple[WorkflowDefinition | None, Workflow | None]:
    """(registered definition, None), (None, active row) or (None, None).

    The row is not parsed, so a caller can check permissions on it first.
    """
    from django_ai_sdk.workflows.models import Workflow

    declared = _registry.get(name)
    if declared is not None:
        return declared, None
    return None, await Workflow.objects.filter(slug=name, active=True).afirst()


def definition_of(row: Workflow) -> WorkflowDefinition:
    """Parse and validate a stored workflow, named by its slug."""
    from django_ai_sdk.workflows.schemas import WorkflowDefinition

    definition = WorkflowDefinition.model_validate({**row.definition, "name": row.slug})
    validate_definition(definition)
    return definition


def reset_registry() -> None:
    """Clear the registry — for tests."""
    _registry.clear()
    _invalid.clear()


__all__ = [
    "ADHOC",
    "NAME_MAX_LENGTH",
    "afind",
    "aget_workflow",
    "definition_of",
    "get_declared_workflows",
    "get_invalid_workflows",
    "is_registered",
    "register_workflow",
    "reset_registry",
    "validate_definition",
    "validate_name",
]
