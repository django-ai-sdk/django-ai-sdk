"""Process-wide registry of declared workflows.

Every installed app's `workflows` module is imported on startup, so calling
`register()` there is enough. A definition that cannot run is kept out of the
registry and reported by the `django_ai_sdk.workflows` check instead of failing boot.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from django.core.exceptions import ImproperlyConfigured
from django.utils.text import slugify

if TYPE_CHECKING:
    from django_ai_sdk.workflows.actions import WorkflowAction
    from django_ai_sdk.workflows.schemas import (
        ActionDefinition,
        FieldDefinition,
        WorkflowDefinition,
    )

logger = logging.getLogger(__name__)

# Matches WorkflowSettings.slug, so a declared and a stored name are one key space.
NAME_MAX_LENGTH = 100

_registry: dict[str, WorkflowDefinition] = {}

# Rejected declarations, name -> reason. Read by the system check.
_invalid: dict[str, str] = {}

# Warned once per process; rows are re-read on every dispatch.
_warned_shadowed: set[str] = set()
_warned_invalid: set[str] = set()


def register(definition: WorkflowDefinition) -> WorkflowDefinition:
    """Add a WorkflowDefinition to the registry unless it cannot run.

    Returns it unchanged either way, so a declaration stays importable on its own.
    """
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
    """Raise ImproperlyConfigured unless `name` is slug form, like WorkflowSettings.slug."""
    if not name:
        raise ImproperlyConfigured(
            "A workflow must have a non-empty `name` — it is the registry key."
        )
    expected = slugify(name)[:NAME_MAX_LENGTH]
    if name != expected:
        raise ImproperlyConfigured(
            f"Workflow name {name!r} is not a slug, so a WorkflowSettings row could never "
            f"collide with it. Use {expected!r}."
        )


def validate_definition(definition: WorkflowDefinition) -> None:
    """Raise ImproperlyConfigured unless the definition is legal to store and compile.

    The step graph is the runner's rule. What is left is the two registries: a step
    type or an action the deployment did not expose cannot be composed.
    """
    from django_ai_sdk.workflows.actions import get_action_registry
    from django_ai_sdk.workflows.definitions import get_step_registry
    from django_ai_sdk.workflows.runner import check_pipeline

    label = f"Workflow {definition.name!r}"
    check_pipeline(definition.steps, label=label)
    _compiles(
        f"Inputs_{definition.name or 'workflow'}",
        definition.input_fields,
        f"{label} input_fields",
    )

    step_types = get_step_registry()
    action_types = get_action_registry()
    declared_inputs = set(definition.input_fields)

    for index, step in enumerate(definition.steps):
        where = f"{label} step {index} ({step.name!r})"
        if step.type != "agent" and step.type not in step_types:
            raise ImproperlyConfigured(
                f"{where} has type {step.type!r}, which is not in AI_SDK_WORKFLOW_STEPS. "
                f"Registered: {sorted(step_types) or 'none'}."
            )
        history = sorted(name for name in step.history if name not in declared_inputs)
        if history:
            raise ImproperlyConfigured(
                f"{where} sends history {history}, which input_fields does not "
                f"declare. Declared: {sorted(declared_inputs) or 'none'}."
            )
        _compiles(f"Output_{step.name}", step.output_fields, f"{where} output_fields")
        _validate_actions(step.actions, where, action_types)

    _validate_actions(definition.actions, label, action_types)


def _compiles(name: str, fields: dict[str, FieldDefinition], where: str) -> None:
    """Build a declared schema now, so a field name pydantic refuses is caught here.

    A field called `model_dump` passes every name rule above and then takes out the
    worker, so the only honest check is to build the model.
    """
    from django_ai_sdk.workflows.definitions import model_from_fields

    if not fields:
        return
    try:
        model_from_fields(name, fields)
    except Exception as exc:
        raise ImproperlyConfigured(f"{where} cannot compile: {exc}") from exc


def _validate_actions(
    actions: list[ActionDefinition], where: str, action_types: dict[str, type[WorkflowAction]]
) -> None:
    """Every action a definition names must be one the deployment exposed."""
    for action in actions:
        if action.type not in action_types:
            raise ImproperlyConfigured(
                f"{where} names action {action.type!r}, which is not in "
                f"AI_SDK_WORKFLOW_ACTIONS. Registered: {sorted(action_types) or 'none'}."
            )


def get_declared_workflows() -> dict[str, WorkflowDefinition]:
    """Workflows declared in code, keyed by name. Use aget_workflows() to include rows."""
    return dict(_registry)


def get_invalid_workflows() -> dict[str, str]:
    """Declarations the registry refused, name -> reason. Read by the system check."""
    return dict(_invalid)


async def aget_workflows() -> dict[str, WorkflowDefinition]:
    """Code-declared workflows merged with active WorkflowSettings rows."""
    merged = await _db_workflows()
    merged.update(_registry)
    return merged


async def aget_workflow(name: str) -> WorkflowDefinition | None:
    """One workflow by name, from code or the database."""
    declared = _registry.get(name)
    if declared is not None:
        return declared
    return (await _db_workflows()).get(name)


async def _db_workflows() -> dict[str, WorkflowDefinition]:
    """Active WorkflowSettings rows, keyed by slug. A code declaration wins a collision."""
    workflows: dict[str, WorkflowDefinition] = {}
    for slug, raw in await _db_rows():
        if slug in _registry:
            _warn_shadowed(slug)
            continue
        definition = _definition_from_row(slug, raw)
        if definition is not None:
            workflows[slug] = definition
    return workflows


async def _db_rows() -> list[tuple[str, dict[str, object]]]:
    """(slug, definition) for every active row, read fresh so an edit takes effect now.

    A database error propagates: swallowing it would read as "no such workflow".
    """
    from django_ai_sdk.workflows.models import WorkflowSettings

    return [
        (row.slug, row.definition) async for row in WorkflowSettings.objects.filter(active=True)
    ]


def _definition_from_row(slug: str, raw: dict[str, object]) -> WorkflowDefinition | None:
    """Parse and validate one stored definition, or None if it cannot run."""
    from django_ai_sdk.workflows.schemas import WorkflowDefinition

    try:
        definition = WorkflowDefinition.model_validate(raw)
        validate_definition(definition)
    except Exception as exc:
        if slug not in _warned_invalid:
            _warned_invalid.add(slug)
            logger.warning(
                "Workflow %r is stored in the database but cannot run (%s); skipping it.",
                slug,
                exc,
            )
        return None
    return definition


def _warn_shadowed(slug: str) -> None:
    if slug in _warned_shadowed:
        return
    _warned_shadowed.add(slug)
    logger.warning(
        "WorkflowSettings row %r is shadowed by a workflow declared in code, which "
        "wins. Rename the row, or remove the declaration.",
        slug,
    )


def reset_registry() -> None:
    """Clear the registry — for tests."""
    _registry.clear()
    _invalid.clear()
    _warned_shadowed.clear()
    _warned_invalid.clear()


__all__ = [
    "aget_workflow",
    "aget_workflows",
    "get_declared_workflows",
    "get_invalid_workflows",
    "register",
    "reset_registry",
    "validate_definition",
    "validate_name",
]
