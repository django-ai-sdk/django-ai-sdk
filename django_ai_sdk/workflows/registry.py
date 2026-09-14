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
    from django_ai_sdk.workflows.actions import BaseAction
    from django_ai_sdk.workflows.schemas import WorkflowAction, WorkflowDefinition, WorkflowStep
    from django_ai_sdk.workflows.steps import Step

logger = logging.getLogger(__name__)

# Matches WorkflowSettings.slug, so a declared and a stored name are one key space.
NAME_MAX_LENGTH = 100

# Injected on every definition run by normalize_workflow_inputs(ensure_messages=True).
RESERVED_INPUT_KEYS = frozenset({"messages"})

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
        validate(definition)
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


def validate(definition: WorkflowDefinition) -> None:
    """Raise ImproperlyConfigured unless the definition is legal to store and compile.

    Covers only what one step cannot see alone: the step and action registries, the
    reserved input names, and the names its neighbours claim. Whether a `requires`
    name is produced needs the run's inputs, so the runner checks that at execute time.
    """
    from django_ai_sdk.workflows.actions import get_action_registry
    from django_ai_sdk.workflows.definitions import get_step_registry

    if not definition.steps:
        raise ImproperlyConfigured(f"Workflow {definition.name!r} has no steps to run.")

    step_types = get_step_registry()
    produced: set[str] = set()
    names: set[str] = set()

    for index, step in enumerate(definition.steps):
        _validate_step(
            step, f"Workflow {definition.name!r} step {index}", step_types, produced, names
        )

    action_types = get_action_registry()
    for action in definition.actions:
        _validate_action(action, definition.name, action_types, produced)


def _validate_step(
    step: WorkflowStep,
    where: str,
    step_types: dict[str, type[Step]],
    produced: set[str],
    names: set[str],
) -> None:
    """Check one step against the registry and the names its neighbours claim.

    Adds what it produces to `produced` and its compiled name to `names`.
    """
    if step.type != "agent" and step.type not in step_types:
        raise ImproperlyConfigured(
            f"{where} has type {step.type!r}, which is not in AI_SDK_WORKFLOW_STEPS. "
            f"Registered: {sorted(step_types) or 'none'}."
        )

    if step.output_key:
        if step.output_key in RESERVED_INPUT_KEYS:
            raise ImproperlyConfigured(
                f"{where} output_key {step.output_key!r} collides with a reserved run input name."
            )
        if step.output_key in produced:
            raise ImproperlyConfigured(
                f"{where} reuses `output_key` {step.output_key!r}, which would overwrite "
                "an earlier step's result."
            )
        produced.add(step.output_key)

    compiled_name = step.name or step.output_key or step.type
    if compiled_name in names:
        raise ImproperlyConfigured(f"{where} compiles to duplicate step name {compiled_name!r}.")
    names.add(compiled_name)

    if not step.error_key:
        return
    if step.on_error != "continue":
        raise ImproperlyConfigured(
            f"{where} sets error_key but on_error is {step.on_error!r}; "
            "error_key is only published when on_error is continue."
        )
    if step.error_key in RESERVED_INPUT_KEYS:
        raise ImproperlyConfigured(
            f"{where} error_key {step.error_key!r} collides with a reserved run input name."
        )
    if step.error_key in produced:
        raise ImproperlyConfigured(
            f"{where} error_key {step.error_key!r} collides with an earlier provides."
        )
    # Counted as produced: on_error is continue, so the runner publishes this name
    # and an action may read it.
    produced.add(step.error_key)


def _validate_action(
    action: WorkflowAction,
    workflow_name: str,
    action_types: dict[str, type[BaseAction]],
    produced: set[str],
) -> None:
    """Check one action against the registry and what the steps produce."""
    if action.type not in action_types:
        raise ImproperlyConfigured(
            f"Workflow {workflow_name!r} action type {action.type!r} is not registered. "
            f"Registered: {sorted(action_types)}."
        )
    if action.input_key and action.input_key not in produced:
        raise ImproperlyConfigured(
            f"Workflow {workflow_name!r} action {action.type!r} reads {action.input_key!r}, "
            f"which no step produces. Available: {sorted(produced)}."
        )


def get_declared_workflows() -> dict[str, WorkflowDefinition]:
    """Workflows declared in code, keyed by name. Use aget_workflows() to include rows."""
    return dict(_registry)


def get_invalid_workflows() -> dict[str, str]:
    """Declarations rejected by validate(), name -> reason. Read by the system check."""
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
        validate(definition)
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
    "RESERVED_INPUT_KEYS",
    "aget_workflow",
    "aget_workflows",
    "get_declared_workflows",
    "get_invalid_workflows",
    "register",
    "reset_registry",
    "validate",
    "validate_name",
]
