"""Compiling a JSON WorkflowDefinition into the objects the runner executes.

A definition composes two kinds of step — an agent named by id, and a `Step` class
the deployment registered under a key in `AI_SDK_WORKFLOW_STEPS` — and declares
its inputs and each step's outputs as fields, which compile to pydantic models.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from django.core.exceptions import ImproperlyConfigured
from django.utils.module_loading import import_string
from pydantic import BaseModel, Field, create_model

from django_ai_sdk.common import ChatMessage
from django_ai_sdk.utils import resolve_setting
from django_ai_sdk.workflows.hooks import WorkflowHook, get_hook_registry
from django_ai_sdk.workflows.steps import AgentStep, OnError, Step

if TYPE_CHECKING:
    from django_ai_sdk.workflows.schemas import (
        FieldDefinition,
        HookDefinition,
        StepDefinition,
        WorkflowDefinition,
    )

logger = logging.getLogger(__name__)

_TYPE_MAP: dict[str, Any] = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list": list,
    "dict": dict,
    "messages": list[ChatMessage],
}


def get_step_registry() -> dict[str, type[Step]]:
    """Step classes a definition may name, by key.

    The registry is the gate: a type that is not listed cannot be composed. A path
    that will not import is left out with a warning.
    """
    registry: dict[str, type[Step]] = {}
    for key, path in resolve_setting("AI_SDK_WORKFLOW_STEPS", {}).items():
        try:
            cls = import_string(path)
        except ImportError:
            logger.warning(
                "Workflow step type %r names %r, which does not import. It is not composable.",
                key,
                path,
            )
            continue
        if not isinstance(cls, type) or not issubclass(cls, Step):
            logger.warning(
                "Workflow step type %r names %r, which is not a Step subclass. "
                "It is not composable.",
                key,
                path,
            )
            continue
        registry[key] = cls
    return registry


def model_from_fields(name: str, fields: dict[str, FieldDefinition]) -> type[BaseModel]:
    """Declared fields as a pydantic model.

    One builder for both directions: a workflow's `input_fields` and a step's
    `output_fields` mean the same thing pointed opposite ways.
    """
    built: dict[str, Any] = {}
    for field_name, field in fields.items():
        if field.type not in _TYPE_MAP:
            logger.warning(
                "Unknown field type %r for %r in %r; using str.", field.type, field_name, name
            )
        annotation = _TYPE_MAP.get(field.type, str)
        if field.required:
            default = Field(description=field.description) if field.description else ...
            built[field_name] = (annotation, default)
        else:
            built[field_name] = (
                annotation | None,
                Field(default=None, description=field.description),
            )
    return cast("type[BaseModel]", create_model(name, **built))


def compile_inputs(workflow: WorkflowDefinition) -> type[BaseModel] | None:
    """The model a run's inputs are validated against, or None if none are declared."""
    if not workflow.input_fields:
        return None
    return model_from_fields(f"Inputs_{workflow.name or 'workflow'}", workflow.input_fields)


def compile_steps(workflow: WorkflowDefinition) -> list[Step]:
    """The definition's steps as Step objects, in the order it declares them."""
    registry = get_step_registry()
    hooks = get_hook_registry()
    history = tuple(
        name for name, field in workflow.input_fields.items() if field.type == "messages"
    )
    return [_compile_step(declared, registry, hooks, history) for declared in workflow.steps]


def _compile_step(
    declared: StepDefinition,
    registry: dict[str, type[Step]],
    hook_registry: dict[str, type[WorkflowHook]],
    history: tuple[str, ...] = (),
) -> Step:
    """One declaration as a fresh Step instance, never a shared registry object."""
    if declared.type == "agent":
        step: Step = _agent_step(declared, history)
    else:
        step_class = registry.get(declared.type)
        if step_class is None:
            raise ImproperlyConfigured(
                f"Workflow step type {declared.type!r} is not registered, so it cannot be "
                f"composed. Add it to AI_SDK_WORKFLOW_STEPS. Registered: {sorted(registry)}."
            )
        step = step_class()
    # The definition is the authority: class attributes on a registered Step are not
    # defaults it falls back to.
    step.name = declared.name
    step.requires = tuple(declared.requires)
    step.on_error = OnError(declared.on_error)
    step.hooks = tuple(compile_hooks(declared.hooks, declared.name, hook_registry))
    return step


def _agent_step(declared: StepDefinition, history: tuple[str, ...] = ()) -> AgentStep:
    """An agent named by id, configured."""
    step = AgentStep()
    step.agent_id = declared.agent_id
    step.history = history
    step.instructions = declared.system_prompt_override or ""
    # Compiled once, so a definition that cannot compile says so before the first
    # model call.
    step.schema = (
        model_from_fields(f"Output_{declared.name}", declared.output_fields)
        if declared.output_fields
        else None
    )
    return step


def compile_hooks(
    declared: list[HookDefinition],
    where: str,
    registry: dict[str, type[WorkflowHook]] | None = None,
) -> list[WorkflowHook]:
    """Hook instances for the keys a definition names."""
    if registry is None:
        registry = get_hook_registry()
    hooks: list[WorkflowHook] = []
    for hook in declared:
        hook_class = registry.get(hook.type)
        if hook_class is None:
            raise ImproperlyConfigured(
                f"Workflow hook {hook.type!r} on {where!r} is not registered. "
                f"Add it to AI_SDK_WORKFLOW_HOOKS. Registered: {sorted(registry)}."
            )
        hooks.append(hook_class(hook.config))
    return hooks


__all__ = [
    "compile_hooks",
    "compile_inputs",
    "compile_steps",
    "get_step_registry",
    "model_from_fields",
]
