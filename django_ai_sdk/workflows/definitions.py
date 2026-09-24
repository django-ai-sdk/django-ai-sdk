from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal, cast

from django.core.exceptions import ImproperlyConfigured
from django.utils.module_loading import import_string
from pydantic import BaseModel, Field, create_model

from django_ai_sdk.utils import resolve_setting
from django_ai_sdk.workflows.actions import WorkflowAction, get_action_registry
from django_ai_sdk.workflows.steps import AgentStep, OnError, Step

if TYPE_CHECKING:
    from django_ai_sdk.workflows.schemas import (
        ActionDefinition,
        FieldDefinition,
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


def field_annotation(field: FieldDefinition, label: str) -> Any:
    """One declared field as a python annotation.

    `label` names the nested models, so a validation error says where it was.
    """
    if field.enum is not None:
        # Subscripting with a tuple is the only way to spell a dynamic Literal.
        return Literal[tuple(field.enum)]  # ty: ignore[invalid-type-form]
    if field.type == "object":
        return model_from_fields(label, field.fields or {})
    if field.type == "list":
        if field.items is None:
            return list
        return list[field_annotation(field.items, label)]  # ty: ignore[invalid-type-form]
    return _TYPE_MAP[field.type]


def model_from_fields(name: str, fields: dict[str, FieldDefinition]) -> type[BaseModel]:
    """Declared fields as a pydantic model.

    One builder for both directions: a workflow's `input_fields` and a step's
    `output_fields` mean the same thing pointed opposite ways.
    """
    built: dict[str, Any] = {}
    for field_name, field in fields.items():
        annotation = field_annotation(field, f"{name}_{field_name}")
        if field.required:
            default = Field(description=field.description) if field.description else ...
            built[field_name] = (annotation, default)
        elif field.default is not None:
            # A real default, so the field keeps its type rather than unioning None.
            built[field_name] = (
                annotation,
                Field(default=field.default, description=field.description),
            )
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


def inputs_json_schema(workflow: WorkflowDefinition) -> dict[str, Any]:
    """The run's inputs as a JSON Schema, for an editor or a tool.

    One artifact, compiled from the same model the run validates against, so a
    form and the executor cannot disagree. A definition that declares nothing
    is open, and the schema says so rather than saying nothing.
    """
    model = compile_inputs(workflow)
    if model is None:
        return {"type": "object", "additionalProperties": True}
    return model.model_json_schema()


def compile_steps(workflow: WorkflowDefinition) -> list[Step]:
    """The definition's steps as Step objects, in the order it declares them."""
    registry = get_step_registry()
    actions = get_action_registry()
    return [_compile_step(declared, registry, actions) for declared in workflow.steps]


def _compile_step(
    declared: StepDefinition,
    registry: dict[str, type[Step]],
    action_registry: dict[str, type[WorkflowAction]],
) -> Step:
    """One declaration as a fresh Step instance, never a shared registry object."""
    if declared.type == "agent":
        step: Step = _agent_step(declared)
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
    step.actions = tuple(compile_actions(declared.actions, declared.name, action_registry))
    return step


def _agent_step(declared: StepDefinition) -> AgentStep:
    """An agent named by id, configured."""
    step = AgentStep()
    step.agent_id = declared.agent_id
    # The inputs this step sends as its conversation — the author names them per
    # step, never the engine inferring a transcript from a value's shape.
    step.history = tuple(declared.history)
    step.instructions = declared.system_prompt_override or ""
    # Compiled once, so a definition that cannot compile says so before the first
    # model call.
    step.schema = (
        model_from_fields(f"Output_{declared.name}", declared.output_fields)
        if declared.output_fields
        else None
    )
    return step


def compile_actions(
    declared: list[ActionDefinition],
    where: str,
    registry: dict[str, type[WorkflowAction]] | None = None,
) -> list[WorkflowAction]:
    """Action instances for the keys a definition names."""
    if registry is None:
        registry = get_action_registry()
    actions: list[WorkflowAction] = []
    for action in declared:
        action_class = registry.get(action.type)
        if action_class is None:
            raise ImproperlyConfigured(
                f"Workflow action {action.type!r} on {where!r} is not registered. "
                f"Add it to AI_SDK_WORKFLOW_ACTIONS. Registered: {sorted(registry)}."
            )
        actions.append(action_class(action.config))
    return actions


__all__ = [
    "compile_actions",
    "compile_inputs",
    "compile_steps",
    "get_step_registry",
    "inputs_json_schema",
    "model_from_fields",
]
