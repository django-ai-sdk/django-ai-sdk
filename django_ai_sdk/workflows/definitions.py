from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

from django.core.exceptions import ImproperlyConfigured
from pydantic import BaseModel, Field, create_model

from django_ai_sdk.workflows.conf import registered
from django_ai_sdk.workflows.listeners import RunListener, get_listener_registry
from django_ai_sdk.workflows.schemas import AGENT_STEP, APPROVAL_STEP, PY_TYPES
from django_ai_sdk.workflows.steps import AgentStep, ApprovalStep, OnError, Step

if TYPE_CHECKING:
    from django_ai_sdk.workflows.schemas import (
        FieldDefinition,
        ListenerDefinition,
        StepDefinition,
        WorkflowDefinition,
    )


def get_step_registry() -> dict[str, type[Step]]:
    return {**registered("STEPS", Step), APPROVAL_STEP: ApprovalStep}


def _field_annotation(field: FieldDefinition, label: str) -> Any:
    if field.enum is not None:
        # Subscripting with a tuple is the only way to spell a dynamic Literal.
        return Literal[tuple(field.enum)]  # ty: ignore[invalid-type-form]
    if field.type == "object":
        return model_from_fields(label, field.fields or {})
    if field.type == "list":
        if field.items is None:
            return list
        return list[_field_annotation(field.items, label)]  # ty: ignore[invalid-type-form]
    return PY_TYPES[field.type]


def model_from_fields(name: str, fields: dict[str, FieldDefinition]) -> type[BaseModel]:
    built: dict[str, Any] = {}
    for field_name, field in fields.items():
        annotation = _field_annotation(field, f"{name}_{field_name}")
        if field.required:
            default = Field(description=field.description) if field.description else ...
            built[field_name] = (annotation, default)
        elif field.default is not None:
            built[field_name] = (
                annotation,
                Field(default=field.default, description=field.description),
            )
        else:
            built[field_name] = (
                annotation | None,
                Field(default=None, description=field.description),
            )
    return create_model(name, **built)


def compile_inputs(workflow: WorkflowDefinition) -> type[BaseModel] | None:
    if not workflow.input_fields:
        return None
    return model_from_fields(f"Inputs_{workflow.name or 'workflow'}", workflow.input_fields)


def inputs_json_schema(workflow: WorkflowDefinition) -> dict[str, Any]:
    model = compile_inputs(workflow)
    if model is None:
        return {"type": "object", "additionalProperties": True}
    return model.model_json_schema()


def compile_steps(workflow: WorkflowDefinition) -> list[Step]:
    registry = get_step_registry()
    listeners = get_listener_registry()
    return [_compile_step(declared, registry, listeners) for declared in workflow.steps]


def _compile_step(
    declared: StepDefinition,
    registry: dict[str, type[Step]],
    listener_registry: dict[str, type[RunListener]],
) -> Step:
    if declared.type == AGENT_STEP:
        step: Step = _agent_step(declared)
    else:
        step_class = registry.get(declared.type)
        if step_class is None:
            raise ImproperlyConfigured(
                f"Workflow step type {declared.type!r} is not registered, so it cannot be "
                f"composed. Add it to AI_SDK_WORKFLOWS['STEPS']. Registered: {sorted(registry)}."
            )
        step = step_class()
        step.config = MappingProxyType(dict(declared.config))
    step.name = declared.name
    step.requires = tuple(declared.requires)
    step.on_error = OnError(declared.on_error)
    step.listeners = tuple(compile_listeners(declared.listeners, declared.name, listener_registry))
    return step


def _agent_step(declared: StepDefinition) -> AgentStep:
    step = AgentStep()
    step.agent_id = declared.agent_id
    step.history = tuple(declared.history)
    step.schema = (
        model_from_fields(f"Output_{declared.name}", declared.output_fields)
        if declared.output_fields
        else None
    )
    return step


def compile_listeners(
    declared: list[ListenerDefinition],
    where: str,
    registry: dict[str, type[RunListener]] | None = None,
) -> list[RunListener]:
    if registry is None:
        registry = get_listener_registry()
    listeners: list[RunListener] = []
    for listener in declared:
        listener_class = registry.get(listener.type)
        if listener_class is None:
            raise ImproperlyConfigured(
                f"Workflow listener {listener.type!r} on {where!r} is not registered. "
                f"Add it to AI_SDK_WORKFLOWS['LISTENERS']. Registered: {sorted(registry)}."
            )
        listeners.append(listener_class(listener.config))
    return listeners


__all__ = [
    "compile_inputs",
    "compile_listeners",
    "compile_steps",
    "get_step_registry",
    "inputs_json_schema",
    "model_from_fields",
]
