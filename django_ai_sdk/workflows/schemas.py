from __future__ import annotations

import re
from types import UnionType
from typing import TYPE_CHECKING, Annotated, Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_core import PydanticUndefined

if TYPE_CHECKING:
    from pydantic.fields import FieldInfo

FieldType = Literal["str", "int", "float", "bool", "list", "dict", "object"]

MAX_DEPTH = 8
MAX_FIELDS = 32
MAX_ENUM = 64
MAX_STEPS = 50
MAX_LISTENERS = 20
MAX_DEFINITION_BYTES = 256 * 1024
MAX_TRIGGERS = 10
TRIGGER_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,49}$")

_SCALARS = ("str", "int", "float", "bool")

PY_TYPES: dict[FieldType, type] = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list": list,
    "dict": dict,
}
_TYPE_OF_ANNOTATION: dict[Any, FieldType] = {py: name for name, py in PY_TYPES.items()}

AGENT_STEP = "agent"
APPROVAL_STEP = "approval"


class FieldDefinition(BaseModel):
    """One field of a declared input or output schema."""

    type: FieldType = "str"
    description: str = ""
    required: bool = True
    default: Any = None
    enum: list[str | int | float | bool] | None = None
    items: FieldDefinition | None = None
    fields: dict[str, FieldDefinition] | None = None

    @model_validator(mode="after")
    def _check_the_field_agrees_with_itself(self) -> FieldDefinition:
        if self.enum is not None:
            if self.type not in _SCALARS:
                raise ValueError(f"enum belongs to a scalar field, not type {self.type!r}.")
            if not self.enum:
                raise ValueError("an enum needs at least one value.")
            if self.default is not None and self.default not in self.enum:
                raise ValueError(f"default {self.default!r} is not one of the enum's values.")
        if self.items is not None and self.type != "list":
            raise ValueError(f"items belong to a list, not type {self.type!r}.")
        if self.type == "object" and self.fields is None:
            raise ValueError("an object declares its fields; use dict for open JSON.")
        if self.fields is not None and self.type != "object":
            raise ValueError(f"fields belong to an object, not type {self.type!r}.")
        if self.default is not None and self.required:
            raise ValueError("a field with a default is not required; set required=False.")
        if self.default is not None:
            _check_default_matches_type("this field's default", self.default, self)
        _check_bounds("this field", self)
        return self


def _check_default_matches_type(where: str, value: Any, field: FieldDefinition) -> None:
    """A default must match the field's type: `create_model` does not validate defaults."""
    if field.type in _SCALARS:
        expected = PY_TYPES[field.type]
        # bool is an int subclass: keep bools out of int/float and 0/1 out of bool.
        if field.type == "bool":
            ok = isinstance(value, bool)
        else:
            ok = isinstance(value, expected) and not isinstance(value, bool)
        if not ok:
            raise ValueError(f"{where} is {value!r}, which is not a {field.type}.")
    elif field.type == "list":
        if not isinstance(value, list):
            raise ValueError(f"{where} is {value!r}, which is not a list.")
        if field.items is not None:
            for index, item in enumerate(value):
                _check_default_matches_type(f"{where}[{index}]", item, field.items)
    elif field.type == "dict":
        if not isinstance(value, dict):
            raise ValueError(f"{where} is {value!r}, which is not a dict.")
    elif field.type == "object":
        if not isinstance(value, dict):
            raise ValueError(f"{where} is {value!r}, which is not an object.")
        fields = field.fields or {}
        for key in value:
            if key not in fields:
                raise ValueError(f"{where} has key {key!r}, which is not a declared field.")
        for name, member in fields.items():
            if name in value:
                _check_default_matches_type(f"{where}.{name!r}", value[name], member)


def _check_bounds(where: str, field: FieldDefinition, depth: int = 1) -> None:
    if field.enum is not None and len(field.enum) > MAX_ENUM:
        raise ValueError(f"{where} declares more than {MAX_ENUM} enum values.")
    if depth > MAX_DEPTH:
        raise ValueError(f"{where} nests deeper than {MAX_DEPTH} levels.")
    if field.fields is not None:
        if len(field.fields) > MAX_FIELDS:
            raise ValueError(f"{where} declares more than {MAX_FIELDS} fields.")
        for name, member in field.fields.items():
            _check_bounds(f"field {name!r}", member, depth + 1)
    if field.items is not None:
        _check_bounds(f"{where}'s items", field.items, depth + 1)


class ListenerDefinition(BaseModel):
    """A listener by key in `AI_SDK_WORKFLOWS['LISTENERS']`; `config` goes to its constructor."""

    type: str
    config: dict[str, Any] = {}


class StepDefinition(BaseModel):
    """One step of a definition: an agent call, or a registered Step class."""

    type: str = AGENT_STEP
    name: str
    requires: list[str] = []
    on_error: Literal["stop", "continue"] = "stop"
    agent_id: str = ""
    output_fields: dict[str, FieldDefinition] = {}
    history: list[str] = []
    listeners: list[ListenerDefinition] = []
    config: dict[str, Any] = {}

    @model_validator(mode="before")
    @classmethod
    def _refuse_a_system_prompt(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("system_prompt_override"):
            raise ValueError(
                "system_prompt_override is not accepted: a definition cannot replace an "
                "agent's system prompt. Change the agent, or write the step in code."
            )
        return data

    @model_validator(mode="after")
    def _check_the_type_and_its_fields_agree(self) -> StepDefinition:
        if not self.name:
            raise ValueError("A step needs a name; its outcome is recorded under it.")
        if self.type == AGENT_STEP:
            if not self.agent_id:
                raise ValueError("An agent step needs an agent_id.")
            if self.config:
                raise ValueError("An agent step takes no config; configure the agent.")
        elif self.agent_id or self.output_fields or self.history:
            raise ValueError(
                f"Step type {self.type!r} is a registered step, not an agent, so it "
                f"takes no agent_id, no output_fields and no history."
            )
        return self


class _Trigger(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    run_as: str = ""
    inputs: dict[str, Any] = {}
    enabled: bool = True

    @model_validator(mode="after")
    def _check_the_name(self) -> _Trigger:
        if not TRIGGER_NAME.match(self.name):
            raise ValueError(
                f"Trigger name {self.name!r}: lowercase letters, digits, - and _, at most 50."
            )
        return self


class CronTrigger(_Trigger):
    """Start a run on a schedule: a 5-field cron expression, read in `timezone`."""

    kind: Literal["cron"] = "cron"
    cron: str
    timezone: str = "UTC"


class EventTrigger(_Trigger):
    """Start a run when the host reports `event` via `triggers.start_for_event`."""

    kind: Literal["event"] = "event"
    event: str = Field(min_length=1, max_length=100)


Trigger = Annotated[CronTrigger | EventTrigger, Field(discriminator="kind")]


class WorkflowDefinition(BaseModel):
    version: Literal[1] = 1
    name: str = ""
    input_fields: dict[str, FieldDefinition] = {}
    steps: list[StepDefinition]
    listeners: list[ListenerDefinition] = []
    triggers: list[Trigger] = []

    @model_validator(mode="after")
    def _check_the_size(self) -> WorkflowDefinition:
        if len(self.steps) > MAX_STEPS:
            raise ValueError(f"a workflow has at most {MAX_STEPS} steps, not {len(self.steps)}.")
        if len(self.triggers) > MAX_TRIGGERS:
            raise ValueError(
                f"a workflow has at most {MAX_TRIGGERS} triggers, not {len(self.triggers)}."
            )
        names = [trigger.name for trigger in self.triggers]
        if len(names) != len(set(names)):
            raise ValueError(f"trigger names must be unique within a workflow: {names}.")
        listeners = len(self.listeners) + sum(len(step.listeners) for step in self.steps)
        if listeners > MAX_LISTENERS:
            raise ValueError(f"a workflow has at most {MAX_LISTENERS} listeners, not {listeners}.")
        size = len(self.model_dump_json().encode())
        if size > MAX_DEFINITION_BYTES:
            raise ValueError(
                f"a workflow definition is at most {MAX_DEFINITION_BYTES} bytes as JSON, "
                f"not {size}."
            )
        return self

    @model_validator(mode="before")
    @classmethod
    def _normalize_a_pydantic_inputs_model(cls, data: Any) -> Any:
        if not isinstance(data, dict) or "inputs" not in data:
            return data
        data = {**data}
        model = data.pop("inputs")
        if model is None:
            return data
        if not (isinstance(model, type) and issubclass(model, BaseModel)):
            raise ValueError("inputs= names a pydantic BaseModel subclass, or nothing.")
        if data.get("input_fields"):
            raise ValueError("give inputs= or input_fields=, not both.")
        data["input_fields"] = fields_from_model(model)
        return data


def fields_from_model(model: type[BaseModel]) -> dict[str, FieldDefinition]:
    decorators = model.__pydantic_decorators__
    if decorators.field_validators or decorators.model_validators:
        raise ValueError(
            f"{model.__name__} carries validators, which do not cross into a declared "
            f"schema. Declare the shape; check cross-field rules in a step."
        )
    return {name: _field_from_info(model, name, info) for name, info in model.model_fields.items()}


def _field_from_info(model: type[BaseModel], name: str, info: FieldInfo) -> FieldDefinition:
    where = f"{model.__name__}.{name}"
    if info.metadata:
        raise ValueError(
            f"{where} is Annotated, which a declared schema does not express. Put the "
            f"shape in the type, or check the rule in a step."
        )
    if info.default_factory is not None:
        raise ValueError(
            f"{where} has a default_factory, which does not cross into a declared schema."
        )
    annotation, optional = _split_optional(info.annotation)
    if optional and info.is_required():
        raise ValueError(
            f"{where} is `X | None` without a default, which a declared schema does not "
            f"express. Give it a default, such as `= None`."
        )
    return FieldDefinition(
        description=info.description or "",
        required=info.is_required(),
        default=None if info.default is PydanticUndefined else info.default,
        **_type_kwargs(annotation, where),
    )


def _split_optional(annotation: Any) -> tuple[Any, bool]:
    if get_origin(annotation) in (Union, UnionType) and type(None) in get_args(annotation):
        without = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(without) == 1:
            return without[0], True
    return annotation, False


def _type_kwargs(annotation: Any, where: str) -> dict[str, Any]:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is Literal:
        kinds = {type(value) for value in args}
        if len(kinds) != 1:
            raise ValueError(f"{where} mixes literal types; a declared enum keeps to one.")
        field_type = _TYPE_OF_ANNOTATION.get(kinds.pop())
        if field_type is None or field_type not in _SCALARS:
            raise ValueError(f"{where} is a Literal a declared schema has no type for.")
        return {"type": field_type, "enum": list(args)}
    if origin is list:
        if not args:
            return {"type": "list"}
        if len(args) > 1:
            raise ValueError(
                f"{where} is a list of several types, which a declared schema cannot express."
            )
        return {"type": "list", "items": FieldDefinition(**_type_kwargs(args[0], where))}
    if origin is dict:
        raise ValueError(
            f"{where} is a typed dict, which a declared schema cannot express. Use a "
            f"bare dict for open JSON, or object fields for a typed record."
        )
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return {"type": "object", "fields": fields_from_model(annotation)}
    field_type = _TYPE_OF_ANNOTATION.get(annotation)
    if field_type is None:
        raise ValueError(
            f"{where} is {annotation!r}, which a declared schema cannot express. Use "
            f"the field types it has, or read the value in a step."
        )
    return {"type": field_type}


class WorkflowCreateRequest(BaseModel):
    name: str
    workflow: WorkflowDefinition


class WorkflowUpdateRequest(BaseModel):
    name: str | None = None
    workflow: WorkflowDefinition | None = None
    active: bool | None = None


class WorkflowStartRequest(BaseModel):
    inputs: dict[str, Any] = {}


__all__ = [
    "MAX_STEPS",
    "FieldDefinition",
    "FieldType",
    "ListenerDefinition",
    "StepDefinition",
    "WorkflowCreateRequest",
    "WorkflowDefinition",
    "WorkflowStartRequest",
    "WorkflowUpdateRequest",
]
