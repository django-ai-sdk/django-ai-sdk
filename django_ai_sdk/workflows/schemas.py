"""The JSON shape of a workflow: what an author declares.

These are declarations, not the objects that run. `definitions.compile_steps` turns a
`StepDefinition` into a `Step`, and `definitions.compile_inputs` turns `input_fields`
into the pydantic model a run's inputs are validated against.

A field is declared structurally — type, description, enum, default, and nested
`items`/`fields` — and the shape is the whole contract: a declared schema has to
survive a dump, cross the queue, and be read back by an editor, so behaviour —
validators, constraints — belongs to a step, not to a field. A workflow written in
code may hand the authoring to a pydantic model instead (`inputs=MyModel`), which
is normalized into the same fields and never rides on the definition.
"""

from __future__ import annotations

from types import UnionType
from typing import TYPE_CHECKING, Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel, model_validator
from pydantic_core import PydanticUndefined

if TYPE_CHECKING:
    from pydantic.fields import FieldInfo

# `object` is a typed record of `fields`; `dict` is the open escape for JSON the
# host shapes itself. A file is a `str` id or a `dict` reference under whatever
# name the host picks — the engine has no opinion about what `pdfs` means, and
# the step that reads it does.
FieldType = Literal["str", "int", "float", "bool", "list", "dict", "object"]

# A declared schema is a shape, not a program: bounds keep one from being a
# pathology the compiler and every editor in front of it has to walk.
MAX_DEPTH = 8
MAX_FIELDS = 32
MAX_ENUM = 64

_SCALARS = ("str", "int", "float", "bool")

_TYPE_OF_ANNOTATION: dict[Any, FieldType] = {
    str: "str",
    int: "int",
    float: "float",
    bool: "bool",
    list: "list",
    dict: "dict",
}


class FieldDefinition(BaseModel):
    """One field of a declared input or output schema."""

    type: FieldType = "str"
    description: str = ""
    required: bool = True
    # The value an absent, not-required field takes. `None` means "no default":
    # the field compiles to `type | None`, as it always has.
    default: Any = None
    # The values a scalar field may take. Compiles to Literal[...].
    enum: list[str | int | float | bool] | None = None
    # The element type of a `list`. `None` leaves the list open.
    items: FieldDefinition | None = None
    # The members of an `object`. An object declares its fields; `dict` is the
    # open JSON object.
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
    """A default has to be a real value of the field's own declared type.

    `create_model` does not validate a `Field(default=...)` on its own, so a
    stored definition can otherwise carry a default of the wrong shape straight
    into a run's inputs silently.
    """
    if field.type in _SCALARS:
        expected = {"str": str, "int": int, "float": float, "bool": bool}[field.type]
        # bool is an int in Python, so an explicit type check keeps a bool default
        # from passing for int/float, and a 0/1 from passing for bool.
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
    """A declared schema cannot be a pathology the compiler has to walk."""
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


class ActionDefinition(BaseModel):
    """An action named by key in `AI_SDK_WORKFLOW_ACTIONS`.

    `config` is handed to the action as it is built, so one registered class serves
    every definition that names it — which recipient, which step's output.
    """

    type: str
    config: dict[str, Any] = {}


class StepDefinition(BaseModel):
    """One step of a definition: an agent call, or a registered Step class."""

    # "agent", or a key of AI_SDK_WORKFLOW_STEPS.
    type: str = "agent"
    # The step's key: its outcome is recorded under it, and later steps require it.
    name: str
    # Names of earlier steps this one reads. A run's inputs are always available,
    # so they are not listed here.
    requires: list[str] = []
    on_error: Literal["fail", "continue"] = "fail"
    agent_id: str = ""
    system_prompt_override: str | None = None
    # Structured output for an agent step. A registered step returns what it returns.
    output_fields: dict[str, FieldDefinition] = {}
    # The inputs this step sends as its conversation, in order. The author names
    # them; the engine never infers a transcript from the shape of a value.
    history: list[str] = []
    # Actions that fire for this step alone.
    actions: list[ActionDefinition] = []

    @model_validator(mode="after")
    def _check_the_type_and_its_fields_agree(self) -> StepDefinition:
        if not self.name:
            raise ValueError("A step needs a name; its outcome is recorded under it.")
        if self.type == "agent":
            if not self.agent_id:
                raise ValueError("An agent step needs an agent_id.")
        elif self.agent_id or self.output_fields or self.history:
            raise ValueError(
                f"Step type {self.type!r} is a registered step, not an agent, so it "
                f"takes no agent_id, no output_fields and no history."
            )
        return self


class WorkflowDefinition(BaseModel):
    # The shape of the stored JSON. Stored rows and run snapshots carry it, so a
    # future format revision can gate on it rather than guess.
    version: Literal[1] = 1
    name: str = ""
    # What the caller must supply. Validated once, before the first step runs.
    input_fields: dict[str, FieldDefinition] = {}
    steps: list[StepDefinition]
    # Actions that fire for the run and for every step in it.
    actions: list[ActionDefinition] = []

    @model_validator(mode="before")
    @classmethod
    def _normalize_a_pydantic_inputs_model(cls, data: Any) -> Any:
        """`inputs=SomeBaseModel` authors `input_fields` with type annotations.

        The model is authoring, not storage: it is normalized into the same
        fields and never rides on the definition, so a stored definition and one
        declared in code have one wire format. What pydantic can express as a
        shape crosses; validators do not, so a model that carries any is
        refused rather than half-honored.
        """
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
    """A pydantic model's fields as FieldDefinitions — the authoring shortcut.

    Validators on the model cannot cross into a schema that has to survive a
    dump, so a model that carries any is refused outright: a check the author
    wrote and the run silently skipped is the footgun.
    """
    decorators = model.__pydantic_decorators__
    if decorators.field_validators or decorators.model_validators:
        raise ValueError(
            f"{model.__name__} carries validators, which do not cross into a declared "
            f"schema. Declare the shape; check cross-field rules in a step."
        )
    return {name: _field_from_info(model, name, info) for name, info in model.model_fields.items()}


def _field_from_info(model: type[BaseModel], name: str, info: FieldInfo) -> FieldDefinition:
    """One model field as a FieldDefinition, or a refusal naming the field."""
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
    return FieldDefinition(
        description=info.description or "",
        # `X | None` means an author's "optional" even though pydantic itself
        # would still require the key with a null in it.
        required=info.is_required() and not optional,
        default=None if info.default is PydanticUndefined else info.default,
        **_type_kwargs(annotation, where),
    )


def _split_optional(annotation: Any) -> tuple[Any, bool]:
    """(the annotation without None, whether it accepted None)."""
    if get_origin(annotation) in (Union, UnionType) and type(None) in get_args(annotation):
        without = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(without) == 1:
            return without[0], True
    return annotation, False


def _type_kwargs(annotation: Any, where: str) -> dict[str, Any]:
    """A python annotation as the parts of a FieldDefinition."""
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


__all__ = [
    "ActionDefinition",
    "FieldDefinition",
    "FieldType",
    "StepDefinition",
    "WorkflowDefinition",
    "fields_from_model",
]
