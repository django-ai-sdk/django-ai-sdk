"""The JSON shape of a workflow: what an author declares.

These are declarations, not the objects that run. `definitions.compile_steps` turns a
`StepDefinition` into a `Step`, and `definitions.compile_inputs` turns `input_fields`
into the pydantic model a run's inputs are validated against.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, model_validator

# `messages` is the one SDK-owned type: it coerces to `list[ChatMessage]`, which is
# what an agent step sends. Everything else is a plain Python type the host shapes
# itself — a file is a `str` id or a `dict` reference under whatever name it picks.
FieldType = Literal["str", "int", "float", "bool", "list", "dict", "messages"]


class FieldDefinition(BaseModel):
    """One field of a declared input or output schema."""

    type: FieldType = "str"
    description: str = ""
    required: bool = True


class HookDefinition(BaseModel):
    """A hook named by key in `AI_SDK_WORKFLOW_HOOKS`.

    `config` is handed to the hook as it is built, so one registered class serves
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
    # Hooks that fire for this step alone.
    hooks: list[HookDefinition] = []

    @model_validator(mode="after")
    def _check_the_type_and_its_fields_agree(self) -> StepDefinition:
        if not self.name:
            raise ValueError("A step needs a name; its outcome is recorded under it.")
        if self.type == "agent":
            if not self.agent_id:
                raise ValueError("An agent step needs an agent_id.")
        elif self.agent_id or self.output_fields:
            raise ValueError(
                f"Step type {self.type!r} is a registered step, not an agent, so it takes "
                f"no agent_id and no output_fields."
            )
        return self


class WorkflowDefinition(BaseModel):
    name: str = ""
    # What the caller must supply. Validated once, before the first step runs.
    input_fields: dict[str, FieldDefinition] = {}
    steps: list[StepDefinition]
    # Hooks that fire for the run and for every step in it.
    hooks: list[HookDefinition] = []


__all__ = ["FieldDefinition", "FieldType", "HookDefinition", "StepDefinition", "WorkflowDefinition"]
