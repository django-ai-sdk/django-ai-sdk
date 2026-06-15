from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

FieldType = Literal["str", "int", "float", "bool"]


class StepField(BaseModel):
    type: FieldType = "str"
    description: str = ""


class WorkflowStep(BaseModel):
    name: str = ""
    assistant_id: str
    output_key: str
    input_key: str | None = None
    system_prompt_override: str | None = None
    output_fields: dict[str, StepField] = {}


class WorkflowAction(BaseModel):
    type: str
    input_key: str | None = None


class WorkflowDefinition(BaseModel):
    name: str = ""
    steps: list[WorkflowStep]
    actions: list[WorkflowAction] = []
