from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, model_validator

FieldType = Literal["str", "int", "float", "bool"]


class StepField(BaseModel):
    type: FieldType = "str"
    description: str = ""


class WorkflowStep(BaseModel):
    """One step of a definition: an agent call, or a registered Step class."""

    # "agent", or a key of AI_SDK_WORKFLOW_STEPS.
    type: str = "agent"
    name: str = ""
    # The run-state name this step's result is filed under.
    output_key: str = ""
    # Run-state names this step reads: an earlier step's output_key, or a key
    # of the run's inputs.
    requires: list[str] = []
    on_error: Literal["fail", "continue"] = "fail"
    # When on_error is continue and the step fails, publish a failure payload
    # under this run-state key so a handler step can require it.
    error_key: str = ""
    agent_id: str = ""
    system_prompt_override: str | None = None
    output_fields: dict[str, StepField] = {}

    @model_validator(mode="after")
    def _check_the_type_and_its_fields_agree(self) -> WorkflowStep:
        if self.error_key and self.error_key == self.output_key:
            raise ValueError("error_key must differ from output_key.")
        if self.type == "agent":
            if not self.agent_id:
                raise ValueError("An agent step needs an agent_id.")
            if not self.output_key:
                raise ValueError("An agent step needs an output_key to file its result under.")
        elif self.agent_id:
            raise ValueError(
                f"Step type {self.type!r} is a registered step, not an agent, so it takes "
                f"no agent_id."
            )
        return self


class WorkflowAction(BaseModel):
    type: str
    input_key: str | None = None


class WorkflowDefinition(BaseModel):
    name: str = ""
    steps: list[WorkflowStep]
    actions: list[WorkflowAction] = []
