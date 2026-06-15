from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, Field, create_model

from django_ai_sdk.assistants.services import AssistantService
from django_ai_sdk.common import ChatMessage
from django_ai_sdk.logger import get_logger
from django_ai_sdk.workflows.actions import get_action_registry

if TYPE_CHECKING:
    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser

    from django_ai_sdk.workflows.schemas import WorkflowDefinition

_logger = get_logger(__name__)

_TYPE_MAP: dict[str, type] = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
}


class WorkflowExecutor:
    async def run(
        self,
        workflow: WorkflowDefinition,
        messages: list[ChatMessage],
        *,
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> dict[str, Any]:
        outputs: dict[str, Any] = {}

        for step in workflow.steps:
            assistant = await AssistantService.get(step.assistant_id)

            if step.input_key:
                if step.input_key not in outputs:
                    _logger.warning(
                        "Workflow step '{}' input_key '{}' not found in outputs — skipping context injection",
                        step.output_key,
                        step.input_key,
                    )
                    step_messages = list(messages)
                else:
                    prior = outputs[step.input_key]
                    context = ChatMessage(
                        role="user",
                        content=f"[Workflow context] Previous step '{step.input_key}' result:\n{json.dumps(prior, default=str, indent=2)}",
                    )
                    step_messages = [*messages, context]
            else:
                step_messages = list(messages)

            system_prompt = step.system_prompt_override or None

            if step.output_fields:
                field_definitions: dict[str, Any] = {}
                for name, f in step.output_fields.items():
                    if f.type not in _TYPE_MAP:
                        _logger.warning(
                            "Unknown output_field type '{}' for field '{}' in step '{}' — defaulting to str",
                            f.type,
                            name,
                            step.output_key,
                        )
                    field_definitions[name] = (
                        _TYPE_MAP.get(f.type, str),
                        Field(description=f.description) if f.description else ...,
                    )
                DynamicModel = cast(
                    "type[BaseModel]",
                    create_model(f"Output_{step.output_key}", **field_definitions),
                )
                result = await assistant.run(
                    step_messages,
                    system_prompt=system_prompt,
                    response_format=DynamicModel,
                    user=user,
                )
                outputs[step.output_key] = (
                    result.model_dump() if isinstance(result, BaseModel) else {}
                )
            else:
                result = await assistant.run(
                    step_messages,
                    system_prompt=system_prompt,
                    user=user,
                )
                outputs[step.output_key] = result

            _logger.debug("Workflow step '{}' complete", step.output_key)

        action_registry = get_action_registry()
        for action in workflow.actions:
            runner_cls = action_registry.get(action.type)
            if runner_cls is None:
                _logger.warning("Unknown workflow action type: {}", action.type)
                continue
            if action.input_key and action.input_key not in outputs:
                _logger.warning(
                    "Workflow action '{}' input_key '{}' not found in outputs — skipping",
                    action.type,
                    action.input_key,
                )
                continue
            payload = outputs.get(action.input_key) if action.input_key else outputs
            await runner_cls().execute(payload)
            _logger.debug("Workflow action '{}' complete", action.type)

        return outputs
