from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

from django.utils import timezone
from pydantic import BaseModel, Field, create_model

from django_ai_sdk.agents.services import AgentService
from django_ai_sdk.common import ChatMessage
from django_ai_sdk.logger import get_logger
from django_ai_sdk.permissions import Operation, check_permissions, get_agent_permissions
from django_ai_sdk.workflows.actions import get_action_registry
from django_ai_sdk.workflows.models import WorkflowRun, WorkflowRunStep
from django_ai_sdk.workflows.tasks import execute_workflow

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


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class WorkflowExecutor:
    @staticmethod
    async def enqueue(run: WorkflowRun) -> None:
        """Schedule a WorkflowRun as a background task."""

        task = await execute_workflow.aenqueue(str(run.id))
        await WorkflowRun.objects.filter(id=run.id).aupdate(task_id=task.id)

    async def run(
        self,
        workflow: WorkflowDefinition,
        messages: list[ChatMessage],
        *,
        user: AbstractBaseUser | AnonymousUser | None = None,
        workflow_run: WorkflowRun | None = None,
    ) -> tuple[dict[str, Any], WorkflowRun]:
        # Guard: idempotent return for already-completed runs
        if workflow_run is not None and workflow_run.status == WorkflowRun.Status.COMPLETED:
            return workflow_run.outputs or {}, workflow_run

        # Transition / create the run record
        if workflow_run is None:
            workflow_run = await WorkflowRun.objects.acreate(
                workflow=None,
                workflow_definition=workflow.model_dump(),
                status=WorkflowRun.Status.RUNNING,
                input_messages=[m.model_dump() for m in messages],
                user_id=user.pk if user and not getattr(user, "is_anonymous", True) else None,
                started_at=timezone.now(),
            )
        else:
            workflow_run.status = WorkflowRun.Status.RUNNING
            if not workflow_run.started_at:
                workflow_run.started_at = timezone.now()
            await workflow_run.asave(update_fields=["status", "started_at", "updated_at"])

        # Replay: load already-completed steps into outputs
        outputs: dict[str, Any] = {}
        completed_seqs: set[int] = set()
        async for s in workflow_run.steps.filter(status=WorkflowRunStep.Status.COMPLETED).order_by(
            "sequence"
        ):
            outputs[s.output_key] = s.output
            completed_seqs.add(s.sequence)

        try:
            for sequence, step in enumerate(workflow.steps):
                if sequence in completed_seqs:
                    _logger.debug("Replaying step {} ({})", sequence, step.output_key)
                    continue

                step_record, _ = await WorkflowRunStep.objects.aupdate_or_create(
                    run=workflow_run,
                    sequence=sequence,
                    defaults={
                        "step_name": step.name,
                        "output_key": step.output_key,
                        "status": WorkflowRunStep.Status.PENDING,
                        "started_at": timezone.now(),
                        "output": None,
                        "error": "",
                    },
                )

                try:
                    agent = await AgentService.get(step.agent_id)

                    await check_permissions(
                        user, Operation.CHAT, get_agent_permissions(agent), agent=agent
                    )

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
                        result = await agent.run(
                            step_messages,
                            system_prompt=system_prompt,
                            response_format=DynamicModel,
                            user=user,
                        )
                        result_value: Any = (
                            result.model_dump() if isinstance(result, BaseModel) else {}
                        )
                    else:
                        result = await agent.run(
                            step_messages,
                            system_prompt=system_prompt,
                            user=user,
                        )
                        result_value = result

                    outputs[step.output_key] = result_value
                    step_record.output = result_value
                    step_record.status = WorkflowRunStep.Status.COMPLETED
                    step_record.completed_at = timezone.now()
                    await step_record.asave(update_fields=["output", "status", "completed_at"])
                    _logger.debug("Workflow step '{}' complete", step.output_key)

                except Exception as step_exc:
                    step_record.status = WorkflowRunStep.Status.FAILED
                    step_record.error = str(step_exc)
                    step_record.completed_at = timezone.now()
                    await step_record.asave(update_fields=["status", "error", "completed_at"])
                    raise

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

            workflow_run.status = WorkflowRun.Status.COMPLETED
            workflow_run.outputs = outputs
            workflow_run.completed_at = timezone.now()
            await workflow_run.asave(
                update_fields=["status", "outputs", "completed_at", "updated_at"]
            )

        except Exception as exc:
            workflow_run.status = WorkflowRun.Status.FAILED
            workflow_run.error = str(exc)
            workflow_run.completed_at = timezone.now()
            await workflow_run.asave(
                update_fields=["status", "error", "completed_at", "updated_at"]
            )
            raise

        return outputs, workflow_run
