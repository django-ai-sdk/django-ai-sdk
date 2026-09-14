"""Compiling a JSON WorkflowDefinition into the Steps the runner executes.

A definition composes two kinds of step: an agent named by id, and a `Step` class
the deployment registered under a key in `AI_SDK_WORKFLOW_STEPS`.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, cast

from django.core.exceptions import ImproperlyConfigured
from django.utils.module_loading import import_string
from pydantic import BaseModel, Field, create_model

from django_ai_sdk.agents.services import AgentService
from django_ai_sdk.common import ChatMessage
from django_ai_sdk.utils import resolve_setting
from django_ai_sdk.workflows.steps import AgentStep, OnError, Step, StepContext, StepOutcome

if TYPE_CHECKING:
    from django_ai_sdk.workflows.schemas import WorkflowDefinition, WorkflowStep

logger = logging.getLogger(__name__)

_TYPE_MAP: dict[str, type] = {"str": str, "int": int, "float": float, "bool": bool}


def get_step_registry() -> dict[str, type[Step]]:
    """Step classes a definition may name, by key.

    The registry gates what a runtime author can reach: a type that is not listed
    cannot be composed. A path that will not import is left out with a warning.
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


def _output_model(spec: WorkflowStep) -> type[BaseModel]:
    """The step's declared output fields as a pydantic model."""
    fields: dict[str, Any] = {}
    for name, field in spec.output_fields.items():
        if field.type not in _TYPE_MAP:
            logger.warning(
                "Unknown output_field type %r for %r in step %r; using str.",
                field.type,
                name,
                spec.output_key,
            )
        fields[name] = (
            _TYPE_MAP.get(field.type, str),
            Field(description=field.description) if field.description else ...,
        )
    return cast("type[BaseModel]", create_model(f"Output_{spec.output_key}", **fields))


class _DefinitionAgentStep(AgentStep):
    """An agent named by id, run as one step of a definition."""

    def __init__(self, spec: WorkflowStep) -> None:
        self.spec = spec
        self.name = spec.name or spec.output_key
        self.provides = spec.output_key
        self.requires = tuple(spec.requires)
        self.on_error = OnError(spec.on_error)
        self.error_key = spec.error_key or ""
        # Compiled once, so a definition that cannot compile says so before the
        # first model call.
        self.schema = _output_model(spec) if spec.output_fields else None

    async def system_prompt(self, ctx: StepContext) -> str:
        """The override the definition declares, if any."""
        return self.spec.system_prompt_override or ""

    async def messages(self, ctx: StepContext) -> list[Any]:
        """The run's transcript, plus what earlier steps produced."""
        # Inputs are JSON-safe dicts on the run row, so coerce before .role access.
        raw = ctx.inputs.get("messages") or []
        history = [
            item if isinstance(item, ChatMessage) else ChatMessage.model_validate(item)
            for item in raw
        ]
        return [*history, *self._context(ctx)]

    async def run(self, ctx: StepContext) -> StepOutcome:
        """Resolve the agent by id and call it."""
        agent = await AgentService.get(self.spec.agent_id)
        return await self.run_agent(
            ctx,
            agent,
            await self.messages(ctx),
            await self.system_prompt(ctx) or None,
        )

    def outcome_for(self, result: Any) -> StepOutcome:
        """Dump the model's answer to JSON, since it goes on the run's record."""
        if self.schema and not isinstance(result, BaseModel):
            # The step declared fields and did not get them, so `on_error` decides
            # what that costs.
            return StepOutcome(status="failed", detail="the agent returned no structured output")
        if isinstance(result, BaseModel):
            return StepOutcome(output=result.model_dump(mode="json"))
        return StepOutcome(output=result)

    def _context(self, ctx: StepContext) -> list[ChatMessage]:
        """What earlier steps produced, as messages this one can read."""
        return [
            ChatMessage(
                role="user",
                content=f"[{name}]\n{json.dumps(ctx.get(name), default=str, indent=2)}",
            )
            for name in self.requires
            if ctx.get(name) is not None
        ]


def _bind_registered_step(step_class: type[Step], spec: WorkflowStep) -> Step:
    """A fresh instance wired by the definition, never a shared registry object.

    The spec owns the graph fields; class attributes on the registered Step are
    not defaults.
    """
    step = step_class()
    step.name = spec.name or spec.output_key or spec.type
    step.requires = tuple(spec.requires)
    step.provides = spec.output_key
    step.on_error = OnError(spec.on_error)
    step.error_key = spec.error_key or ""
    return step


def compile_steps(workflow: WorkflowDefinition) -> list[Step]:
    """The definition's steps as Step objects, in the order it declares them."""
    registry = get_step_registry()
    steps: list[Step] = []
    for spec in workflow.steps:
        if spec.type == "agent":
            steps.append(_DefinitionAgentStep(spec))
            continue

        step_class = registry.get(spec.type)
        if step_class is None:
            raise ImproperlyConfigured(
                f"Workflow step type {spec.type!r} is not registered, so it cannot be "
                f"composed. Add it to AI_SDK_WORKFLOW_STEPS. Registered: {sorted(registry)}."
            )
        steps.append(_bind_registered_step(step_class, spec))
    return steps


__all__ = ["compile_steps", "get_step_registry"]
