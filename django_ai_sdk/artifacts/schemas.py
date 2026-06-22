from __future__ import annotations

import json
import re
from enum import StrEnum
from typing import TYPE_CHECKING, Any, ClassVar, Literal
from uuid import uuid4

from haystack.tools import Tool
from pydantic import BaseModel, Field, ValidationError

from django_ai_sdk.common import prompt

if TYPE_CHECKING:
    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser


def _to_snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


class ArtifactType(StrEnum):
    DATA_TABLE = "data_table"
    OPTION_LIST = "option_list"
    QUESTION_FLOW = "question_flow"
    APPROVAL = "approval"
    PLAN = "plan"
    PROGRESS_TRACKER = "progress_tracker"
    TERMINAL = "terminal"


class ArtifactSchema(BaseModel):
    """Base class for structured artifacts."""

    artifact_type: ClassVar[ArtifactType]
    system_prompt_hint: ClassVar[str] = ""
    data: Any

    @classmethod
    def as_tool(
        cls,
        thread_id: str,
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> Tool:

        data_annotation = cls.model_fields["data"].annotation
        schema = data_annotation.model_json_schema()
        artifact_type = str(cls.artifact_type)
        resolved_user = user if (user and not getattr(user, "is_anonymous", True)) else None

        def submit(**kwargs: Any) -> str:
            from django_ai_sdk.artifacts.models import Artifact  # noqa: PLC0415

            try:
                data_annotation.model_validate(kwargs)
            except ValidationError as e:
                return json.dumps({"error": str(e)})

            artifact = Artifact.objects.create(
                thread_id=thread_id,
                schema_name=cls.__name__,
                artifact_type=artifact_type,
                data=kwargs,
                created_by=resolved_user,
            )
            return json.dumps({"artifact_id": str(artifact.id), **kwargs})

        hint = f" {cls.system_prompt_hint}" if cls.system_prompt_hint else ""
        return Tool(
            name=f"artifact_{_to_snake(cls.__name__)}",
            description=(
                "# Information"
                f"{hint}"
                f"Submit a structured {cls.__name__} when you have gathered "
                "enough information to fill all fields. "
                "If the tool returns an error, fix the data and retry. "
                "Once it returns artifact_id, respond with exactly one sentence that briefly "
                "describes what the data shows, do NOT repeat the data as text or markdown."
            ),
            parameters=schema,
            function=submit,
        )


# ── QuestionFlow ─────────────────────────────────────────────────────────────


class QuestionFlowOption(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    label: str
    description: str | None = None
    disabled: bool | None = None


class QuestionFlowStep(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    description: str | None = None
    options: list[QuestionFlowOption]
    selectionMode: Literal["single", "multi"] | None = None


class QuestionFlowData(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    steps: list[QuestionFlowStep]


class QuestionFlowArtifact(ArtifactSchema):
    artifact_type: ClassVar[ArtifactType] = ArtifactType.QUESTION_FLOW
    system_prompt_hint: ClassVar[str] = prompt("""
        Use artifact_question_flow_artifact() when you need to guide the user through
        a multi-step questionnaire or decision tree.
        Provide a unique `id` slug (e.g. 'onboarding-flow').
        Each step needs its own `id`, `title`, optional `description`,
        a list of options (each with unique `id` and `label`, optional `description`/`disabled`),
        and optional `selectionMode` ('single' or 'multi', default single).
        Call it once you have all steps ready.
    """)
    data: QuestionFlowData


# ── OptionList ───────────────────────────────────────────────────────────────


class OptionListOption(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    label: str
    description: str | None = None
    disabled: bool | None = None


class OptionListData(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    options: list[OptionListOption]
    selectionMode: Literal["multi", "single"] | None = None
    minSelections: int | None = None
    maxSelections: int | None = None


class OptionListArtifact(ArtifactSchema):
    artifact_type: ClassVar[ArtifactType] = ArtifactType.OPTION_LIST
    system_prompt_hint: ClassVar[str] = prompt("""
        Use artifact_option_list_artifact() to present a list of choices to the user.
        Provide a unique `id` slug (e.g. 'deploy-target').
        Each option needs a unique `id`, a `label`, optional `description` and `disabled`.
        Set `selectionMode` to 'single' or 'multi' (default multi).
        Use `minSelections`/`maxSelections` to constrain picks.
        Call it when the user needs to pick from a set of alternatives.
    """)
    data: OptionListData


# ── DataTable ────────────────────────────────────────────────────────────────


class DataTableColumn(BaseModel):
    key: str
    label: str


class DataTableData(BaseModel):
    columns: list[DataTableColumn]
    rows: list[dict[str, str]]


class DataTableArtifact(ArtifactSchema):
    artifact_type: ClassVar[ArtifactType] = ArtifactType.DATA_TABLE
    system_prompt_hint: ClassVar[str] = prompt("""
        Use artifact_data_table_artifact() to present structured tabular data.
        Define columns with key and label, and rows as dicts keyed by column key.
        All row values must be strings.
        Call it when the user requests a table, comparison, or ranked list.
    """)
    data: DataTableData


# ── ApprovalCard ─────────────────────────────────────────────────────────────


class ApprovalCardMetadataItem(BaseModel):
    key: str
    value: str


class ApprovalCardData(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    description: str | None = None
    icon: str | None = None
    metadata: list[ApprovalCardMetadataItem] | None = None
    variant: Literal["default", "destructive"] | None = None
    confirmLabel: str | None = None
    cancelLabel: str | None = None


class ApprovalCardArtifact(ArtifactSchema):
    artifact_type: ClassVar[ArtifactType] = ArtifactType.APPROVAL
    system_prompt_hint: ClassVar[str] = prompt("""
        Use artifact_approval_card_artifact() when an action requires explicit user confirmation.
        Provide a unique `id` slug (e.g. 'delete-account'), a `title`,
        optional `description`, optional `icon` (lucide icon name in kebab-case e.g. 'trash-2'),
        optional `metadata` items as `key`/`value` pairs for context,
        optional `variant` ('destructive' for dangerous actions, else 'default'),
        and optional `confirmLabel`/`cancelLabel` to override button text.
    """)
    data: ApprovalCardData


# ── Plan ─────────────────────────────────────────────────────────────────────


class PlanTodo(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    label: str
    status: Literal["pending", "in_progress", "completed", "cancelled"]
    description: str | None = None


class PlanData(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str | None = None
    description: str | None = None
    todos: list[PlanTodo]


class PlanArtifact(ArtifactSchema):
    artifact_type: ClassVar[ArtifactType] = ArtifactType.PLAN
    system_prompt_hint: ClassVar[str] = prompt("""
        Use artifact_plan_artifact() to show a structured task plan with progress tracking.
        Provide a unique `id` slug (e.g. 'migration-plan'), optional `title` and `description`.
        Each todo needs a unique `id`, a `label`, and a `status`:
        'pending' — not started,
        'in_progress' — currently active (use for at most one item at a time),
        'completed' — done,
        'cancelled' — skipped/abandoned.
        Add optional `description` to a todo for extra context shown on expand.
        Call it to present a plan before execution, or to show final state after.
    """)
    data: PlanData


# ── ProgressTracker ───────────────────────────────────────────────────────────


class ProgressStep(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    label: str
    status: Literal["pending", "in-progress", "completed", "failed"]
    description: str | None = None


class ProgressTrackerChoice(BaseModel):
    outcome: Literal["success", "partial", "failed", "cancelled"]
    summary: str


class ProgressTrackerData(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    steps: list[ProgressStep]
    elapsedTime: int | None = None
    choice: ProgressTrackerChoice | None = None


class ProgressTrackerArtifact(ArtifactSchema):
    artifact_type: ClassVar[ArtifactType] = ArtifactType.PROGRESS_TRACKER
    system_prompt_hint: ClassVar[str] = prompt("""
        Use artifact_progress_tracker_artifact() to show a live or completed multi-step process.
        Provide a unique `id` slug (e.g. 'deploy-pipeline').
        Each step needs a unique `id`, a `label`, a `status`:
        'pending' — not yet started,
        'in-progress' — currently running (at most one at a time),
        'completed' — succeeded,
        'failed' — errored.
        Add optional `description` to a step for detail shown while active or failed.
        Set `elapsedTime` in milliseconds if duration is known.
        Once the process finishes, set `choice` with `outcome`
        ('success', 'partial', 'failed', or 'cancelled') and a short `summary` string.
        Call it once after the process completes to show the full receipt.
    """)
    data: ProgressTrackerData


# ── Terminal ──────────────────────────────────────────────────────────────────


class TerminalData(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    command: str
    stdout: str | None = None
    stderr: str | None = None
    exitCode: int | None = None
    durationMs: int | None = None
    cwd: str | None = None
    truncated: bool | None = None
    maxCollapsedLines: int | None = None


class TerminalArtifact(ArtifactSchema):
    artifact_type: ClassVar[ArtifactType] = ArtifactType.TERMINAL
    system_prompt_hint: ClassVar[str] = prompt("""
        Use artifact_terminal_artifact() to display command output to the user.
        Provide a unique `id` slug (e.g. 'build-output'), the `command` string that was run,
        optional `stdout` and `stderr` output as strings,
        optional `exitCode` (0 = success), optional `durationMs` for elapsed time,
        optional `cwd` for the working directory shown in the header,
        optional `truncated` true if output was cut, and optional `maxCollapsedLines`
        to collapse long output (default shows all).
        Call it after executing a shell command to show its result.
    """)
    data: TerminalData
