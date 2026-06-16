from __future__ import annotations

import json
import re
from enum import StrEnum
from typing import TYPE_CHECKING, Any, ClassVar

from haystack.tools import Tool
from pydantic import BaseModel, ValidationError

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
                f"enough information to fill all fields. "
                "If the tool returns an error, fix the data and retry. "
                "Once it returns artifact_id, respond with exactly one sentence that briefly "
                "describes what the data shows, do NOT repeat the data as text or markdown."
            ),
            parameters=schema,
            function=submit,
        )


# ── QuestionFlow ────────────────────────────────────────────────────────────────


class QuestionFlowOption(BaseModel):
    label: str
    value: str


class QuestionFlowStep(BaseModel):
    question: str
    options: list[QuestionFlowOption] = []


class QuestionFlowData(BaseModel):
    title: str
    steps: list[QuestionFlowStep]


class QuestionFlowArtifact(ArtifactSchema):
    artifact_type: ClassVar[ArtifactType] = ArtifactType.QUESTION_FLOW
    system_prompt_hint: ClassVar[str] = (
        "Use artifact_question_flow_artifact() when you need to guide the user through "
        "a multi-step questionnaire or decision tree or interviews. "
        "Each step has a question and a list of options. "
        "Call it once you have all steps ready."
    )
    data: QuestionFlowData


# ── OptionList ────────────────────────────────────────────────────────────────


class OptionListOption(BaseModel):
    label: str
    description: str = ""


class OptionListData(BaseModel):
    title: str
    options: list[OptionListOption]


class OptionListArtifact(ArtifactSchema):
    artifact_type: ClassVar[ArtifactType] = ArtifactType.OPTION_LIST
    system_prompt_hint: ClassVar[str] = (
        "Use artifact_option_list_artifact() to present a titled list of choices to the user. "
        "Each option has a label and an optional description. Call it when the user needs "
        "to pick from a set of alternatives."
    )
    data: OptionListData


# ── DataTable ─────────────────────────────────────────────────────────────────


class DataTableColumn(BaseModel):
    key: str
    label: str


class DataTableData(BaseModel):
    columns: list[DataTableColumn]
    rows: list[dict[str, str]]


class DataTableArtifact(ArtifactSchema):
    artifact_type: ClassVar[ArtifactType] = ArtifactType.DATA_TABLE
    system_prompt_hint: ClassVar[str] = (
        "Use artifact_data_table_artifact() to present structured tabular data. "
        "Define columns with key and label, and rows as dicts keyed by column key. "
        "All row values must be strings. Call it when the user requests a table, "
        "comparison, or ranked list."
    )
    data: DataTableData


# ── ApprovalCard ──────────────────────────────────────────────────────────────


class ApprovalCardMetadataItem(BaseModel):
    label: str
    value: str


class ApprovalCardData(BaseModel):
    title: str
    description: str
    metadata: list[ApprovalCardMetadataItem] = []


class ApprovalCardArtifact(ArtifactSchema):
    artifact_type: ClassVar[ArtifactType] = ArtifactType.APPROVAL
    system_prompt_hint: ClassVar[str] = (
        "Use artifact_approval_card_artifact() when an action requires explicit user "
        "confirmation. Provide a title, a description of what will happen, and optional "
        "metadata items as label/value pairs for context."
    )
    data: ApprovalCardData
