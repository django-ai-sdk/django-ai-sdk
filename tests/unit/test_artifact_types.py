"""Tests for built-in ArtifactSchema subclasses matching tool-ui component schemas."""

from __future__ import annotations


class TestQuestionFlowArtifact:
    def test_artifact_type(self):
        from django_ai_sdk.artifacts import QuestionFlowArtifact
        from django_ai_sdk.artifacts import ArtifactType

        assert QuestionFlowArtifact.artifact_type == ArtifactType.QUESTION_FLOW
        assert str(QuestionFlowArtifact.artifact_type) == "question_flow"

    def test_json_schema_has_data_no_artifact_type(self):
        from django_ai_sdk.artifacts import QuestionFlowArtifact

        schema = QuestionFlowArtifact.model_json_schema()
        assert "artifact_type" not in schema.get("properties", {})
        assert "data" in schema.get("properties", {})

    def test_roundtrip(self):
        from django_ai_sdk.artifacts import (
            QuestionFlowArtifact,
            QuestionFlowData,
            QuestionFlowOption,
            QuestionFlowStep,
        )

        artifact = QuestionFlowArtifact(
            data=QuestionFlowData(
                title="Quick survey",
                steps=[
                    QuestionFlowStep(
                        question="What is your goal?",
                        options=[
                            QuestionFlowOption(label="Learn", value="learn"),
                            QuestionFlowOption(label="Build", value="build"),
                        ],
                    )
                ],
            )
        )
        dumped = artifact.model_dump()
        assert dumped["data"]["title"] == "Quick survey"
        step = dumped["data"]["steps"][0]
        assert step["question"] == "What is your goal?"
        assert len(step["options"]) == 2
        assert step["options"][0] == {"label": "Learn", "value": "learn"}

    def test_options_default_empty(self):
        from django_ai_sdk.artifacts import QuestionFlowData, QuestionFlowStep

        step = QuestionFlowStep(question="Free-form?")
        assert step.options == []
        data = QuestionFlowData(title="t", steps=[step])
        assert data.steps[0].options == []


class TestOptionListArtifact:
    def test_artifact_type(self):
        from django_ai_sdk.artifacts import OptionListArtifact
        from django_ai_sdk.artifacts import ArtifactType

        assert OptionListArtifact.artifact_type == ArtifactType.OPTION_LIST

    def test_json_schema_no_artifact_type_field(self):
        from django_ai_sdk.artifacts import OptionListArtifact

        schema = OptionListArtifact.model_json_schema()
        assert "artifact_type" not in schema.get("properties", {})

    def test_roundtrip(self):
        from django_ai_sdk.artifacts import OptionListArtifact, OptionListData, OptionListOption

        artifact = OptionListArtifact(
            data=OptionListData(
                title="Pick a plan",
                options=[
                    OptionListOption(label="Free", description="0 $/mo"),
                    OptionListOption(label="Pro"),
                ],
            )
        )
        dumped = artifact.model_dump()
        assert dumped["data"]["title"] == "Pick a plan"
        assert len(dumped["data"]["options"]) == 2
        assert dumped["data"]["options"][1]["description"] == ""

    def test_option_description_defaults_empty(self):
        from django_ai_sdk.artifacts import OptionListOption

        opt = OptionListOption(label="X")
        assert opt.description == ""


class TestDataTableArtifact:
    def test_artifact_type(self):
        from django_ai_sdk.artifacts import DataTableArtifact
        from django_ai_sdk.artifacts import ArtifactType

        assert DataTableArtifact.artifact_type == ArtifactType.DATA_TABLE

    def test_roundtrip(self):
        from django_ai_sdk.artifacts import DataTableArtifact, DataTableColumn, DataTableData

        artifact = DataTableArtifact(
            data=DataTableData(
                columns=[
                    DataTableColumn(key="name", label="Name"),
                    DataTableColumn(key="age", label="Age"),
                ],
                rows=[{"name": "Alice", "age": "30"}, {"name": "Bob", "age": "25"}],
            )
        )
        dumped = artifact.model_dump()
        assert len(dumped["data"]["columns"]) == 2
        assert len(dumped["data"]["rows"]) == 2
        assert dumped["data"]["columns"][0] == {"key": "name", "label": "Name"}
        assert dumped["data"]["rows"][0]["name"] == "Alice"

    def test_json_schema_no_artifact_type_field(self):
        from django_ai_sdk.artifacts import DataTableArtifact

        schema = DataTableArtifact.model_json_schema()
        assert "artifact_type" not in schema.get("properties", {})


class TestApprovalCardArtifact:
    def test_artifact_type(self):
        from django_ai_sdk.artifacts import ApprovalCardArtifact
        from django_ai_sdk.artifacts import ArtifactType

        assert ApprovalCardArtifact.artifact_type == ArtifactType.APPROVAL

    def test_roundtrip(self):
        from django_ai_sdk.artifacts import ApprovalCardArtifact, ApprovalCardData

        artifact = ApprovalCardArtifact(
            data=ApprovalCardData(title="Deploy?", description="Deploy to production")
        )
        dumped = artifact.model_dump()
        assert dumped["data"]["title"] == "Deploy?"
        assert dumped["data"]["description"] == "Deploy to production"
        assert dumped["data"]["metadata"] == []

    def test_metadata_items(self):
        from django_ai_sdk.artifacts import (
            ApprovalCardArtifact,
            ApprovalCardData,
            ApprovalCardMetadataItem,
        )

        artifact = ApprovalCardArtifact(
            data=ApprovalCardData(
                title="Confirm?",
                description="Action needs approval",
                metadata=[ApprovalCardMetadataItem(label="Environment", value="production")],
            )
        )
        dumped = artifact.model_dump()
        assert dumped["data"]["metadata"][0] == {"label": "Environment", "value": "production"}

    def test_json_schema_no_artifact_type_field(self):
        from django_ai_sdk.artifacts import ApprovalCardArtifact

        schema = ApprovalCardArtifact.model_json_schema()
        assert "artifact_type" not in schema.get("properties", {})


class TestPublicAPIExports:
    def test_artifact_classes_importable_from_artifacts(self):
        from django_ai_sdk.artifacts import (
            ApprovalCardArtifact,
            ApprovalCardData,
            ApprovalCardMetadataItem,
            DataTableArtifact,
            DataTableColumn,
            DataTableData,
            OptionListArtifact,
            OptionListData,
            OptionListOption,
            QuestionFlowArtifact,
            QuestionFlowData,
            QuestionFlowOption,
            QuestionFlowStep,
        )

        assert QuestionFlowArtifact is not None
        assert OptionListArtifact is not None
        assert DataTableArtifact is not None
        assert ApprovalCardArtifact is not None

    def test_top_level_exports_artifact_schema_and_type(self):
        from django_ai_sdk import ArtifactSchema, ArtifactType, DataTableArtifact, QuestionFlowArtifact

        assert ArtifactSchema is not None
        assert ArtifactType is not None
        assert DataTableArtifact is not None
        assert QuestionFlowArtifact is not None
