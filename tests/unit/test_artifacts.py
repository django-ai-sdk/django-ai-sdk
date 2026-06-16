"""
Tests for the artifact layer:
  - ArtifactSchema.as_tool() produces valid Haystack Tool
  - as_tool() closure stores Artifact in DB via ORM
  - ArtifactType enum values
"""

from __future__ import annotations

import json
import uuid
from unittest.mock import MagicMock

import pytest


# ============================================================================
# ArtifactType enum
# ============================================================================


class TestArtifactType:
    def test_values(self):
        from django_ai_sdk.artifacts import ArtifactType

        assert ArtifactType.DATA_TABLE == "data_table"
        assert ArtifactType.OPTION_LIST == "option_list"
        assert ArtifactType.QUESTION_FLOW == "question_flow"
        assert ArtifactType.APPROVAL == "approval"


# ============================================================================
# ArtifactSchema
# ============================================================================


class TestArtifactSchema:
    def test_classvar_not_in_json_schema(self):
        from typing import ClassVar

        from pydantic import BaseModel

        from django_ai_sdk.artifacts import ArtifactSchema, ArtifactType

        class Inner(BaseModel):
            value: str

        class MyArtifact(ArtifactSchema):
            artifact_type: ClassVar[ArtifactType] = ArtifactType.APPROVAL
            data: Inner

        schema = MyArtifact.model_json_schema()
        assert "artifact_type" not in schema.get("properties", {})
        assert "data" in schema.get("properties", {})

    def test_as_tool_returns_haystack_tool(self):
        from typing import ClassVar

        from pydantic import BaseModel

        from django_ai_sdk.artifacts import ArtifactSchema, ArtifactType

        class Inner(BaseModel):
            summary: str

        class MyArtifact(ArtifactSchema):
            artifact_type: ClassVar[ArtifactType] = ArtifactType.APPROVAL
            data: Inner

        tool = MyArtifact.as_tool(thread_id=str(uuid.uuid4()))

        from haystack.tools import Tool

        assert isinstance(tool, Tool)
        assert tool.name == "artifact_my_artifact"
        assert "MyArtifact" in tool.description

    def test_as_tool_exposes_inner_data_schema(self):
        from typing import ClassVar

        from pydantic import BaseModel

        from django_ai_sdk.artifacts import ArtifactSchema, ArtifactType

        class Inner(BaseModel):
            title: str
            count: int

        class MyArtifact(ArtifactSchema):
            artifact_type: ClassVar[ArtifactType] = ArtifactType.DATA_TABLE
            data: Inner

        tool = MyArtifact.as_tool(thread_id="thread-1")
        params = tool.parameters
        assert "title" in params.get("properties", {})
        assert "count" in params.get("properties", {})
        # artifact_type must NOT appear
        assert "artifact_type" not in params.get("properties", {})


# ============================================================================
# as_tool() DB write (DB tests)
# ============================================================================


@pytest.mark.django_db
class TestArtifactToolDbWrite:
    def test_tool_fn_creates_artifact(self):
        from typing import ClassVar

        from pydantic import BaseModel

        from django_ai_sdk.artifacts import ArtifactSchema, ArtifactType
        from django_ai_sdk.artifacts.models import Artifact
        from django_ai_sdk.conversation.models import Thread

        thread = Thread.objects.create(title="test")

        class Inner(BaseModel):
            summary: str

        class MyArtifact(ArtifactSchema):
            artifact_type: ClassVar[ArtifactType] = ArtifactType.QUESTION_FLOW
            data: Inner

        tool = MyArtifact.as_tool(thread_id=str(thread.id))
        result = tool.function(summary="hello world")

        payload = json.loads(result)
        assert "artifact_id" in payload

        artifact = Artifact.objects.get(id=payload["artifact_id"])
        assert artifact.schema_name == "MyArtifact"
        assert artifact.artifact_type == "question_flow"
        assert artifact.data == {"summary": "hello world"}
        assert str(artifact.thread_id) == str(thread.id)

    def test_tool_fn_anonymous_user_no_creator(self):
        from typing import ClassVar

        from pydantic import BaseModel

        from django_ai_sdk.artifacts import ArtifactSchema, ArtifactType
        from django_ai_sdk.artifacts.models import Artifact
        from django_ai_sdk.conversation.models import Thread

        thread = Thread.objects.create(title="test")

        class Inner(BaseModel):
            title: str

        class MyArtifact(ArtifactSchema):
            artifact_type: ClassVar[ArtifactType] = ArtifactType.APPROVAL
            data: Inner

        anon = MagicMock()
        anon.is_anonymous = True
        tool = MyArtifact.as_tool(thread_id=str(thread.id), user=anon)
        result = tool.function(title="test card")

        payload = json.loads(result)
        artifact = Artifact.objects.get(id=payload["artifact_id"])
        assert artifact.created_by_id is None
