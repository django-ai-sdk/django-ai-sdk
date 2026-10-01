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
    @pytest.mark.asyncio
    async def test_tool_fn_creates_artifact(self):
        from typing import ClassVar

        from pydantic import BaseModel

        from django_ai_sdk.artifacts import ArtifactSchema, ArtifactType
        from django_ai_sdk.artifacts.models import Artifact
        from django_ai_sdk.conversation.models import Thread

        thread = await Thread.objects.acreate(title="test")

        class Inner(BaseModel):
            summary: str

        class MyArtifact(ArtifactSchema):
            artifact_type: ClassVar[ArtifactType] = ArtifactType.QUESTION_FLOW
            data: Inner

        tool = MyArtifact.as_tool(thread_id=str(thread.id))
        result = await tool.invoke_async(summary="hello world")

        payload = json.loads(result)
        assert "artifact_id" in payload

        artifact = await Artifact.objects.aget(id=payload["artifact_id"])
        assert artifact.schema_name == "MyArtifact"
        assert artifact.artifact_type == "question_flow"
        assert artifact.data == {"summary": "hello world"}
        assert str(artifact.thread_id) == str(thread.id)

    @pytest.mark.asyncio
    async def test_tool_fn_anonymous_user_no_creator(self):
        from typing import ClassVar

        from pydantic import BaseModel

        from django_ai_sdk.artifacts import ArtifactSchema, ArtifactType
        from django_ai_sdk.artifacts.models import Artifact
        from django_ai_sdk.conversation.models import Thread

        thread = await Thread.objects.acreate(title="test")

        class Inner(BaseModel):
            title: str

        class MyArtifact(ArtifactSchema):
            artifact_type: ClassVar[ArtifactType] = ArtifactType.APPROVAL
            data: Inner

        anon = MagicMock()
        anon.is_anonymous = True
        tool = MyArtifact.as_tool(thread_id=str(thread.id), user=anon)
        result = await tool.invoke_async(title="test card")

        payload = json.loads(result)
        artifact = await Artifact.objects.aget(id=payload["artifact_id"])
        assert artifact.created_by_id is None


# ============================================================================
# FileArtifact: thread files resolved server-side
# ============================================================================


@pytest.mark.django_db(transaction=True)
class TestFileArtifact:
    @pytest.fixture
    def thread_files(self, settings, tmp_path, mock_agents_registry):
        from asgiref.sync import async_to_sync
        from django.core.files.base import ContentFile

        from django_ai_sdk.conversation.models import Thread
        from django_ai_sdk.memories.models import EntryDocument
        from django_ai_sdk.memories.services import MemoryService

        settings.MEDIA_ROOT = str(tmp_path)
        thread_id = str(Thread.objects.create(metadata={"agent_id": "test-agent"}).id)
        memory = async_to_sync(MemoryService.get_or_create_thread_file_memory)(thread_id)

        def upload(name, content_type):
            doc = EntryDocument(memory=memory, file_name=name, content_type=content_type)
            doc.file.save(name, ContentFile(b"x"), save=True)
            return str(doc.id)

        return thread_id, upload("cat.png", "image/png"), upload("a.pdf", "application/pdf")

    @pytest.mark.asyncio
    async def test_fills_in_files_from_the_thread(self, thread_files):
        from django_ai_sdk.artifacts import FileArtifact
        from django_ai_sdk.artifacts.models import Artifact

        thread_id, image_id, pdf_id = thread_files
        tool = FileArtifact.as_tool(thread_id=thread_id)
        assert tool.name == "artifact_file_artifact"

        # A made-up url from the model is replaced by the real one.
        result = json.loads(
            await tool.invoke_async(
                files=[{"documentId": image_id.upper(), "url": "http://evil"}, {"documentId": pdf_id}]
            )
        )

        files = result["files"]
        assert [f["documentId"] for f in files] == [image_id, pdf_id]
        assert [f["filename"] for f in files] == ["cat.png", "a.pdf"]
        assert [f["mediaType"] for f in files] == ["image/png", "application/pdf"]
        assert all("evil" not in f["url"] for f in files)
        artifact = await Artifact.objects.aget(id=result["artifact_id"])
        assert artifact.artifact_type == "file"
        assert artifact.data == {"files": files}

    @pytest.mark.asyncio
    async def test_rejects_files_outside_the_thread(self, thread_files):
        from django_ai_sdk.artifacts import FileArtifact
        from django_ai_sdk.artifacts.models import Artifact

        thread_id, image_id, _ = thread_files
        tool = FileArtifact.as_tool(thread_id=thread_id)
        other = str(uuid.uuid4())

        result = json.loads(
            await tool.invoke_async(files=[{"documentId": image_id}, {"documentId": other}, {"documentId": "cat"}])
        )

        assert other in result["error"] and "cat" in result["error"]
        assert not await Artifact.objects.aexists()
