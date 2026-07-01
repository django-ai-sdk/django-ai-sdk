from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from django.core.files.base import ContentFile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _make_entry_doc(memory):
    from django_ai_sdk.memories.models import EntryDocument

    entry_doc = EntryDocument(
        entry=None,
        file_name="test.txt",
        file_size=12,
        content_type="text/plain",
        file_extension="txt",
        processing_status=EntryDocument.ProcessingStatus.PENDING,
    )
    # file.save() is sync; Django wraps it fine inside acreate's sync path
    entry_doc.file.save("test.txt", ContentFile(b"hello world!"), save=False)
    await entry_doc.asave()
    return entry_doc


def _mock_task_result(task_id: str = "task-abc"):
    result = MagicMock()
    result.id = task_id
    return result


# ---------------------------------------------------------------------------
# run_file_pipeline
# ---------------------------------------------------------------------------


@pytest.mark.django_db
@pytest.mark.asyncio
class TestProcessAsync:
    @pytest.fixture(autouse=True)
    def tmp_media(self, tmp_path, settings):
        settings.MEDIA_ROOT = str(tmp_path / "media")

    async def test_success_creates_entry_and_marks_completed(self):
        from django_ai_sdk.memories.models import Entry, EntryDocument, Memory
        from django_ai_sdk.memories.tasks import run_file_pipeline

        memory = await Memory.objects.acreate(name="task-success")
        entry_doc = await _make_entry_doc(memory)

        mock_result = MagicMock()
        mock_result.content = "extracted text"
        mock_result.data = {"key": "val"}

        mock_pipeline = MagicMock()
        mock_pipeline.run = AsyncMock(return_value=mock_result)

        with patch(
            "django_ai_sdk.files.common.get_default_file_pipeline",
            return_value=mock_pipeline,
        ):
            await run_file_pipeline(str(entry_doc.id), str(memory.id), None)

        await entry_doc.arefresh_from_db()
        assert entry_doc.processing_status == EntryDocument.ProcessingStatus.COMPLETED
        assert entry_doc.entry_id is not None
        assert entry_doc.extracted is True

        entry = await Entry.objects.aget(id=entry_doc.entry_id)
        assert entry.content == "extracted text"
        assert entry.memory_id == memory.id

    async def test_pipeline_returns_none_marks_failed(self):
        from django_ai_sdk.memories.models import EntryDocument, Memory
        from django_ai_sdk.memories.tasks import run_file_pipeline

        memory = await Memory.objects.acreate(name="task-none")
        entry_doc = await _make_entry_doc(memory)

        mock_pipeline = MagicMock()
        mock_pipeline.run = AsyncMock(return_value=None)

        with patch(
            "django_ai_sdk.files.common.get_default_file_pipeline",
            return_value=mock_pipeline,
        ):
            await run_file_pipeline(str(entry_doc.id), str(memory.id), None)

        await entry_doc.arefresh_from_db()
        assert entry_doc.processing_status == EntryDocument.ProcessingStatus.FAILED
        assert "Unsupported or empty file" in entry_doc.processing_error
        assert entry_doc.entry_id is None

    async def test_pipeline_raises_marks_failed_and_reraises(self):
        from django_ai_sdk.memories.models import EntryDocument, Memory
        from django_ai_sdk.memories.tasks import run_file_pipeline

        memory = await Memory.objects.acreate(name="task-raise")
        entry_doc = await _make_entry_doc(memory)

        mock_pipeline = MagicMock()
        mock_pipeline.run = AsyncMock(side_effect=RuntimeError("OCR failed"))

        with patch(
            "django_ai_sdk.files.common.get_default_file_pipeline",
            return_value=mock_pipeline,
        ):
            with pytest.raises(RuntimeError, match="OCR failed"):
                await run_file_pipeline(str(entry_doc.id), str(memory.id), None)

        await entry_doc.arefresh_from_db()
        assert entry_doc.processing_status == EntryDocument.ProcessingStatus.FAILED
        assert "OCR failed" in entry_doc.processing_error

    async def test_sets_processing_status_before_running_pipeline(self):
        """EntryDocument flips to PROCESSING before pipeline starts."""
        from django_ai_sdk.memories.models import EntryDocument, Memory
        from django_ai_sdk.memories.tasks import run_file_pipeline

        memory = await Memory.objects.acreate(name="task-ordering")
        entry_doc = await _make_entry_doc(memory)

        statuses_seen = []

        async def capture_status(file):
            doc = await EntryDocument.objects.aget(id=entry_doc.id)
            statuses_seen.append(doc.processing_status)
            result = MagicMock()
            result.content = "x"
            result.data = {}
            return result

        mock_pipeline = MagicMock()
        mock_pipeline.run = capture_status

        with patch(
            "django_ai_sdk.files.common.get_default_file_pipeline",
            return_value=mock_pipeline,
        ):
            await run_file_pipeline(str(entry_doc.id), str(memory.id), None)

        assert statuses_seen[0] == EntryDocument.ProcessingStatus.PROCESSING

    async def test_uses_assistant_pipeline_when_assistant_id_given(self):
        from django_ai_sdk.memories.models import EntryDocument, Memory
        from django_ai_sdk.memories.tasks import run_file_pipeline

        memory = await Memory.objects.acreate(name="task-assistant")
        entry_doc = await _make_entry_doc(memory)

        mock_result = MagicMock()
        mock_result.content = "from assistant pipeline"
        mock_result.data = {}

        mock_custom_pipeline = MagicMock()
        mock_custom_pipeline.run = AsyncMock(return_value=mock_result)

        mock_assistant = MagicMock()
        mock_assistant.get_file_pipeline = AsyncMock(return_value=mock_custom_pipeline)

        with patch(
            "django_ai_sdk.assistants.services.AssistantService.get",
            new=AsyncMock(return_value=mock_assistant),
        ):
            await run_file_pipeline(str(entry_doc.id), str(memory.id), "asst-123")

        mock_assistant.get_file_pipeline.assert_called_once()
        mock_custom_pipeline.run.assert_called_once()

        await entry_doc.arefresh_from_db()
        assert entry_doc.processing_status == EntryDocument.ProcessingStatus.COMPLETED


# ---------------------------------------------------------------------------
# upload_document — enqueues, does NOT block
# ---------------------------------------------------------------------------


@pytest.mark.django_db
@pytest.mark.asyncio
class TestUploadDocumentEnqueues:
    @pytest.fixture(autouse=True)
    def tmp_media(self, tmp_path, settings):
        settings.MEDIA_ROOT = str(tmp_path / "media")

    async def test_enqueues_task_not_runs_pipeline(self):
        from django.core.files.uploadedfile import SimpleUploadedFile

        from django_ai_sdk.memories.models import EntryDocument, Memory
        from django_ai_sdk.memories.services import MemoryService
        from tests.mocks.permissions import memory_permissions

        memory = await Memory.objects.acreate(name="upload-enqueue")
        uploaded = SimpleUploadedFile("doc.txt", b"content", content_type="text/plain")

        mock_pipeline = MagicMock()
        mock_pipeline.run = AsyncMock()

        with (
            memory_permissions("django_ai_sdk.permissions.AllowAll"),
            patch(
                "django_ai_sdk.memories.services.process_document_upload",
            ) as mock_task,
            patch(
                "django_ai_sdk.files.common.get_default_file_pipeline",
                return_value=mock_pipeline,
            ),
        ):
            mock_task.aenqueue = AsyncMock(return_value=_mock_task_result("task-xyz"))
            await MemoryService.upload_document(str(memory.id), uploaded, user=None)

        mock_task.aenqueue.assert_called_once()
        # pipeline.run NOT called — task was enqueued, not run inline
        mock_pipeline.run.assert_not_called()

    async def test_returns_document_upload_response(self):
        from django.core.files.uploadedfile import SimpleUploadedFile

        from django_ai_sdk.memories.models import Memory
        from django_ai_sdk.memories.schemas import DocumentUploadResponse
        from django_ai_sdk.memories.services import MemoryService
        from tests.mocks.permissions import memory_permissions

        memory = await Memory.objects.acreate(name="upload-response")
        uploaded = SimpleUploadedFile("doc.pdf", b"%PDF", content_type="application/pdf")

        with (
            memory_permissions("django_ai_sdk.permissions.AllowAll"),
            patch("django_ai_sdk.memories.services.process_document_upload") as mock_task,
        ):
            mock_task.aenqueue = AsyncMock(return_value=_mock_task_result("task-resp"))
            result = await MemoryService.upload_document(str(memory.id), uploaded, user=None)

        assert isinstance(result, DocumentUploadResponse)
        assert result.id  # has a UUID
        assert result.status == "processing"

    async def test_entry_doc_saved_before_task_enqueued(self):
        """EntryDocument must be in DB before task is enqueued (task needs the ID)."""
        from django.core.files.uploadedfile import SimpleUploadedFile

        from django_ai_sdk.memories.models import EntryDocument, Memory
        from django_ai_sdk.memories.services import MemoryService
        from tests.mocks.permissions import memory_permissions

        memory = await Memory.objects.acreate(name="upload-order")
        uploaded = SimpleUploadedFile("doc.txt", b"data", content_type="text/plain")

        captured_args = []

        async def capture_enqueue(*args, **kwargs):
            # verify the doc exists in DB at enqueue time
            doc_id = args[0]
            exists = await EntryDocument.objects.filter(id=doc_id).aexists()
            captured_args.append((doc_id, exists))
            return _mock_task_result("task-order")

        with (
            memory_permissions("django_ai_sdk.permissions.AllowAll"),
            patch("django_ai_sdk.memories.services.process_document_upload") as mock_task,
        ):
            mock_task.aenqueue = capture_enqueue
            await MemoryService.upload_document(str(memory.id), uploaded, user=None)

        assert captured_args[0][1] is True  # doc existed in DB when enqueue was called


# ---------------------------------------------------------------------------
# get_document_status
# ---------------------------------------------------------------------------


@pytest.mark.django_db
@pytest.mark.asyncio
class TestGetDocumentStatus:
    @pytest.fixture(autouse=True)
    def tmp_media(self, tmp_path, settings):
        settings.MEDIA_ROOT = str(tmp_path / "media")

    async def test_returns_status_with_task_info(self):
        from django_ai_sdk.memories.models import EntryDocument, Memory
        from django_ai_sdk.memories.schemas import DocumentStatusOut
        from django_ai_sdk.memories.services import MemoryService
        from django_ai_sdk.tasks import TaskStatus

        memory = await Memory.objects.acreate(name="status-test")
        entry_doc = await _make_entry_doc(memory)
        entry_doc.task_id = "task-status-1"
        entry_doc.processing_status = EntryDocument.ProcessingStatus.PROCESSING
        await entry_doc.asave(update_fields=["task_id", "processing_status"])

        mock_task_status = TaskStatus(
            id="task-status-1",
            status="running",
            enqueued_at=None,
            started_at=None,
            finished_at=None,
        )

        with patch(
            "django_ai_sdk.memories.services.aget_task_status",
            new=AsyncMock(return_value=mock_task_status),
        ):
            result = await MemoryService.get_document_status(str(entry_doc.id))

        assert isinstance(result, DocumentStatusOut)
        assert result.id == str(entry_doc.id)
        assert result.status == "processing"
        assert result.task is not None
        assert result.task.id == "task-status-1"

    async def test_returns_status_without_task_when_no_task_id(self):
        from django_ai_sdk.memories.models import EntryDocument, Memory
        from django_ai_sdk.memories.services import MemoryService

        memory = await Memory.objects.acreate(name="status-no-task")
        entry_doc = await _make_entry_doc(memory)
        # task_id left null

        result = await MemoryService.get_document_status(str(entry_doc.id))

        assert result.id == str(entry_doc.id)
        assert result.status == "pending"
        assert result.task is None

    async def test_returns_failed_status_with_error(self):
        from django_ai_sdk.memories.models import EntryDocument, Memory
        from django_ai_sdk.memories.services import MemoryService

        memory = await Memory.objects.acreate(name="status-failed")
        entry_doc = await _make_entry_doc(memory)
        entry_doc.processing_status = EntryDocument.ProcessingStatus.FAILED
        entry_doc.processing_error = "File type not supported"
        await entry_doc.asave(update_fields=["processing_status", "processing_error"])

        result = await MemoryService.get_document_status(str(entry_doc.id))

        assert result.status == "failed"
        assert result.error == "File type not supported"
