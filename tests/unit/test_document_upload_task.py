from __future__ import annotations

import hashlib
import asyncio

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


async def _processing_entry_doc(memory, task_id: str):
    from django_ai_sdk.memories.models import EntryDocument

    entry_doc = await _make_entry_doc(memory)
    entry_doc.task_id = task_id
    entry_doc.processing_status = EntryDocument.ProcessingStatus.PROCESSING
    await entry_doc.asave(update_fields=["task_id", "processing_status"])
    return entry_doc


async def _get_status_with_mocked_task(entry_doc, task_status):
    from django_ai_sdk.memories.services import MemoryService

    with patch(
        "django_ai_sdk.memories.services.aget_task_status",
        new=AsyncMock(return_value=task_status),
    ):
        return await MemoryService.get_document_status(
            str(entry_doc.id), user=MagicMock(is_authenticated=True)
        )


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
            result = await run_file_pipeline(str(entry_doc.id), str(memory.id), None)

        await entry_doc.arefresh_from_db()
        assert entry_doc.processing_status == EntryDocument.ProcessingStatus.COMPLETED
        assert entry_doc.entry_id is not None
        assert entry_doc.extracted is True

        entry = await Entry.objects.aget(id=entry_doc.entry_id)
        assert entry.content == "extracted text"
        assert entry.memory_id == memory.id
        assert result == {"entry_id": str(entry.id), "extracted": True}

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

    async def test_cancelled_at_stops_at_next_step_and_does_not_clobber(self):
        """cancel_document already wrote CANCELLED; the pipeline should stop
        at the next on_step checkpoint and leave that status alone."""
        from django.utils import timezone

        from django_ai_sdk.memories.models import EntryDocument, Memory
        from django_ai_sdk.memories.tasks import run_file_pipeline

        memory = await Memory.objects.acreate(name="task-cancel-mid")
        entry_doc = await _make_entry_doc(memory)

        async def _pipeline_run(file, on_step=None):
            await EntryDocument.objects.filter(id=entry_doc.id).aupdate(
                cancelled_at=timezone.now(),
                processing_status=EntryDocument.ProcessingStatus.CANCELLED,
                processing_error="Cancelled by user",
            )
            await on_step("extracting")
            raise AssertionError("should not reach here")

        mock_pipeline = MagicMock()
        mock_pipeline.run = _pipeline_run

        with patch(
            "django_ai_sdk.files.common.get_default_file_pipeline",
            return_value=mock_pipeline,
        ):
            result = await run_file_pipeline(str(entry_doc.id), str(memory.id), None)

        assert result is None
        await entry_doc.arefresh_from_db()
        assert entry_doc.processing_status == EntryDocument.ProcessingStatus.CANCELLED
        assert entry_doc.processing_error == "Cancelled by user"

    async def test_cancelled_at_after_last_checkpoint_does_not_clobber_completed(self):
        """Cancelled after the last on_step call but before the pipeline
        finished — the COMPLETED write must not overwrite CANCELLED."""
        from django.utils import timezone

        from django_ai_sdk.memories.models import Entry, EntryDocument, Memory
        from django_ai_sdk.memories.tasks import run_file_pipeline

        memory = await Memory.objects.acreate(name="task-cancel-late")
        entry_doc = await _make_entry_doc(memory)

        async def _pipeline_run(file, on_step=None):
            await on_step("ocr")
            await EntryDocument.objects.filter(id=entry_doc.id).aupdate(
                cancelled_at=timezone.now(),
                processing_status=EntryDocument.ProcessingStatus.CANCELLED,
                processing_error="Cancelled by user",
            )
            result = MagicMock()
            result.content = "x"
            result.data = {}
            return result

        mock_pipeline = MagicMock()
        mock_pipeline.run = _pipeline_run

        with patch(
            "django_ai_sdk.files.common.get_default_file_pipeline",
            return_value=mock_pipeline,
        ):
            result = await run_file_pipeline(str(entry_doc.id), str(memory.id), None)

        assert result is None
        await entry_doc.arefresh_from_db()
        assert entry_doc.processing_status == EntryDocument.ProcessingStatus.CANCELLED
        assert not await Entry.objects.filter(document=entry_doc).aexists()

    async def test_step_callback_updates_and_resets_processing_step(self):
        """Also covers that PROCESSING is set before the pipeline starts."""
        from django_ai_sdk.memories.models import EntryDocument, Memory
        from django_ai_sdk.memories.tasks import run_file_pipeline

        memory = await Memory.objects.acreate(name="task-step")
        entry_doc = await _make_entry_doc(memory)

        steps_seen = []

        async def capture_step(file, on_step=None):
            await on_step("ocr")
            doc = await EntryDocument.objects.aget(id=entry_doc.id)
            steps_seen.append(doc.processing_step)
            assert doc.processing_status == EntryDocument.ProcessingStatus.PROCESSING
            await on_step("extracting")
            doc = await EntryDocument.objects.aget(id=entry_doc.id)
            steps_seen.append(doc.processing_step)
            result = MagicMock()
            result.content = "x"
            result.data = {}
            return result

        mock_pipeline = MagicMock()
        mock_pipeline.run = capture_step

        with patch(
            "django_ai_sdk.files.common.get_default_file_pipeline",
            return_value=mock_pipeline,
        ):
            await run_file_pipeline(str(entry_doc.id), str(memory.id), None)

        assert steps_seen == ["ocr", "extracting"]

        await entry_doc.arefresh_from_db()
        assert entry_doc.processing_status == EntryDocument.ProcessingStatus.COMPLETED
        assert entry_doc.processing_step is None  # reset once the run finishes

    async def test_timeout_marks_failed_with_clear_message(self):
        from django_ai_sdk.memories import tasks as tasks_module
        from django_ai_sdk.memories.models import EntryDocument, Memory
        from django_ai_sdk.memories.tasks import run_file_pipeline

        memory = await Memory.objects.acreate(name="task-timeout")
        entry_doc = await _make_entry_doc(memory)

        async def _hangs_forever(file, on_step=None):
            await asyncio.sleep(10)

        mock_pipeline = MagicMock()
        mock_pipeline.run = _hangs_forever

        with (
            patch.object(tasks_module, "PIPELINE_TIMEOUT_SECONDS", 0.05),
            patch(
                "django_ai_sdk.files.common.get_default_file_pipeline",
                return_value=mock_pipeline,
            ),
        ):
            with pytest.raises(TimeoutError):
                await run_file_pipeline(str(entry_doc.id), str(memory.id), None)

        await entry_doc.arefresh_from_db()
        assert entry_doc.processing_status == EntryDocument.ProcessingStatus.FAILED
        assert "timed out" in entry_doc.processing_error.lower()
        assert entry_doc.processing_step is None

    async def test_uses_agent_pipeline_when_agent_id_given(self):
        from django_ai_sdk.memories.models import EntryDocument, Memory
        from django_ai_sdk.memories.tasks import run_file_pipeline

        memory = await Memory.objects.acreate(name="task-agent")
        entry_doc = await _make_entry_doc(memory)

        mock_result = MagicMock()
        mock_result.content = "from agent pipeline"
        mock_result.data = {}

        mock_custom_pipeline = MagicMock()
        mock_custom_pipeline.run = AsyncMock(return_value=mock_result)

        mock_agent = MagicMock()
        mock_agent.get_file_pipeline = AsyncMock(return_value=mock_custom_pipeline)

        with patch(
            "django_ai_sdk.agents.services.AgentService.get",
            new=AsyncMock(return_value=mock_agent),
        ):
            await run_file_pipeline(str(entry_doc.id), str(memory.id), "asst-123")

        mock_agent.get_file_pipeline.assert_called_once()
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
        from django_ai_sdk.memories.models import Memory
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
# Deduplication — same hash, same memory
# ---------------------------------------------------------------------------


@pytest.mark.django_db
@pytest.mark.asyncio
class TestUploadDocumentDedup:
    @pytest.fixture(autouse=True)
    def tmp_media(self, tmp_path, settings):
        settings.MEDIA_ROOT = str(tmp_path / "media")

    async def test_duplicate_file_raises_conflict(self):
        from django.core.files.uploadedfile import SimpleUploadedFile
        from django_ai_sdk.memories.models import Memory
        from django_ai_sdk.memories.services import MemoryService
        from django_ai_sdk.permissions import ConflictError
        from tests.mocks.permissions import memory_permissions

        memory = await Memory.objects.acreate(name="dedup-conflict")
        uploaded = SimpleUploadedFile("doc.txt", b"same content", content_type="text/plain")

        with (
            memory_permissions("django_ai_sdk.permissions.AllowAll"),
            patch("django_ai_sdk.memories.services.process_document_upload") as mock_task,
        ):
            mock_task.aenqueue = AsyncMock(return_value=_mock_task_result("task-1"))
            await MemoryService.upload_document(str(memory.id), uploaded, user=None)

        uploaded2 = SimpleUploadedFile("doc.txt", b"same content", content_type="text/plain")
        with (
            memory_permissions("django_ai_sdk.permissions.AllowAll"),
            pytest.raises(ConflictError, match="File already exists"),
        ):
            await MemoryService.upload_document(str(memory.id), uploaded2, user=None)

    async def test_duplicate_file_different_memory_allowed(self):
        from django.core.files.uploadedfile import SimpleUploadedFile
        from django_ai_sdk.memories.models import Memory
        from django_ai_sdk.memories.services import MemoryService
        from tests.mocks.permissions import memory_permissions

        memory_a = await Memory.objects.acreate(name="dedup-mem-a")
        memory_b = await Memory.objects.acreate(name="dedup-mem-b")
        uploaded = SimpleUploadedFile("doc.txt", b"cross memory", content_type="text/plain")

        with (
            memory_permissions("django_ai_sdk.permissions.AllowAll"),
            patch("django_ai_sdk.memories.services.process_document_upload") as mock_task,
        ):
            mock_task.aenqueue = AsyncMock(return_value=_mock_task_result("task-1"))
            await MemoryService.upload_document(str(memory_a.id), uploaded, user=None)

        uploaded2 = SimpleUploadedFile("doc.txt", b"cross memory", content_type="text/plain")
        with (
            memory_permissions("django_ai_sdk.permissions.AllowAll"),
            patch("django_ai_sdk.memories.services.process_document_upload") as mock_task,
        ):
            mock_task.aenqueue = AsyncMock(return_value=_mock_task_result("task-2"))
            result = await MemoryService.upload_document(str(memory_b.id), uploaded2, user=None)

        assert result.status == "processing"

    async def test_duplicate_after_failed_allows_retry(self):
        from django.core.files.uploadedfile import SimpleUploadedFile
        from django_ai_sdk.memories.models import EntryDocument, Memory
        from django_ai_sdk.memories.services import MemoryService
        from tests.mocks.permissions import memory_permissions

        memory = await Memory.objects.acreate(name="dedup-retry")
        uploaded = SimpleUploadedFile("doc.txt", b"retry content", content_type="text/plain")

        with (
            memory_permissions("django_ai_sdk.permissions.AllowAll"),
            patch("django_ai_sdk.memories.services.process_document_upload") as mock_task,
        ):
            mock_task.aenqueue = AsyncMock(return_value=_mock_task_result("task-1"))
            result = await MemoryService.upload_document(str(memory.id), uploaded, user=None)

        # Manually mark the first doc as FAILED
        doc = await EntryDocument.objects.aget(id=result.id)
        doc.processing_status = EntryDocument.ProcessingStatus.FAILED
        await doc.asave(update_fields=["processing_status"])

        # Re-upload with same content — should succeed
        uploaded2 = SimpleUploadedFile("doc.txt", b"retry content", content_type="text/plain")
        with (
            memory_permissions("django_ai_sdk.permissions.AllowAll"),
            patch("django_ai_sdk.memories.services.process_document_upload") as mock_task,
        ):
            mock_task.aenqueue = AsyncMock(return_value=_mock_task_result("task-2"))
            result2 = await MemoryService.upload_document(str(memory.id), uploaded2, user=None)

        assert result2.status == "processing"
        assert result2.id != result.id


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
            result = await MemoryService.get_document_status(
                str(entry_doc.id), user=MagicMock(is_authenticated=True)
            )

        assert isinstance(result, DocumentStatusOut)
        assert result.id == str(entry_doc.id)
        assert result.status == "processing"
        assert result.task is not None
        assert result.task.id == "task-status-1"

    async def test_returns_status_without_task_when_no_task_id(self):
        from django_ai_sdk.memories.models import Memory
        from django_ai_sdk.memories.services import MemoryService

        memory = await Memory.objects.acreate(name="status-no-task")
        entry_doc = await _make_entry_doc(memory)
        # task_id left null

        result = await MemoryService.get_document_status(
            str(entry_doc.id), user=MagicMock(is_authenticated=True)
        )

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

        result = await MemoryService.get_document_status(
            str(entry_doc.id), user=MagicMock(is_authenticated=True)
        )

        assert result.status == "failed"
        assert result.error == "File type not supported"


# ---------------------------------------------------------------------------
# compute_file_hash
# ---------------------------------------------------------------------------


class TestComputeFileHash:
    def test_bytes_input(self):
        from django_ai_sdk.files.common import compute_file_hash

        result = compute_file_hash(b"hello world")
        assert result == hashlib.sha256(b"hello world").hexdigest()
        assert len(result) == 64

    def test_io_input_resets_pointer(self):
        from io import BytesIO

        from django_ai_sdk.files.common import compute_file_hash

        data = b"some file content"
        stream = BytesIO(data)
        result = compute_file_hash(stream)
        assert result == hashlib.sha256(data).hexdigest()
        # pointer should be reset to 0
        assert stream.tell() == 0
        assert stream.read() == data

    def test_empty_bytes(self):
        from django_ai_sdk.files.common import compute_file_hash

        result = compute_file_hash(b"")
        assert result == hashlib.sha256(b"").hexdigest()
        assert len(result) == 64

    def test_large_content_chunking(self):
        from django_ai_sdk.files.common import compute_file_hash

        big_data = b"x" * 200000  # > 64KB
        result = compute_file_hash(big_data)
        assert result == hashlib.sha256(big_data).hexdigest()


# ---------------------------------------------------------------------------
# get_task_status
# ---------------------------------------------------------------------------


@pytest.mark.django_db
@pytest.mark.asyncio
class TestGetTaskStatus:
    @pytest.fixture(autouse=True)
    def tmp_media(self, tmp_path, settings):
        settings.MEDIA_ROOT = str(tmp_path / "media")

    async def test_returns_same_status_as_get_document_status(self):
        from django_ai_sdk.memories.models import EntryDocument, Memory
        from django_ai_sdk.memories.schemas import DocumentStatusOut
        from django_ai_sdk.memories.services import MemoryService
        from django_ai_sdk.tasks import TaskStatus

        memory = await Memory.objects.acreate(name="task-status-test")
        entry_doc = await _make_entry_doc(memory)
        entry_doc.task_id = "task-lookup-1"
        entry_doc.processing_status = EntryDocument.ProcessingStatus.PROCESSING
        entry_doc.processing_step = "ocr"
        await entry_doc.asave(
            update_fields=["task_id", "processing_status", "processing_step"]
        )

        mock_task_status = TaskStatus(
            id="task-lookup-1",
            status="running",
            enqueued_at=None,
            started_at=None,
            finished_at=None,
        )

        with patch(
            "django_ai_sdk.memories.services.aget_task_status",
            new=AsyncMock(return_value=mock_task_status),
        ):
            by_task = await MemoryService.get_task_status(
                "task-lookup-1", user=MagicMock(is_authenticated=True)
            )
            by_doc = await MemoryService.get_document_status(
                str(entry_doc.id), user=MagicMock(is_authenticated=True)
            )

        assert isinstance(by_task, DocumentStatusOut)
        assert by_task == by_doc
        assert by_task.processing_step == "ocr"
        assert by_task.task is not None
        assert by_task.task.id == "task-lookup-1"

    async def test_unknown_task_id_raises_value_error(self):
        from django_ai_sdk.memories.services import MemoryService

        with pytest.raises(ValueError):
            await MemoryService.get_task_status(
                "no-such-task", user=MagicMock(is_authenticated=True)
            )


# ---------------------------------------------------------------------------
# cancel_document
# ---------------------------------------------------------------------------


@pytest.mark.django_db
@pytest.mark.asyncio
class TestCancelDocument:
    @pytest.fixture(autouse=True)
    def tmp_media(self, tmp_path, settings):
        settings.MEDIA_ROOT = str(tmp_path / "media")

    async def _make_owned_entry_doc(self, *, status):
        import uuid

        from django.contrib.auth import get_user_model

        from django_ai_sdk.memories.models import EntryDocument, Memory, MemoryUser

        UserModel = get_user_model()
        user = await UserModel.objects.acreate(username=f"owner-{uuid.uuid4()}")
        # Private memory — an unrelated authenticated user must still be denied.
        memory = await Memory.objects.acreate(name="cancel-test", is_public=False)
        await MemoryUser.objects.acreate(memory=memory, user=user, can_manage=True)
        entry_doc = await _make_entry_doc(memory)
        entry_doc.memory = memory
        entry_doc.processing_status = status
        await entry_doc.asave(update_fields=["memory", "processing_status"])
        return user, entry_doc

    async def test_cancels_pending_document(self):
        from django_ai_sdk.memories.models import EntryDocument
        from django_ai_sdk.memories.services import MemoryService

        user, entry_doc = await self._make_owned_entry_doc(
            status=EntryDocument.ProcessingStatus.PENDING
        )

        result = await MemoryService.cancel_document(str(entry_doc.id), user=user)

        assert result.status == "cancelled"
        await entry_doc.arefresh_from_db()
        assert entry_doc.processing_status == EntryDocument.ProcessingStatus.CANCELLED
        assert entry_doc.cancelled_at is not None

    async def test_cancels_processing_document(self):
        from django_ai_sdk.memories.models import EntryDocument
        from django_ai_sdk.memories.services import MemoryService

        user, entry_doc = await self._make_owned_entry_doc(
            status=EntryDocument.ProcessingStatus.PROCESSING
        )

        result = await MemoryService.cancel_document(str(entry_doc.id), user=user)

        assert result.status == "cancelled"
        assert result.error == "Cancelled by user"
        await entry_doc.arefresh_from_db()
        assert entry_doc.processing_status == EntryDocument.ProcessingStatus.CANCELLED
        assert entry_doc.cancelled_at is not None

    async def test_cannot_cancel_completed_document(self):
        from django_ai_sdk.memories.models import EntryDocument
        from django_ai_sdk.memories.services import MemoryService

        user, entry_doc = await self._make_owned_entry_doc(
            status=EntryDocument.ProcessingStatus.COMPLETED
        )

        with pytest.raises(ValueError):
            await MemoryService.cancel_document(str(entry_doc.id), user=user)

    async def test_requires_permission(self):
        from django.contrib.auth import get_user_model

        from django_ai_sdk.memories.models import EntryDocument
        from django_ai_sdk.memories.services import MemoryService
        from django_ai_sdk.permissions import PermissionDenied

        UserModel = get_user_model()
        other_user = await UserModel.objects.acreate(username="not-the-owner")
        _, entry_doc = await self._make_owned_entry_doc(
            status=EntryDocument.ProcessingStatus.PROCESSING
        )

        with pytest.raises(PermissionDenied):
            await MemoryService.cancel_document(str(entry_doc.id), user=other_user)


# ---------------------------------------------------------------------------
# self-healing stuck PROCESSING documents
# ---------------------------------------------------------------------------


@pytest.mark.django_db
@pytest.mark.asyncio
class TestSelfHealStaleProcessing:
    @pytest.fixture(autouse=True)
    def tmp_media(self, tmp_path, settings):
        settings.MEDIA_ROOT = str(tmp_path / "media")

    async def test_marks_failed_when_task_already_finished(self):
        """Worker died after the task backend recorded FAILED/SUCCESSFUL but
        before our own EntryDocument write landed."""
        from django_ai_sdk.memories.models import EntryDocument, Memory
        from django_ai_sdk.tasks import TaskError, TaskStatus

        memory = await Memory.objects.acreate(name="heal-finished")
        entry_doc = await _processing_entry_doc(memory, "task-heal-1")

        mock_task_status = TaskStatus(
            id="task-heal-1",
            status="FAILED",
            enqueued_at=None,
            started_at=None,
            finished_at=None,
            errors=[TaskError(type="RuntimeError", traceback="...\nRuntimeError: boom")],
        )
        result = await _get_status_with_mocked_task(entry_doc, mock_task_status)

        assert result.status == "failed"
        assert "RuntimeError: boom" in result.error
        await entry_doc.arefresh_from_db()
        assert entry_doc.processing_status == EntryDocument.ProcessingStatus.FAILED

    async def test_marks_failed_when_running_past_timeout_grace(self):
        """Task backend still says RUNNING but the worker is long gone."""
        from datetime import timedelta

        from django.utils import timezone

        from django_ai_sdk.memories.models import EntryDocument, Memory
        from django_ai_sdk.memories.tasks import PIPELINE_TIMEOUT_SECONDS
        from django_ai_sdk.tasks import TaskStatus

        memory = await Memory.objects.acreate(name="heal-hung")
        entry_doc = await _processing_entry_doc(memory, "task-heal-2")

        long_ago = timezone.now() - timedelta(seconds=PIPELINE_TIMEOUT_SECONDS + 3600)
        mock_task_status = TaskStatus(
            id="task-heal-2",
            status="RUNNING",
            enqueued_at=long_ago,
            started_at=long_ago,
            finished_at=None,
        )
        result = await _get_status_with_mocked_task(entry_doc, mock_task_status)

        assert result.status == "failed"
        await entry_doc.arefresh_from_db()
        assert entry_doc.processing_status == EntryDocument.ProcessingStatus.FAILED

    async def test_does_not_touch_recently_started_running_task(self):
        from django_ai_sdk.memories.models import EntryDocument, Memory
        from django_ai_sdk.tasks import TaskStatus

        memory = await Memory.objects.acreate(name="heal-fresh")
        entry_doc = await _processing_entry_doc(memory, "task-heal-3")

        mock_task_status = TaskStatus(
            id="task-heal-3",
            status="RUNNING",
            enqueued_at=None,
            started_at=None,
            finished_at=None,
        )
        result = await _get_status_with_mocked_task(entry_doc, mock_task_status)

        assert result.status == "processing"
        await entry_doc.arefresh_from_db()
        assert entry_doc.processing_status == EntryDocument.ProcessingStatus.PROCESSING
