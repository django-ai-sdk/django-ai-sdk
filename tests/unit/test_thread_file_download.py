"""Thread file downloads through both contrib layers (tests/urls.py)."""

import pytest
from asgiref.sync import async_to_sync
from django.core.files.base import ContentFile

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


@pytest.fixture
def owner(django_user_model):
    return django_user_model.objects.create(email="owner@example.com")


@pytest.fixture
def stranger(django_user_model):
    return django_user_model.objects.create(email="stranger@example.com")


@pytest.fixture
def thread_file(owner, settings, tmp_path, mock_agents_registry):
    """A thread with two uploaded files: an image and a PDF."""
    from django_ai_sdk.conversation.models import Thread
    from django_ai_sdk.memories.models import EntryDocument
    from django_ai_sdk.memories.services import MemoryService

    settings.MEDIA_ROOT = str(tmp_path)
    thread_id = str(Thread.objects.create(user=owner, metadata={"agent_id": "test-agent"}).id)
    memory = async_to_sync(MemoryService.get_or_create_thread_file_memory)(thread_id)

    def upload(name, content_type, data):
        doc = EntryDocument(memory=memory, file_name=name, content_type=content_type)
        doc.file.save(name, ContentFile(data), save=True)
        return str(doc.id)

    return thread_id, upload("cat.png", "image/png", PNG), upload("a.pdf", "application/pdf", b"%PDF")


URLS = [
    "/api/memories/thread/{t}/files/{d}/download",  # ninja
    "/drf/threads/{t}/files/{d}/download/",  # drf
]


@pytest.mark.django_db
@pytest.mark.parametrize("url", URLS)
class TestDownload:
    def test_an_image_renders_inline_but_sandboxed(self, client, owner, thread_file, url):
        thread_id, image_id, _ = thread_file
        client.force_login(owner)

        response = client.get(url.format(t=thread_id, d=image_id))

        assert response.status_code == 200
        assert response["Content-Type"] == "image/png"
        assert "attachment" not in response.get("Content-Disposition", "")
        assert response["X-Content-Type-Options"] == "nosniff"
        assert response["Content-Security-Policy"] == "sandbox"
        assert b"".join(response.streaming_content) == PNG

    def test_anything_else_downloads(self, client, owner, thread_file, url):
        thread_id, _, pdf_id = thread_file
        client.force_login(owner)

        response = client.get(url.format(t=thread_id, d=pdf_id))

        assert response.status_code == 200
        assert response["Content-Type"] == "application/octet-stream"
        assert response["Content-Disposition"].startswith("attachment")

    def test_someone_elses_file_is_not_served(self, client, stranger, thread_file, url):
        thread_id, image_id, _ = thread_file
        client.force_login(stranger)

        response = client.get(url.format(t=thread_id, d=image_id))

        assert response.status_code in (403, 404)
        assert "code" in response.json()

    def test_an_unknown_file_is_not_found(self, client, owner, thread_file, url):
        thread_id, _, _ = thread_file
        client.force_login(owner)

        response = client.get(url.format(t=thread_id, d="00000000-0000-0000-0000-000000000000"))

        assert response.status_code == 404
        assert response.json()["code"] == "not_found"


@pytest.mark.django_db
class TestThreadFileUrl:
    def test_reverses_the_download_view_or_falls_back_to_storage(self, settings, tmp_path):
        from django_ai_sdk.memories.models import EntryDocument
        from django_ai_sdk.memories.services import get_thread_file_url

        settings.MEDIA_ROOT = str(tmp_path)
        doc = EntryDocument(file_name="cat.png")
        doc.file.save("cat.png", ContentFile(PNG), save=False)

        # tests/urls.py mounts the contrib memories router on the "ai-test" API.
        settings.AI_SDK_THREAD_FILE_URL_NAME = "ai-test:download_thread_file"
        assert get_thread_file_url(doc, "t1") == (
            f"/api/memories/thread/t1/files/{doc.id}/download"
        )

        settings.AI_SDK_THREAD_FILE_URL_NAME = None
        assert get_thread_file_url(doc, "t1") == doc.file.url
