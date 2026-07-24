"""Image bytes are offloaded to the storage backend, not stored in the DB."""

from __future__ import annotations

import base64
import uuid

import pytest
from asgiref.sync import sync_to_async

from django_ai_sdk.common import ChatMessage, ImageAttachment
from django_ai_sdk.conversation.models import Message, MessageImage
from django_ai_sdk.storage.db import DbStorageAdapter

RAW = b"\x89PNG\r\n\x1a\nfake-image-bytes"
B64 = base64.b64encode(RAW).decode()


@pytest.fixture(autouse=True)
def _media_root(settings, tmp_path):
    # Setting MEDIA_ROOT via the settings fixture resets Django's storage cache,
    # so files land in a throwaway tmp dir instead of the project tree.
    settings.MEDIA_ROOT = str(tmp_path)


@pytest.mark.django_db(transaction=True)
class TestImageStorageOffload:
    async def _thread(self) -> str:
        thread_id = str(uuid.uuid4())
        await DbStorageAdapter.create_thread(title="t", thread_id=thread_id)
        return thread_id

    def _user_message(self) -> ChatMessage:
        return ChatMessage(
            role="user",
            content="what is this?",
            id=str(uuid.uuid4()),
            images=[ImageAttachment(media_type="image/png", data=B64)],
        )

    @pytest.mark.asyncio
    async def test_base64_is_not_stored_in_the_db(self):
        thread_id = await self._thread()
        adapter = DbStorageAdapter(thread_id)

        message_id = await adapter.store_chat_message(self._user_message())

        # Message.result keeps a reference (id) only — never the base64 payload.
        message = await Message.objects.aget(id=message_id)
        [ref] = message.result["images"]
        assert ref["media_type"] == "image/png"
        assert ref["id"]
        assert "data" not in ref

        # The bytes live in a MessageImage row / storage backend.
        image = await MessageImage.objects.aget(id=ref["id"])
        assert image.file_size == len(RAW)
        assert image.file_hash
        assert image.file.name.endswith(".png")

    @pytest.mark.asyncio
    async def test_bytes_rehydrate_on_read(self):
        thread_id = await self._thread()
        adapter = DbStorageAdapter(thread_id)
        await adapter.store_chat_message(self._user_message())

        [loaded] = await adapter.get_messages()
        assert len(loaded.images) == 1
        assert loaded.images[0].data == B64  # round-trips back to the original base64
        assert loaded.images[0].id

    @pytest.mark.asyncio
    async def test_file_removed_when_message_deleted(self):
        thread_id = await self._thread()
        adapter = DbStorageAdapter(thread_id)
        message_id = await adapter.store_chat_message(self._user_message())

        image = await MessageImage.objects.aget(message_id=message_id)
        storage, name = image.file.storage, image.file.name
        assert await sync_to_async(storage.exists)(name)

        # Hard-deleting the thread cascades to messages and their images.
        await DbStorageAdapter.delete_thread(thread_id)

        assert not await MessageImage.objects.filter(id=image.id).aexists()
        assert not await sync_to_async(storage.exists)(name)
