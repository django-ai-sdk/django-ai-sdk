from unittest.mock import AsyncMock, patch

import pytest
from django.core.exceptions import ImproperlyConfigured
from durastream import AsyncStore

from django_ai_sdk import streams


@pytest.fixture(autouse=True)
def _reset_store_singleton():
    streams._store = None
    yield
    streams._store = None


class TestGetStore:
    def test_raises_when_setting_unset(self, settings):
        settings.AI_SDK_DURABLE_STREAMS_PATH = None
        with pytest.raises(ImproperlyConfigured):
            streams.get_store()

    def test_builds_and_caches_store(self, settings, tmp_path):
        settings.AI_SDK_DURABLE_STREAMS_PATH = str(tmp_path / "durable_streams")
        store = streams.get_store()
        assert isinstance(store, AsyncStore)
        assert streams.get_store() is store  # cached singleton


@pytest.mark.asyncio
class TestRunAndTailDurableStream:
    async def test_persists_chunks_and_cleans_up_on_close(self, tmp_path, monkeypatch):
        monkeypatch.setattr(streams, "_DELETE_GRACE_SECONDS", 0)
        store = AsyncStore(str(tmp_path / "durable_streams"))

        async def sse_gen():
            yield b"data: {\"type\": \"start\"}\n\n"
            yield b"data: {\"type\": \"text-delta\", \"delta\": \"hi\"}\n\n"

        with patch.object(
            streams, "clear_active_stream", new_callable=AsyncMock
        ) as mock_clear:
            await streams.run_durable_stream(sse_gen(), store, "msg-1", "thread-1")

        # active_stream_id
        mock_clear.assert_awaited_once_with("thread-1")

        # cleanup deletes the log
        fresh = await store.create("msg-1", "text/event-stream")
        assert fresh.next_offset == 0

    async def test_tail_stream_replays_the_persisted_log_without_double_done(
        self, tmp_path
    ):
        store = AsyncStore(str(tmp_path / "durable_streams"))
        stream = await store.create("msg-2", "text/event-stream")
        await stream.append(b"data: {\"type\": \"start\"}\n\n")
        await stream.append(b"data: [DONE]\n\n")
        await stream.close()

        chunks = [chunk async for chunk in streams.tail_stream(store, "msg-2")]

        assert chunks == [
            b"data: {\"type\": \"start\"}\n\n",
            b"data: [DONE]\n\n",
        ]

    async def test_spawned_task_is_not_garbage_collected_before_completion(
        self, tmp_path, monkeypatch
    ):
        import asyncio
        import gc

        monkeypatch.setattr(streams, "_DELETE_GRACE_SECONDS", 0)
        store = AsyncStore(str(tmp_path / "durable_streams"))

        async def slow_sse_gen():
            for i in range(5):
                await asyncio.sleep(0)
                yield f"data: {i}\n\n".encode()

        with patch.object(streams, "clear_active_stream", new_callable=AsyncMock):
            task = streams.spawn_background(
                streams.run_durable_stream(slow_sse_gen(), store, "msg-3", "thread-1")
            )
            del task
            gc.collect()  # would GC an untracked task
            await asyncio.sleep(0.05)

        assert len(streams._background_tasks) == 0  # ran to completion, self-removed

    async def test_background_task_does_not_inherit_request_context(self):
        import asyncio
        import contextvars

        probe: contextvars.ContextVar[str] = contextvars.ContextVar("probe")
        probe.set("request-scoped")
        seen = []

        async def read_probe():
            seen.append(probe.get("not-inherited"))

        streams.spawn_background(read_probe())
        await asyncio.sleep(0.01)

        assert seen == ["not-inherited"]

    async def test_background_task_failure_is_logged_not_swallowed(self):
        import asyncio

        async def boom():
            raise RuntimeError("generation exploded")

        with patch.object(streams, "logger") as mock_logger:
            streams.spawn_background(boom())
            await asyncio.sleep(0.01)

        # loguru attaches traceback
        mock_logger.opt.assert_called_once()
        assert isinstance(mock_logger.opt.call_args.kwargs["exception"], RuntimeError)
        assert mock_logger.opt.return_value.error.called
        assert not streams._background_tasks  # retired either way

    async def test_live_sync_delivers_chunks_without_waiting_on_the_log(self, tmp_path):
        import asyncio

        store = AsyncStore(str(tmp_path / "durable_streams"))
        sync = streams.StreamSync()
        released = asyncio.Event()

        async def sse_gen():
            yield b"data: first\n\n"
            await released.wait()  # generation stalls here
            yield b"data: [DONE]\n\n"

        with patch.object(streams, "clear_active_stream", new_callable=AsyncMock):
            writer = asyncio.create_task(
                streams.run_durable_stream(
                    sse_gen(), store, "msg-5", "thread-1", sync
                )
            )
            live = sync.stream()
            # The first chunk is available while the generator is still
            # stalled — no waiting for the stream to finish or for a poll.
            first = await asyncio.wait_for(live.__anext__(), timeout=1)
            assert first == b"data: first\n\n"

            released.set()
            rest = [chunk async for chunk in live]
            await writer

        assert rest == [b"data: [DONE]\n\n"]

    async def test_live_sync_detaches_when_nobody_drains_it(self):
        # A client that vanishes must not make the generating task block or
        # let the queue grow without bound.
        sync = streams.StreamSync()
        for _ in range(streams._LIVE_QUEUE_MAX + 50):
            sync.publish(b"data: x\n\n")  # must never raise or block
        assert sync._attached is False

    async def test_grace_period_lets_a_live_reader_finish_before_delete(
        self, tmp_path
    ):
        import asyncio

        store = AsyncStore(str(tmp_path / "durable_streams"))

        async def sse_gen():
            yield b"data: {\"type\": \"start\"}\n\n"
            await asyncio.sleep(0.06)  # let the reader's first poll land
            yield b"data: [DONE]\n\n"

        with patch.object(streams, "clear_active_stream", new_callable=AsyncMock):
            writer = asyncio.create_task(
                streams.run_durable_stream(sse_gen(), store, "msg-4", "thread-1")
            )
            await asyncio.sleep(0.01)  # let the writer create the stream first
            chunks = [chunk async for chunk in streams.tail_stream(store, "msg-4")]
            await writer

        assert chunks == [
            b"data: {\"type\": \"start\"}\n\n",
            b"data: [DONE]\n\n",
        ]
