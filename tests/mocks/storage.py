"""
Storage-related mock helpers.

Mock patterns commonly needed when testing storage services:
  - ``mock_get_storage()`` — patch ``_get_storage`` with a mock adapter
  - ``setup_thread_adapter()`` — register a mock adapter with a ThreadInfo return
  - ``mock_thread_model()`` — patch ``conversation.models.Thread``
"""

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch


@contextmanager
def mock_get_storage(method="rate_message", return_value=True):
    """Context manager that patches ``_get_storage`` with a controlled mock.

    Usage::

        with mock_get_storage("rate_message", return_value=True) as storage:
            # storage.rate_message is an AsyncMock returning True
    """
    mock_storage = MagicMock()
    setattr(mock_storage, method, AsyncMock(return_value=return_value))
    with patch(
        "django_ai_sdk.storage.services._get_storage",
        new_callable=AsyncMock,
    ) as m:
        m.return_value = mock_storage
        yield mock_storage


def setup_thread_adapter(registry, user_id="user-1", assistant_id="test-assistant"):
    """Build a ThreadInfo + adapter class and register it on the storage registry.

    Returns ``(thread_info, adapter_cls)`` for further configuration.
    """
    from tests.mocks.assistant import create_mock_adapter_class
    from tests.factories.schemas import ThreadInfoFactory

    thread_info = ThreadInfoFactory.build(
        assistant_id=assistant_id, user_id=user_id
    )
    adapter_cls = create_mock_adapter_class(get_thread=thread_info)
    registry.get_all_adapters.return_value = [adapter_cls]
    return thread_info, adapter_cls


@contextmanager
def mock_thread_model(aexists=True, thread_db=None):
    """Context manager that patches ``conversation.models.Thread``.

    Sets up ``objects.filter.return_value.aexists`` and (optionally)
    ``objects.select_related.return_value.aget``.

    Usage::

        with mock_thread_model(aexists=True, thread_db=my_thread_db) as mock_thread:
            result = await aget_thread_file_meta("thread-1", user=None)
    """
    with patch("django_ai_sdk.conversation.models.Thread") as mock_thread:
        mock_thread.objects.filter.return_value.aexists = AsyncMock(
            return_value=aexists
        )
        if thread_db is not None:
            mock_thread.objects.select_related.return_value.aget = AsyncMock(
                return_value=thread_db
            )
        yield mock_thread
