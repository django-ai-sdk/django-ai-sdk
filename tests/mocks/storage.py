from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch


@contextmanager
def mock_get_storage(method="rate_message", return_value=True):
    mock_storage = MagicMock()
    setattr(mock_storage, method, AsyncMock(return_value=return_value))
    with patch(
        "django_ai_sdk.storage.services._get_storage",
        new_callable=AsyncMock,
    ) as m:
        m.return_value = mock_storage
        yield mock_storage


def setup_thread_adapter(registry, user_id="user-1", agent_id="test-agent"):
    from tests.mocks.agent import create_mock_adapter_class
    from tests.factories.schemas import ThreadInfoFactory

    thread_info = ThreadInfoFactory.build(
        agent_id=agent_id, user_id=user_id
    )
    adapter_cls = create_mock_adapter_class(get_thread=thread_info)
    registry.get_all_adapters.return_value = [adapter_cls]
    return thread_info, adapter_cls


@contextmanager
def mock_thread_model(aexists=True, thread_db=None):
    with patch("django_ai_sdk.conversation.models.Thread") as mock_thread:
        mock_thread.objects.filter.return_value.aexists = AsyncMock(
            return_value=aexists
        )
        if thread_db is not None:
            mock_thread.objects.select_related.return_value.aget = AsyncMock(
                return_value=thread_db
            )
        yield mock_thread
