"""ThreadMessageAction, the built-in that puts a finished run in someone's chat.

Only the title-generating model call is faked; it is the one step that leaves the process.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from django_ai_sdk.conversation.models import Message, Thread
from django_ai_sdk.permissions import AllowAll
from django_ai_sdk.storage.db import DbStorageAdapter
from django_ai_sdk.workflows.actions import ActionContext, ThreadMessageAction


async def make_user():
    from tests.factories.db import UserFactory

    return await UserFactory.acreate()


def make_agent(title="Ship sighting report", title_generation=True):
    agent = MagicMock()
    agent.id = "log-keeper"
    agent.name = "Log keeper"
    agent.model = "gpt-4o-mini"
    agent.permissions = [AllowAll]
    agent.storage_adapter = DbStorageAdapter
    agent.title_generation = title_generation
    agent.run = AsyncMock(return_value=title)
    agent.get_title_generation_prompt = MagicMock(return_value="Give this a short title.")
    return agent


def serving(agent):
    return patch(
        "django_ai_sdk.agents.services.AgentService.get",
        AsyncMock(return_value=agent),
    )


@pytest.mark.django_db(transaction=True)
class TestDelivery:
    async def test_the_payload_arrives_as_a_message_in_a_thread_the_user_owns(self):
        user = await make_user()
        context = ActionContext(user=user, agent_id="log-keeper", source="ships-log")

        with serving(make_agent()):
            await ThreadMessageAction().execute("Two ships sighted, no losses.", context)

        thread = await Thread.objects.filter(user=user).afirst()
        assert thread is not None
        message = await Message.objects.filter(thread=thread).afirst()
        assert "Two ships sighted" in str(message.result)

    async def test_a_structured_payload_arrives_as_json_not_a_python_repr(self):
        user = await make_user()
        context = ActionContext(user=user, agent_id="log-keeper", source="ships-log")

        with serving(make_agent()):
            await ThreadMessageAction().execute({"sighted": 2, "lost": 0}, context)

        thread = await Thread.objects.filter(user=user).afirst()
        message = await Message.objects.filter(thread=thread).afirst()
        assert '"sighted": 2' in str(message.result)

    async def test_the_thread_gets_a_generated_title_not_the_bare_source(self):
        # A scheduled workflow would otherwise fill the thread list with one title.
        user = await make_user()
        context = ActionContext(user=user, agent_id="log-keeper", source="ships-log")

        with serving(make_agent(title="Ship sighting report")):
            await ThreadMessageAction().execute("Two ships sighted.", context)

        thread = await Thread.objects.filter(user=user).afirst()
        assert thread.title == "Ship sighting report"

    async def test_an_agent_with_titles_off_keeps_the_source_as_the_title(self):
        user = await make_user()
        context = ActionContext(user=user, agent_id="log-keeper", source="ships-log")

        with serving(make_agent(title_generation=False)):
            await ThreadMessageAction().execute("Two ships sighted.", context)

        thread = await Thread.objects.filter(user=user).afirst()
        assert thread.title == "ships-log"


@pytest.mark.django_db(transaction=True)
class TestNothingToDeliverTo:
    async def test_a_run_without_a_user_warns_and_creates_no_thread(self, caplog):
        context = ActionContext(agent_id="log-keeper", source="ships-log")

        with caplog.at_level("WARNING", logger="django_ai_sdk.workflows.actions"):
            with serving(make_agent()):
                await ThreadMessageAction().execute("Two ships sighted.", context)

        assert not await Thread.objects.aexists()
        assert "ships-log" in caplog.text

    async def test_a_run_without_an_agent_warns_and_creates_no_thread(self, caplog):
        user = await make_user()
        context = ActionContext(user=user, source="ships-log")

        with caplog.at_level("WARNING", logger="django_ai_sdk.workflows.actions"):
            await ThreadMessageAction().execute("Two ships sighted.", context)

        assert not await Thread.objects.aexists()
        assert "ships-log" in caplog.text
