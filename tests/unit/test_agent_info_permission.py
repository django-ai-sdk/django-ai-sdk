"""Reading one agent must agree with listing them."""

import uuid

import pytest

from django_ai_sdk.agents.models import AgentSettings, AgentUser
from django_ai_sdk.agents.services import AgentService
from django_ai_sdk.permissions import PermissionDenied


async def make_user(name):
    from django.contrib.auth import get_user_model

    return await get_user_model().objects.acreate(email=f"{name}-{uuid.uuid4().hex[:8]}@example.com")


async def make_agent(name="Secret", *, is_public=False):
    return await AgentSettings.objects.acreate(name=name, model="gpt-4o-mini", is_public=is_public)


@pytest.mark.django_db(transaction=True)
class TestGetAgentInfo:
    async def test_a_non_member_cannot_read_a_private_agent(self):
        outsider = await make_user("outsider")
        config = await make_agent()

        with pytest.raises(PermissionDenied):
            await AgentService.get_agent_info(str(config.id), user=outsider)

    async def test_a_member_can_read_it(self):
        member = await make_user("member")
        config = await make_agent()
        await AgentUser.objects.acreate(agent=config, user=member, can_manage=False)

        info = await AgentService.get_agent_info(str(config.id), user=member)
        assert info.name == "Secret"

    async def test_a_public_agent_stays_readable_by_anyone_authenticated(self):
        outsider = await make_user("outsider")
        config = await make_agent("Open", is_public=True)

        info = await AgentService.get_agent_info(str(config.id), user=outsider)
        assert info.name == "Open"

    async def test_reading_by_id_agrees_with_what_listing_shows(self):
        outsider = await make_user("outsider")
        config = await make_agent()

        listed = await AgentService.list_agents(user=outsider)
        assert str(config.id) not in {row["id"] for row in listed}
        with pytest.raises(PermissionDenied):
            await AgentService.get_agent_info(str(config.id), user=outsider)


@pytest.mark.django_db(transaction=True)
class TestGetIntegrationStatus:
    async def test_a_non_member_cannot_read_a_private_agents_integrations(self):
        outsider = await make_user("outsider")
        config = await make_agent()
        agent = await AgentService.get(str(config.id))

        with pytest.raises(PermissionDenied):
            await AgentService.get_integration_status(agent, user=outsider)
