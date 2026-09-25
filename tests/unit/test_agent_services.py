"""AgentService writes to AgentSettings: only the fields it declares, whatever it is handed."""

from uuid import uuid4

import pytest


@pytest.mark.django_db
@pytest.mark.asyncio
class TestUpdateRuntimeAgentFields:
    async def _setup(self):
        from django_ai_sdk.agents.models import AgentSettings, AgentUser
        from tests.factories.db import UserFactory

        admin = await UserFactory.acreate()
        config = await AgentSettings.objects.acreate(
            name="Support", slug=str(uuid4()), agent="test", is_public=False
        )
        await AgentUser.objects.acreate(agent=config, user=admin, can_manage=True)
        return admin, config

    async def test_a_declared_field_is_updated(self):
        from django_ai_sdk.agents.services import AgentService

        admin, config = await self._setup()

        await AgentService.update_runtime_agent(str(config.id), {"name": "Helpdesk"}, user=admin)

        await config.arefresh_from_db()
        assert config.name == "Helpdesk"

    async def test_an_undeclared_field_is_refused(self):
        from django_ai_sdk.agents.services import AgentService

        admin, config = await self._setup()

        with pytest.raises(TypeError, match="is_public"):
            await AgentService.update_runtime_agent(
                str(config.id), {"name": "Helpdesk", "is_public": True}, user=admin
            )

        await config.arefresh_from_db()
        assert (config.name, config.is_public) == ("Support", False)
