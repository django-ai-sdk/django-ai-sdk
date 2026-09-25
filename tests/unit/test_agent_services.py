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


@pytest.mark.django_db
@pytest.mark.asyncio
async def test_a_db_agent_runs_under_the_same_limits_as_a_code_agent(settings):
    from unittest.mock import AsyncMock, patch

    from django_ai_sdk.agents import runtime
    from django_ai_sdk.agents.models import AgentSettings
    from django_ai_sdk.agents.tool_agent import ToolCallBudgetHook

    settings.OPENAI_API_KEY = "sk-test"
    config = await AgentSettings.objects.acreate(name="DB", slug=str(uuid4()), agent="test")
    agent = runtime.RuntimeAgent(config)

    with (
        patch.object(runtime.RuntimeAgent, "get_tools", AsyncMock(return_value=[])),
        patch.object(runtime, "ToolAgent", wraps=runtime.ToolAgent) as built,
    ):
        await agent.get_pipeline_adapter(thread_id=None)

    tool_config = built.call_args.kwargs["config"]
    assert tool_config.max_agent_steps == agent.max_agent_steps
    assert any(isinstance(h, ToolCallBudgetHook) for h in tool_config.hooks["before_tool"])
