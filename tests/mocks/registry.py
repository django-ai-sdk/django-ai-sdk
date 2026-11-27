from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from tests.mocks.agent import create_agent_mock


@contextmanager
def patch_registry(agent=None, agents=None, permissions=None):
    if agent is None:
        agent = create_agent_mock(permissions=permissions)
    if agents is None:
        agents = {agent.id: agent}

    with patch("django_ai_sdk.agents.registry.registry") as reg, \
         patch("django_ai_sdk.agents.services.registry", reg):
        reg.get = MagicMock(return_value=agent)
        reg.all = MagicMock(return_value=agents)
        yield reg
