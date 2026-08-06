"""
Registry patching helpers.

The ``AgentRegistry`` is a singleton with auto-registration side effects.
These helpers patch it at both import paths simultaneously to avoid that.
"""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from tests.mocks.agent import create_agent_mock


@contextmanager
def patch_registry(agent=None, agents=None, permissions=None):
    """Context manager that patches the global agent registry.

    Both import paths (``django_ai_sdk.agents.registry.registry`` and
    ``django_ai_sdk.agents.services.registry``) are patched to the same
    mock, ensuring consistency.

    Yields the mock registry so callers can reconfigure it:
        with patch_registry() as reg:
            reg.get.return_value = None   # simulate "not found"

    Parameters:
        agent:   The mock agent to return from ``reg.get()``.
                     Default: ``create_agent_mock()``
        agents:  Dict mapping id -> agent for ``reg.all()``.
                     Default: ``{agent.id: agent}``
        permissions: Passed to ``create_agent_mock()`` when *agent*
                     is not provided.
    """
    if agent is None:
        agent = create_agent_mock(permissions=permissions)
    if agents is None:
        agents = {agent.id: agent}

    with patch("django_ai_sdk.agents.registry.registry") as reg, \
         patch("django_ai_sdk.agents.services.registry", reg):
        reg.get = MagicMock(return_value=agent)
        reg.all = MagicMock(return_value=agents)
        yield reg
