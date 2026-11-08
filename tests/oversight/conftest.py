"""Fixtures for the oversight tests: the test policies, a fresh resolution cache."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from django_ai_sdk.oversight.registry import get_policy_class, register_policy
from django_ai_sdk.oversight.resolve import invalidate
from tests.oversight.helpers import TEST_POLICIES


@pytest.fixture(autouse=True)
def test_policies():
    """Register the shared test policies (again, when a test reset the registry)."""
    for policy in TEST_POLICIES:
        if get_policy_class(policy.name) is not policy:
            register_policy(policy)


@pytest.fixture(autouse=True)
def _fresh_resolutions():
    invalidate()
    yield
    invalidate()


@pytest.fixture
def no_reflect_queue():
    """Close turns without queueing reflection."""
    with patch("django_ai_sdk.oversight.persist.enqueue_reflect"):
        yield


@pytest.fixture
def registry_snapshot():
    """Leave the policy registry as it was."""
    from django_ai_sdk.oversight import registry

    saved = (dict(registry._registry), dict(registry._invalid), dict(registry._duplicates))
    yield registry
    registry._registry.clear()
    registry._registry.update(saved[0])
    registry._invalid.clear()
    registry._invalid.update(saved[1])
    registry._duplicates.clear()
    registry._duplicates.update(saved[2])

