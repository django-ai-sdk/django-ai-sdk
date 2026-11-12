"""The authoring tests run with a fresh loader and a code policy of their own to extend."""

from __future__ import annotations

import pytest

from django_ai_sdk.oversight.authoring.loader import loader
from django_ai_sdk.oversight.registry import register_policy
from tests.oversight.authoring.helpers import AUTHORING_POLICIES


@pytest.fixture(autouse=True)
def authoring_policies(registry_snapshot):
    for policy in AUTHORING_POLICIES:
        register_policy(policy)


@pytest.fixture(autouse=True)
def fresh_loader():
    loader.__init__()
    yield loader
    loader.__init__()
