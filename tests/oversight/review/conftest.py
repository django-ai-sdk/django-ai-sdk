from __future__ import annotations

import pytest

from django_ai_sdk.permissions import get_domain_permissions


@pytest.fixture(autouse=True)
def _fresh_permissions():
    get_domain_permissions.cache_clear()
    yield
    get_domain_permissions.cache_clear()
