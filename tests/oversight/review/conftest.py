"""Fixtures for the review tests: a fresh cache of the oversight permission classes."""

from __future__ import annotations

import pytest

from django_ai_sdk.permissions import get_domain_permissions


@pytest.fixture(autouse=True)
def _fresh_permissions():
    """Tests swap AI_SDK_PERMISSIONS; the classes it names are cached per domain."""
    get_domain_permissions.cache_clear()
    yield
    get_domain_permissions.cache_clear()
