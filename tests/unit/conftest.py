"""Fixtures shared by the contrib Ninja and DRF tests."""

import pytest


@pytest.fixture
def users(django_user_model):
    """A workflow manager, a stranger, and staff without the manage permission."""
    from tests.factories.db import MANAGE_WORKFLOWS, grant

    return (
        grant(django_user_model.objects.create(email="owner@example.com"), MANAGE_WORKFLOWS),
        django_user_model.objects.create(email="stranger@example.com"),
        django_user_model.objects.create(email="staff@example.com", is_staff=True),
    )
