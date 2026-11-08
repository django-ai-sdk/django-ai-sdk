"""Fixtures shared by the contrib Ninja and DRF tests."""

import pytest


@pytest.fixture
def users(django_user_model):
    """An owner, a stranger, and staff."""
    return (
        django_user_model.objects.create(email="owner@example.com"),
        django_user_model.objects.create(email="stranger@example.com"),
        django_user_model.objects.create(email="staff@example.com", is_staff=True),
    )
