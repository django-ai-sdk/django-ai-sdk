import pytest


@pytest.fixture
def users(django_user_model):
    from tests.factories.db import MANAGE_WORKFLOWS, grant

    return (
        grant(django_user_model.objects.create(email="owner@example.com"), MANAGE_WORKFLOWS),
        django_user_model.objects.create(email="stranger@example.com"),
        django_user_model.objects.create(email="staff@example.com", is_staff=True),
    )
