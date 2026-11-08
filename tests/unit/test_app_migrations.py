"""The apps that ship with 0.2.0 start from one migration each: no history to replay."""

from __future__ import annotations

import pytest
from django.apps import apps
from django.db.migrations.loader import MigrationLoader

ONE_MIGRATION = [
    "django_ai_sdk_workflows",
    "django_ai_sdk_oversight",
    "django_ai_sdk_oversight_switches",
    "django_ai_sdk_vault",
    "django_ai_sdk_oversight_ledger",
    "django_ai_sdk_oversight_authoring",
]


@pytest.mark.parametrize("label", ONE_MIGRATION)
def test_one_initial_migration(label):
    if not any(config.label == label for config in apps.get_app_configs()):
        pytest.skip(f"{label} is not installed")
    loader = MigrationLoader(None, ignore_no_migrations=True)
    assert sorted(name for app, name in loader.disk_migrations if app == label) == [
        "0001_initial"
    ]


def test_the_opt_in_apps_depend_on_no_other_migration():
    """Each can be installed, or removed, without another app's migrations moving (the user
    model aside)."""
    from django.conf import settings

    users = settings.AUTH_USER_MODEL.split(".")[0]
    loader = MigrationLoader(None, ignore_no_migrations=True)
    for label in ONE_MIGRATION:
        migration = loader.disk_migrations.get((label, "0001_initial"))
        if migration is None:
            continue
        others = [
            dep
            for dep in migration.dependencies
            if dep[0] not in {"__setting__", label, users}
        ]
        assert others == [], label


def _config(label):
    return next(config for config in apps.get_app_configs() if config.label == label)


@pytest.mark.django_db(transaction=True)
@pytest.mark.parametrize(
    ("label", "table"),
    [
        ("django_ai_sdk_oversight_ledger", "django_ai_sdk_oversight_ledger"),
        ("django_ai_sdk_vault", "django_ai_sdk_data_keys"),
        ("django_ai_sdk_oversight_switches", "django_ai_sdk_oversight_switches"),
        ("django_ai_sdk_oversight_authoring", "django_ai_sdk_oversight_authored_policies"),
    ],
)
def test_an_opt_in_app_migrates_to_zero_and_back(label, table):
    """How the docs say to remove one: its tables go, and nothing else's."""
    from django.core.management import call_command
    from django.db import connection

    if not any(config.label == label for config in apps.get_app_configs()):
        pytest.skip(f"{label} is not installed")
    before = set(connection.introspection.table_names())
    call_command("migrate", label, "zero", verbosity=0)
    after = set(connection.introspection.table_names())
    owned = {model._meta.db_table for model in _config(label).get_models()}
    assert table in owned
    assert before - after == owned
    call_command("migrate", label, verbosity=0)
    assert set(connection.introspection.table_names()) == before
    if label == "django_ai_sdk_oversight_ledger":
        from django_ai_sdk.oversight.ledger.models import LedgerHead

        assert LedgerHead.objects.filter(id=1, seq=0).exists()
