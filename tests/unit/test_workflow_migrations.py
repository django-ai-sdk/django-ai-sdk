"""The workflows app's first migration in a project whose core app already made its tables."""

import importlib

import pytest
from django.db import connection
from django.db.migrations.loader import MigrationLoader

APP = "django_ai_sdk_workflows"


# transaction=True: SQLite's schema editor refuses to start inside an atomic block.
@pytest.mark.django_db(transaction=True)
def test_turning_the_app_on_later_keeps_the_tables_the_core_app_made():
    """Up to 0.1.2a21 the core app created these tables with or without this app."""
    initial = importlib.import_module("django_ai_sdk.workflows.migrations.0001_initial")
    loader = MigrationLoader(connection)
    before = loader.project_state(("django_ai_sdk", "0001_initial"))
    after = loader.project_state((APP, "0001_initial"))
    assert "django_ai_sdk_workflow_runs" in connection.introspection.table_names()
    with connection.schema_editor() as editor:
        for operation in initial.Migration.operations:
            # A plain CreateModel would fail here: the table exists.
            operation.database_forwards(APP, editor, before, after)
