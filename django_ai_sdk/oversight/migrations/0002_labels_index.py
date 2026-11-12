from __future__ import annotations

from typing import Any

from django.db import migrations

# The turns feed filters on a label (`labels ? 'topic'`, `labels @> '{"topic": "hr"}'`);
# on Postgres a GIN index answers both. Other databases scan the window.
LABELS_INDEX = "oversight_turn_labels_gin"


def add_labels_index(apps: Any, schema_editor: Any) -> None:
    if schema_editor.connection.vendor == "postgresql":
        schema_editor.execute(
            f"CREATE INDEX IF NOT EXISTS {LABELS_INDEX} "
            "ON django_ai_sdk_oversight_turns USING gin (labels)"
        )


def drop_labels_index(apps: Any, schema_editor: Any) -> None:
    if schema_editor.connection.vendor == "postgresql":
        schema_editor.execute(f"DROP INDEX IF EXISTS {LABELS_INDEX}")


class Migration(migrations.Migration):
    dependencies = [("django_ai_sdk_oversight", "0001_initial")]

    operations = [
        migrations.RunPython(add_labels_index, drop_labels_index, elidable=False),
    ]
