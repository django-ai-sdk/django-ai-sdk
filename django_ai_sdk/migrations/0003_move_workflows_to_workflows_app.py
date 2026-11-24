from __future__ import annotations

from django.db import migrations


class Migration(migrations.Migration):
    """Empty, kept for the graph: the workflows app now creates its own tables."""

    dependencies = [
        ("django_ai_sdk", "0002_memory_metadata_storage"),
    ]

    operations = []
