from __future__ import annotations

from django.db import migrations


class Migration(migrations.Migration):
    """
    Kept for its place in the graph. It used to hand the workflow models over to
    `django_ai_sdk.workflows`; the workflows app now creates its tables itself, so
    a project without that app has none. A database that ran the old version keeps
    its tables, which the workflows app's migrations go on from.
    """

    dependencies = [
        ("django_ai_sdk", "0002_memory_metadata_storage"),
    ]

    operations = []
