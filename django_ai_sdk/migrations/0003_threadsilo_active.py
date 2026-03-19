# Generated manually - Add active field to ThreadSilo model

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("django_ai_sdk", "0002_remove_thread_assistant_id_remove_thread_model"),
    ]

    operations = [
        migrations.AddField(
            model_name="threadsilo",
            name="active",
            field=models.BooleanField(default=True),
        ),
    ]
