"""Replace WorkflowRun.input_messages with the `inputs` run-state bag.

Stored definitions move from a single `input_key` to the `requires` list in the
same pass, so no row is left naming a field the schema does not accept.
"""

from __future__ import annotations

from typing import Any

from django.db import migrations, models


def copy_messages_into_inputs(apps: Any, schema_editor: Any) -> None:
    """Keep the transcript of every run before the column holding it is dropped."""
    WorkflowRun = apps.get_model("django_ai_sdk", "WorkflowRun")
    for run in WorkflowRun.objects.all().iterator():
        if not run.input_messages:
            continue
        inputs = dict(run.inputs or {})
        if inputs.get("messages"):
            continue
        inputs["messages"] = list(run.input_messages)
        run.inputs = inputs
        run.save(update_fields=["inputs"])


def _rewrite_step(step: dict[str, Any]) -> dict[str, Any]:
    out = dict(step)
    legacy = out.pop("input_key", None)
    if not legacy:
        return out
    requires = list(out.get("requires") or [])
    if legacy not in requires:
        requires.append(legacy)
    out["requires"] = requires
    return out


def _rewrite_definition(definition: Any) -> tuple[Any, bool]:
    if not isinstance(definition, dict):
        return definition, False
    steps = definition.get("steps")
    if not isinstance(steps, list):
        return definition, False
    changed = False
    rewritten: list[Any] = []
    for step in steps:
        if isinstance(step, dict) and "input_key" in step:
            changed = True
            rewritten.append(_rewrite_step(step))
        else:
            rewritten.append(step)
    if not changed:
        return definition, False
    return {**definition, "steps": rewritten}, True


def rewrite_stored_input_keys(apps: Any, schema_editor: Any) -> None:
    """Rewrite `input_key` into `requires` in every stored definition and snapshot."""
    WorkflowSettings = apps.get_model("django_ai_sdk", "WorkflowSettings")
    for row in WorkflowSettings.objects.all().iterator():
        rewritten, changed = _rewrite_definition(row.definition)
        if changed:
            row.definition = rewritten
            row.save(update_fields=["definition"])

    WorkflowRun = apps.get_model("django_ai_sdk", "WorkflowRun")
    for run in WorkflowRun.objects.all().iterator():
        rewritten, changed = _rewrite_definition(run.workflow_definition)
        if changed:
            run.workflow_definition = rewritten
            run.save(update_fields=["workflow_definition"])


class Migration(migrations.Migration):
    dependencies = [
        ("django_ai_sdk", "0002_memory_metadata_storage"),
    ]

    operations = [
        migrations.AddField(
            model_name="workflowrun",
            name="inputs",
            field=models.JSONField(blank=True, default=dict),
        ),
        # Copy before the drop: reversing restores the column, never its contents.
        migrations.RunPython(copy_messages_into_inputs, migrations.RunPython.noop),
        migrations.RemoveField(
            model_name="workflowrun",
            name="input_messages",
        ),
        migrations.RunPython(rewrite_stored_input_keys, migrations.RunPython.noop),
    ]
