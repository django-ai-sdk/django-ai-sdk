"""Durable workflows and the task queue on DBOS, in the project's own database.

Opt-in: add "django_ai_sdk.durable" to INSTALLED_APPS (needs the `dbos` extra) and
run `manage.py ai_sdk_worker`. Workflow runs then survive a dead worker, approval
steps wait for a decision across restarts, cron triggers fire once per occurrence,
and `DBOSTaskBackend` runs every other django-tasks task on the same queues.
"""
