"""Mark runs whose worker stopped beating, or that were never queued, as interrupted.

See "A worker that dies" in the workflows manual.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from django.core.management.base import CommandError
from django.db.models import Case, Q, Value, When
from django.db.models.functions import Now

from django_ai_sdk.management.sweep import SweepCommand, parse_age
from django_ai_sdk.workflows.runs import HEARTBEAT_SECONDS

if TYPE_CHECKING:
    from datetime import datetime

    from django.db.models import QuerySet

INTERRUPTED = "Interrupted: its worker stopped sending heartbeats."
NEVER_QUEUED = "Interrupted: it was never queued for a worker."


class Command(SweepCommand):
    help = (
        "Mark workflow runs with no heartbeat, or never queued, for --stale-after as "
        "interrupted. A worker beats every minute, also while a step runs."
    )
    noun = "run"
    done = "interrupted"
    default_stale_after = "30m"
    marked: ClassVar[dict[str, Any]] = {
        "status": "interrupted",
        "error": Case(When(status="pending", then=Value(NEVER_QUEUED)), default=Value(INTERRUPTED)),
        "completed_at": Now(),
    }

    def handle(self, *args: Any, **options: Any) -> None:
        # Under two beats, a healthy run between beats counts as stale and runs twice.
        try:
            too_soon = parse_age(options["stale_after"]).total_seconds() < 2 * HEARTBEAT_SECONDS
        except ValueError:
            too_soon = False  # The base command reports the bad age.
        if too_soon:
            raise CommandError(f"--stale-after must be at least {2 * HEARTBEAT_SECONDS}s.")
        super().handle(*args, **options)

    def stale(self, cutoff: datetime) -> QuerySet[Any]:
        from django_ai_sdk.workflows.models import WorkflowRun

        # A claim sets heartbeat_at, so a running run always has one.
        running = Q(status=WorkflowRun.Status.RUNNING, heartbeat_at__lt=cutoff)
        # A run with a task_id reached the queue, which owns delivery. A long backlog
        # is not a crash.
        never_queued = Q(
            status=WorkflowRun.Status.PENDING, task_id__isnull=True, updated_at__lt=cutoff
        )
        return WorkflowRun.objects.filter(running | never_queued)

    def act(self, stale: QuerySet[Any], *, limit: int, dry_run: bool) -> int:
        from django_ai_sdk.workflows.models import StepRun, WorkflowRun

        keys = list(stale.order_by(self.order_by).values_list("pk", flat=True)[:limit])
        if dry_run or not keys:
            return len(keys)
        # The update re-checks staleness, so a run that moved on is left alone.
        count = stale.filter(pk__in=keys).update(**self.marked)
        # The step each swept run was in when its worker died; no other run's rows.
        StepRun.objects.filter(
            run_id__in=keys,
            run__status=WorkflowRun.Status.INTERRUPTED,
            status=StepRun.Status.RUNNING,
        ).update(
            status=StepRun.Status.FAILED,
            error=INTERRUPTED,
            detail="interrupted",
            completed_at=Now(),
        )
        return count
