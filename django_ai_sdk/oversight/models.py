"""The oversight record: turns, lens answers, rule matches and reviews.

Anything that can quote the turn lives in `TurnContent` (and a reviewer's note),
so it can be forgotten while the record, holding only values, labels, scores,
counts and errors, stays.
"""

from __future__ import annotations

import uuid

from django.db import models

from django_ai_sdk.oversight.conf import APP_LABEL
from django_ai_sdk.oversight.types import Outcome

OUTCOME_CHOICES = [(str(o), str(o).capitalize()) for o in Outcome]


class TurnContent(models.Model):
    """A turn's content at the policy's capture level, plus `texts` per lens answer id."""

    id = models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True)
    capture = models.CharField(max_length=16)
    stored = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField(null=True, blank=True, db_index=True)

    class Meta:
        app_label = APP_LABEL
        db_table = "django_ai_sdk_oversight_content"


class Turn(models.Model):
    """One question and its answer, on any channel (chat, workflow, API)."""

    class Status(models.TextChoices):
        # Written when the gate ends, before the model runs, so a crash still leaves a row.
        OPEN = "open", "Open"
        CLOSED = "closed", "Closed"
        # Set by `oversight_sweep` on turns that never closed, e.g. after a killed worker.
        INTERRUPTED = "interrupted", "Interrupted"

    class ReflectState(models.TextChoices):
        NONE = "", "None"
        PENDING = "pending", "Pending"
        # Claimed by a worker until `reflect_lease_until`. After that another may claim it.
        RUNNING = "running", "Running"
        DONE = "done", "Done"
        # Gave up after `reflection.REFLECT_ATTEMPTS` tries.
        FAILED = "failed", "Failed"

    id = models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True)
    created_at = models.DateTimeField(db_index=True)
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.OPEN)

    channel = models.CharField(max_length=100, db_index=True)
    agent_id = models.CharField(max_length=64)
    agent_name = models.CharField(max_length=255, blank=True)
    # Lines up quality changes with agent changes.
    agent_fingerprint = models.CharField(max_length=16)
    policy = models.CharField(max_length=100)

    # Ids only, no foreign keys: deleting a thread must not delete its audit trail.
    thread_id = models.CharField(max_length=64, blank=True, db_index=True)
    user_message_id = models.CharField(max_length=64, blank=True)
    message_id = models.CharField(max_length=64, blank=True)
    parent_turn_id = models.UUIDField(null=True, blank=True, db_index=True)

    # "" when anonymous or once the user is deleted.
    actor_id = models.CharField(max_length=64, blank=True, db_index=True)

    model = models.CharField(max_length=255, blank=True)
    prompt_tokens = models.IntegerField(null=True, blank=True)
    completion_tokens = models.IntegerField(null=True, blank=True)
    cost = models.FloatField(null=True, blank=True)
    # The agent's latency. Oversight's own cost is on its lens answers.
    latency_ms = models.FloatField(null=True, blank=True)

    # Blank while open, and after an error with no rule acting: nothing was let through.
    outcome = models.CharField(max_length=16, choices=OUTCOME_CHOICES, blank=True)
    report_outcome = models.CharField(
        max_length=16, choices=OUTCOME_CHOICES, blank=True, db_index=True
    )
    # Never contains content.
    error = models.CharField(max_length=500, blank=True)

    content = models.ForeignKey(TurnContent, null=True, blank=True, on_delete=models.SET_NULL)
    content_id: uuid.UUID | None
    # Set by forgetting while the turn was open, so its close stores no content.
    content_forgotten = models.BooleanField(default=False)
    expires_at = models.DateTimeField(null=True, blank=True, db_index=True)

    # Denormalized, updated as later stages answer.
    labels = models.JSONField(default=dict, blank=True)
    severity = models.CharField(max_length=16, blank=True, db_index=True)
    reflect_state = models.CharField(
        max_length=16, choices=ReflectState.choices, blank=True, default=ReflectState.NONE
    )
    reflect_attempts = models.PositiveSmallIntegerField(default=0)
    reflect_lease_until = models.DateTimeField(null=True, blank=True)
    reflected_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        app_label = APP_LABEL
        db_table = "django_ai_sdk_oversight_turns"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["agent_id", "created_at"]),
            models.Index(fields=["policy", "created_at"]),
            models.Index(fields=["outcome", "created_at"]),
            models.Index(fields=["status", "created_at"]),
            models.Index(fields=["reflect_state", "created_at"]),
        ]

    def __str__(self) -> str:
        return f"{self.channel} {self.outcome} {self.created_at:%Y-%m-%d %H:%M}"

    @property
    def tokens(self) -> int | None:
        if self.prompt_tokens is None and self.completion_tokens is None:
            return None
        return (self.prompt_tokens or 0) + (self.completion_tokens or 0)


class LensAnswer(models.Model):
    """One answer to one lens at one stage. Holds no text that can quote the turn."""

    id = models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True)
    turn = models.ForeignKey(Turn, on_delete=models.CASCADE, related_name="answers", db_index=False)
    turn_id: uuid.UUID
    lens = models.CharField(max_length=100, db_index=True)
    stage = models.CharField(max_length=16)
    answerer = models.CharField(max_length=200)
    model = models.CharField(max_length=255, blank=True)
    value = models.JSONField(null=True, blank=True)
    confidence = models.FloatField(null=True, blank=True)
    spans = models.JSONField(default=list, blank=True)
    trail = models.JSONField(default=list, blank=True)
    error = models.CharField(max_length=200, blank=True)
    cost = models.FloatField(null=True, blank=True)
    latency_ms = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        app_label = APP_LABEL
        db_table = "django_ai_sdk_oversight_answers"
        ordering = ["created_at"]
        indexes = [models.Index(fields=["turn", "lens"])]


class RuleMatch(models.Model):
    id = models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True)
    turn = models.ForeignKey(Turn, on_delete=models.CASCADE, related_name="matches")
    turn_id: uuid.UUID
    rule = models.CharField(max_length=100, db_index=True)
    stage = models.CharField(max_length=16)
    mode = models.CharField(max_length=16)
    actions = models.JSONField(default=list)
    # False in report mode, or when the stage was too late for the action.
    acted = models.BooleanField(default=False)
    condition = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        app_label = APP_LABEL
        db_table = "django_ai_sdk_oversight_matches"
        ordering = ["created_at"]
        constraints = [
            # Reflect matches a rule once per turn, even when retried. Inline stages may
            # match more than once (history and question, or each tool call).
            models.UniqueConstraint(
                fields=["turn", "rule"],
                condition=models.Q(stage="reflect"),
                name="oversight_one_reflect_match_per_rule",
            )
        ]


class ReviewItem(models.Model):
    """A turn a rule queued for a person to look at."""

    class Status(models.TextChoices):
        OPEN = "open", "Open"
        DONE = "done", "Done"
        DISMISSED = "dismissed", "Dismissed"

    id = models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True)
    turn = models.ForeignKey(Turn, on_delete=models.CASCADE, related_name="reviews", db_index=False)
    turn_id: uuid.UUID
    queue = models.CharField(max_length=100, db_index=True, default="default")
    rule = models.CharField(max_length=100, blank=True)
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.OPEN)
    created_at = models.DateTimeField(auto_now_add=True)
    resolved_at = models.DateTimeField(null=True, blank=True)
    # A plain id, not a foreign key, so deleting a user keeps the review.
    resolved_by_id = models.CharField(max_length=64, blank=True)
    # None when dismissed or open.
    agree = models.BooleanField(null=True, blank=True)
    # Content: forgetting the turn's thread blanks it.
    note = models.TextField(blank=True)

    class Meta:
        app_label = APP_LABEL
        db_table = "django_ai_sdk_oversight_reviews"
        ordering = ["created_at"]
        indexes = [models.Index(fields=["status", "queue", "created_at"])]
        constraints = [
            models.UniqueConstraint(
                fields=["turn", "queue", "rule"],
                name="oversight_one_review_per_question",
            )
        ]
