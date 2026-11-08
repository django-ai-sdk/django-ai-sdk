"""One data key per scope, wrapped by a master key."""

from __future__ import annotations

import uuid

from django.db import models

APP_LABEL = "django_ai_sdk_vault"


class DataKey(models.Model):
    """The data key for one scope, such as "thread:<id>" or "turn:<id>".

    Shredding clears `wrapped`, so the scope's data stops decrypting. Older backups
    still hold the wrapped key, and the master key can open it until they expire.
    """

    id = models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True)
    scope = models.CharField(max_length=255, unique=True)
    # Id of the master key that wrapped it (`keys.key_id`). Rotation updates it.
    key_id = models.CharField(max_length=255, blank=True, db_index=True)
    wrapped = models.BinaryField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    rewrapped_at = models.DateTimeField(null=True, blank=True)
    shredded_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        app_label = APP_LABEL
        db_table = "django_ai_sdk_data_keys"

    def __str__(self) -> str:
        return self.scope

    @property
    def shredded(self) -> bool:
        # `shred` is the only writer, and it sets this and clears `wrapped` together.
        return self.shredded_at is not None


__all__ = ["DataKey"]
