from __future__ import annotations

from typing import Any

from django.db import models


class NaturalKeyManager(models.Manager):
    """Manager with ``get_by_natural_key`` driven by field names.

    Pass the model's natural-key fields to the constructor::

        objects = NaturalKeyManager("slug")

    or subclass and set them as a class attribute::

        class BySlugManager(NaturalKeyManager):
            natural_key_fields = ("slug",)
    """

    natural_key_fields: tuple[str, ...] = ()

    def __init__(self, *fields: str) -> None:
        super().__init__()
        if fields:
            self.natural_key_fields = tuple(fields)

    def get_by_natural_key(self, *values: Any) -> models.Model:
        if not self.natural_key_fields:
            raise TypeError(
                f"{type(self).__name__} has no natural-key fields configured: "
                "pass field names to __init__() or set natural_key_fields."
            )
        # zip(strict=True) turns an arity mismatch between fields and
        # values into a TypeError instead of silently truncating.
        lookup = dict(zip(self.natural_key_fields, values, strict=True))
        return self.get(**lookup)
