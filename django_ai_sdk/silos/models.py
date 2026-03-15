import uuid
from typing import Any

from django.db import models
from django.utils import timezone

from django_ai_sdk.silos.schemas import DocumentExtraction


class Silo(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    # Reverse relation type hint
    documents: models.Manager["Document"]
    # Annotated field from queries
    document_count: int

    class Meta:
        db_table = "django_ai_sdk_silos"

    def __str__(self) -> str:
        return self.name


class Document(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    silo = models.ForeignKey(Silo, on_delete=models.CASCADE, related_name="documents")
    file = models.FileField(upload_to="silos/documents/")
    content = models.TextField(blank=True, default="")
    data = models.JSONField(default=dict, blank=True)  # stores DocumentExtraction as dict
    file_name = models.CharField(max_length=255, blank=True, default="")
    file_size = models.PositiveIntegerField(default=0)
    content_type = models.CharField(max_length=100, blank=True, default="")
    file_extension = models.CharField(max_length=20, blank=True, default="")
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "django_ai_sdk_documents"

    def __str__(self) -> str:
        return self.file_name or (self.file.name if self.file else f"Document {self.id}")

    @property
    def extraction(self) -> DocumentExtraction | None:
        """Parse data JSON to DocumentExtraction Pydantic model."""
        if not self.data:
            return None
        try:
            return DocumentExtraction(**self.data)
        except Exception:
            return None

    @extraction.setter
    def extraction(self, value: DocumentExtraction | None) -> None:
        """Set data from DocumentExtraction Pydantic model."""
        if value is None:
            self.data = {}
        else:
            self.data = value.model_dump()

    def save(self, *args: Any, **kwargs: Any) -> None:
        if self.file:
            # TODO: this sucks donkey balls, but we need to move
            # to a proper mime type detection and file metadata extraction
            if not self.file_name:
                self.file_name = self.file.name
            if not self.file_size:
                self.file_size = self.file.size or 0
            if not self.content_type:
                if hasattr(self.file, "content_type"):
                    self.content_type = self.file.content_type or ""
                elif hasattr(self.file, "content_type") and self.file.content_type:
                    self.content_type = self.file.content_type
                else:
                    self.content_type = ""
            if not self.file_extension:
                import os

                _, ext = os.path.splitext(self.file.name)
                self.file_extension = ext.lower().lstrip(".")
        super().save(*args, **kwargs)


class ThreadSilo(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    thread = models.ForeignKey(
        "django_ai_sdk.Thread",
        on_delete=models.CASCADE,
        related_name="silo_links",
    )
    silo = models.ForeignKey(Silo, on_delete=models.CASCADE, related_name="thread_links")
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        db_table = "django_ai_sdk_thread_silos"
        unique_together = [["thread", "silo"]]

    def __str__(self) -> str:
        return f"{self.thread} - {self.silo}"
