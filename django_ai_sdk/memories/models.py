from __future__ import annotations

import re
import uuid
from typing import Any

from django.conf import settings
from django.db import models
from django.utils import timezone
from django.utils.text import slugify

from django_ai_sdk.memories.schemas import DocumentExtraction
from django_ai_sdk.rags.schemas import RagDocument, ToolSpec


class Memory(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    slug = models.SlugField(max_length=255, unique=True, blank=True, editable=False)
    description = models.TextField(blank=True, default="")
    is_hidden = models.BooleanField(default=False)
    is_public = models.BooleanField(default=True)
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    # Reverse relation type hints
    entries: models.Manager[Entry]
    memory_users: models.Manager[MemoryUser]
    memory_groups: models.Manager[MemoryGroup]

    # Annotated field from queries
    document_count: int

    class Meta:
        db_table = "django_ai_sdk_memories"
        indexes = [
            models.Index(fields=["is_hidden"]),
            models.Index(fields=["is_public"]),
        ]

    def __str__(self) -> str:
        return self.name

    def save(self, *args: Any, **kwargs: Any) -> None:
        if self._state.adding:
            if not self.slug and self.name:
                base_slug = slugify(self.name)
                slug = base_slug
                counter = 1
                while Memory.objects.filter(slug=slug).exists():
                    slug = f"{base_slug}-{counter}"
                    counter += 1
                self.slug = slug
        super().save(*args, **kwargs)

    @property
    def tool_name(self) -> str:
        """Provider-safe function name for this memory's RAG tool.

        Function names are restricted to ``[a-zA-Z0-9_-]``; anything else is
        rejected for the whole request. Names that sanitise to nothing (e.g.
        non-Latin scripts) fall back to the memory id.

        Truncation means this is not unique on its own — ``get_rag_tools``
        de-duplicates across the memories active on a thread.
        """
        slug = re.sub(r"[^a-z0-9]+", "_", self.name.lower()).strip("_")[:20].strip("_")
        return f"search_{slug}" if slug else f"search_memory_{str(self.id).replace('-', '')[:8]}"

    async def get_tool_spec(self, doc_count: int | None = None) -> ToolSpec:
        """Generate ToolSpec for this memory."""
        if doc_count is None:
            doc_count = await Entry.objects.filter(memory_id=self.id).acount()

        return ToolSpec(
            name=self.tool_name,
            description=(
                f"Search knowledge base: {self.name}. "
                f"Contains {doc_count} entries. "
                f"{self.description[:80] if self.description else ''} "
                f"Use this when you need information from {self.name}."
            ),
            doc_count=doc_count,
            metadata={"memory_id": str(self.id), "memory_name": self.name},
        )


class MemoryUser(models.Model):
    memory = models.ForeignKey(Memory, on_delete=models.CASCADE, related_name="memory_users")
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="memories",
    )
    can_manage = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    # Reverse relation type hints
    memory_id: int
    user_id: int

    class Meta:
        app_label = "django_ai_sdk"
        unique_together = [["memory", "user"]]
        db_table = "django_ai_sdk_memory_users"

    def __str__(self) -> str:
        return f"{self.user} - {self.memory.name}"


class MemoryGroup(models.Model):
    memory = models.ForeignKey(Memory, on_delete=models.CASCADE, related_name="memory_groups")
    group = models.ForeignKey(
        "auth.Group",
        on_delete=models.CASCADE,
        related_name="memories",
    )
    can_manage = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    # Reverse relation type hints
    memory_id: int
    group_id: int

    class Meta:
        app_label = "django_ai_sdk"
        db_table = "django_ai_sdk_memory_groups"
        unique_together = [["memory", "group"]]

    def __str__(self) -> str:
        return f"{self.group} - {self.memory.name}"


class Entry(models.Model):
    """
    A piece of knowledge stored in a Memory.
    Text content + JSON metadata. No file columns — clean and lightweight.
    File-backed entries are represented by an associated EntryDocument (OneToOne).
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    memory = models.ForeignKey(Memory, on_delete=models.CASCADE, related_name="entries")
    memory_id: str
    content = models.TextField(blank=True, default="")
    data = models.JSONField(default=dict, blank=True)
    name = models.CharField(max_length=255, blank=True, default="")
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "django_ai_sdk_documents"

    def __str__(self) -> str:
        return self.name or f"Entry {self.id}"

    @property
    def extraction(self) -> DocumentExtraction | None:
        """Parse data JSON as DocumentExtraction. Returns None for non-file entries."""
        if not self.data:
            return None
        try:
            return DocumentExtraction(**self.data)
        except Exception:
            return None

    @extraction.setter
    def extraction(self, value: DocumentExtraction | None) -> None:
        if value is None:
            self.data = {}
        else:
            self.data = value.model_dump()

    def to_rag_document(self) -> RagDocument:
        """Convert to RagDocument for retrieval. Enriches with extraction data when present."""
        extraction = self.extraction
        if extraction:
            parts = [self.content]
            if extraction.summary:
                parts.append(f"Summary: {extraction.summary}")
            if extraction.facts:
                parts.append(f"Key facts: {'. '.join(f.text for f in extraction.facts)}")
            if extraction.keywords:
                parts.append(f"Keywords: {', '.join(extraction.keywords)}")
            if extraction.entities:
                parts.append(
                    f"Named entities: {', '.join(f'{e.text} ({e.type})' for e in extraction.entities)}"
                )
            content = "\n\n".join(parts)
        else:
            content = self.content
        return RagDocument(
            id=str(self.id),
            content=content,
            metadata={
                "memory_id": str(self.memory_id),
                "name": self.name,
                "keywords": ". ".join(extraction.keywords) if extraction else "",
                "facts": ". ".join(f.text for f in extraction.facts) if extraction else "",
            },
        )


class EntryDocument(models.Model):
    class ProcessingStatus(models.TextChoices):
        PENDING = "pending", "Pending"
        PROCESSING = "processing", "Processing"
        COMPLETED = "completed", "Completed"
        FAILED = "failed", "Failed"
        CANCELLED = "cancelled", "Cancelled"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    entry = models.OneToOneField(
        Entry,
        on_delete=models.CASCADE,
        related_name="document",
        null=True,
        blank=True,
    )
    # Direct link to the owning Memory, set at upload time. Lets in-flight and
    # failed documents (which have no Entry yet) still be listed by memory.
    memory = models.ForeignKey(
        Memory,
        on_delete=models.CASCADE,
        related_name="documents",
        null=True,
        blank=True,
    )
    file = models.FileField(upload_to="memories/documents/")
    file_name = models.CharField(max_length=255, blank=True, default="")
    file_size = models.PositiveIntegerField(default=0)
    content_type = models.CharField(max_length=100, blank=True, default="")
    file_extension = models.CharField(max_length=20, blank=True, default="")
    file_hash = models.CharField(max_length=64, blank=True, default="")
    extracted = models.BooleanField(default=False)
    processing_status = models.CharField(
        max_length=20,
        choices=ProcessingStatus.choices,
        default=ProcessingStatus.PENDING,
        db_index=True,
    )
    task_id = models.CharField(max_length=64, null=True, blank=True)
    # Pipeline-defined progress marker (e.g. "ocr", "extracting"), null except
    # while processing_status=PROCESSING. Not a TextChoices enum: any
    # agent's custom processor/transform can supply its own step string.
    processing_step = models.CharField(max_length=32, null=True, blank=True, default=None)
    processing_error = models.TextField(blank=True, default="")
    # Checked cooperatively at each pipeline step boundary (see
    # memories/tasks.py); cancellation only takes effect between steps, not
    # mid-call. Null means not cancelled.
    cancelled_at = models.DateTimeField(null=True, blank=True, default=None)
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        app_label = "django_ai_sdk"
        db_table = "django_ai_sdk_entry_documents"

    def __str__(self) -> str:
        return self.file_name or f"EntryDocument {self.id}"


class ThreadMemory(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    thread = models.ForeignKey(
        "django_ai_sdk.Thread",
        on_delete=models.CASCADE,
        related_name="memory_links",
    )
    memory = models.ForeignKey(Memory, on_delete=models.CASCADE, related_name="thread_links")
    active = models.BooleanField(default=True)
    created_at = models.DateTimeField(default=timezone.now)

    # Annotated field from queries
    document_count: int

    class Meta:
        app_label = "django_ai_sdk"
        db_table = "django_ai_sdk_thread_memories"
        unique_together = [["thread", "memory"]]
        indexes = [
            models.Index(fields=["thread", "active"]),
        ]

    def __str__(self) -> str:
        return f"{self.thread} - {self.memory}"
