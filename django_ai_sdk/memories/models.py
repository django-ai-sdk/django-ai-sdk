import uuid

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
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="owned_memories",
    )
    contributors = models.ManyToManyField(
        settings.AUTH_USER_MODEL,
        blank=True,
        related_name="memories",
    )
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    # Reverse relation type hints
    entries: models.Manager["Entry"]
    # Annotated field from queries
    document_count: int

    class Meta:
        db_table = "django_ai_sdk_memories"

    def __str__(self) -> str:
        return self.name

    def save(self, *args: object, **kwargs: object) -> None:
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

    async def get_tool_spec(self) -> ToolSpec:
        """Generate ToolSpec for this memory."""
        doc_count = await Entry.objects.filter(memory_id=self.id).acount()

        return ToolSpec(
            name=f"search_{self.name.lower().replace(' ', '_')[:20]}",
            description=(
                f"Search knowledge base: {self.name}. "
                f"Contains {doc_count} entries. "
                f"{self.description[:80] if self.description else ''} "
                f"Use this when you need information from {self.name}."
            ),
            doc_count=doc_count,
            metadata={"memory_id": str(self.id), "memory_name": self.name},
        )


class Entry(models.Model):
    """
    A piece of knowledge stored in a Memory.
    Text content + JSON metadata. No file columns — clean and lightweight.
    File-backed entries are represented by an associated EntryDocument (OneToOne).
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    memory = models.ForeignKey(Memory, on_delete=models.CASCADE, related_name="entries")
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
        from django_ai_sdk.memories.utils import get_prompt_metadata

        extraction = self.extraction
        content = get_prompt_metadata(self.content, extraction) if extraction else self.content
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
    """
    An uploaded file attached to an Entry.
    Stored in a separate table — only joined when file metadata is actually needed.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    entry = models.OneToOneField(
        Entry,
        on_delete=models.CASCADE,
        related_name="document",
        null=True,
        blank=True,
    )
    file = models.FileField(upload_to="memories/documents/")
    file_name = models.CharField(max_length=255, blank=True, default="")
    file_size = models.PositiveIntegerField(default=0)
    content_type = models.CharField(max_length=100, blank=True, default="")
    file_extension = models.CharField(max_length=20, blank=True, default="")
    extracted = models.BooleanField(default=False)
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
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
        db_table = "django_ai_sdk_thread_memories"
        unique_together = [["thread", "memory"]]

    def __str__(self) -> str:
        return f"{self.thread} - {self.memory}"
