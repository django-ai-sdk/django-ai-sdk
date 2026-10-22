from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from django_ai_sdk.tasks import TaskStatus

# Data schema


class Predicate(StrEnum):
    """Allowed predicate values for extracted facts."""

    BIRTH_PLACE = "birthPlace"
    WORKS_FOR = "worksFor"
    ALUMNI_OF = "alumniOf"
    MEMBER_OF = "memberOf"
    KNOWS = "knows"
    HAS_SKILL = "hasSkill"
    HAS_INTEREST = "hasInterest"
    LOCATED_IN = "locatedIn"
    DATE_STARTED = "dateStarted"
    DATE_COMPLETED = "dateCompleted"


class Fact(BaseModel):
    """A structured fact extracted from the document."""

    subject: str = Field(..., description="Entity or concept being described.")
    predicate: Predicate = Field(..., description="Normalized relation type.")
    object: str = Field(..., description="Value or entity the predicate relates to.")
    text: str = Field(..., description="Human-readable sentence expressing the fact.")
    evidence: str = Field(..., description="Short excerpt from document supporting the fact.")


class Entity(BaseModel):
    """Representation of a named entity with its type."""

    text: str = Field(..., description="Named entity text")
    type: str = Field(..., description="NER type (e.g., PERSON, ORG, DATE)")


class Section(BaseModel):
    """Representation of a document section with an optional heading."""

    heading: str = Field(..., description="Section heading if available")
    content: str = Field(..., description="Text under this section")


class Event(BaseModel):
    """Structured representation of an event extracted from text."""

    description: str = Field(..., description="Event description")
    participants: list[str] = Field(..., description="Entities involved in the event")


class DocumentExtraction(BaseModel):
    """Structured output for document content extraction."""

    summary: str = Field(..., description="High-level summary of the document")
    keywords: list[str] = Field(..., description="Important keywords or topics")
    entities: list[Entity] = Field(..., description="Named entities with types")
    facts: list[Fact] = Field(..., description="Extracted facts for graph + narrative context.")
    sections: list[Section] = Field(..., description="Detected sections with headings and content")
    events: list[Event] = Field(..., description="Events extracted from text")


# What MemoryService returns (also reused as API response shapes)


class MemoryOut(BaseModel):
    """Schema for memory output."""

    id: str
    name: str
    slug: str
    description: str
    is_public: bool
    document_count: int
    created_at: str
    updated_at: str


class DocumentOut(BaseModel):
    """Schema for document output.

    `id` is the EntryDocument id (stable across the upload → processing →
    completed/failed lifecycle). `status` reflects EntryDocument.processing_status
    and `error_code`/`error` describe a processing failure (never its raw text). `content`/`extraction` are
    only populated once processing has produced an Entry.
    """

    id: str
    file: str
    content: str
    extraction: DocumentExtraction | None = None
    file_name: str
    data: dict = Field(default_factory=dict)
    file_size: int
    content_type: str
    file_extension: str
    status: str
    error: str = ""
    error_code: str = ""
    processing_step: str | None = None
    created_at: str
    updated_at: str


class ThreadMemoryOut(BaseModel):
    """Schema for thread-memory relationship output."""

    id: str
    name: str
    description: str
    document_count: int
    active: bool
    created_at: str


class DocumentUploadResponse(BaseModel):
    """Schema to return after upload"""

    id: str
    status: str
    processing_step: str | None = None
    task_id: str | None = None


class DocumentStatusOut(BaseModel):
    """Processing status for an uploaded document."""

    id: str
    status: str
    error: str = ""
    error_code: str = ""
    processing_step: str | None = None
    task: TaskStatus | None = None


class MemoryUserOut(BaseModel):
    """Schema for memory user output."""

    user_id: str
    email: str = ""
    first_name: str = ""
    last_name: str = ""
    can_manage: bool
    created_at: str


class MemoryGroupOut(BaseModel):
    """Schema for memory group output."""

    group_id: int
    group_name: str
    can_manage: bool
    created_at: str
