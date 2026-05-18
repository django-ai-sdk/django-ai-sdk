"""
TODO:
I moved these data schema out of the package and embedded them here for now.
We still use a light version, the package has better extraction handling as well,
but I need to finalize this.

Also check the PyPi account we are going to release from: there are now two versions,
The one for Caren will be separated and then the one for public release, both
need to be in sync.

"""

from enum import StrEnum

from pydantic import BaseModel, Field

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


# View Schemas


class MemoryIn(BaseModel):
    """Schema for creating a memory."""

    name: str
    slug: str = ""
    description: str = ""


class MemoryOut(BaseModel):
    """Schema for memory output."""

    id: str
    name: str
    slug: str
    description: str
    document_count: int
    created_at: str
    updated_at: str


class DocumentIn(BaseModel):
    """Schema for creating a document."""

    content: str = ""


class DocumentOut(BaseModel):
    """Schema for document output."""

    id: str
    file: str
    content: str
    extraction: DocumentExtraction | None = None
    file_name: str
    file_size: int
    content_type: str
    file_extension: str
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


class BulkConnectMemoriesIn(BaseModel):
    """Schema for bulk connecting memories to a thread."""

    memory_ids: list[str]


class ToggleMemoryActiveIn(BaseModel):
    """Schema for toggling memory active status."""

    active: bool
