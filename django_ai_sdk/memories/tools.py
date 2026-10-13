from __future__ import annotations

import uuid
from typing import Any

from haystack.tools import Tool

from django_ai_sdk.artifacts.schemas import FileArtifact
from django_ai_sdk.artifacts.tool_artifacts import ToolArtifact
from django_ai_sdk.files.processors import describe_image
from django_ai_sdk.memories.models import EntryDocument
from django_ai_sdk.memories.services import aread_document, get_thread_file_url

ASK_IMAGE_TOOL = "ask_image"


async def get_thread_image(thread_id: str, image: str) -> EntryDocument | None:
    """An image uploaded to this thread, by document id or (partial) filename."""
    qs = EntryDocument.objects.filter(
        memory__thread_files__id=thread_id, content_type__startswith="image/"
    )
    try:
        return await qs.filter(id=uuid.UUID(image)).afirst()
    except ValueError:
        return await qs.filter(file_name__icontains=image).order_by("-created_at").afirst()


def ask_image_tool(thread_id: str) -> Tool:
    """Answer a question about an image in the thread with the vision model.

    Lets agents whose own model can't see images, and follow-up turns after
    the one the image was sent on, still look at the picture.
    """

    async def _run(image: str, question: str) -> str:
        doc = await get_thread_image(thread_id, image)
        if doc is None:
            return f"No image {image!r} in this thread. Tell the user it is not available."
        data = await aread_document(doc)
        answer = await describe_image(data, doc.content_type, question)
        return answer or "The vision model returned no answer."

    return Tool(
        name=ASK_IMAGE_TOOL,
        description=(
            "Look at an image uploaded to this thread and answer a question about "
            "it. Use this for anything about an image's visual content (what it "
            "shows, colours, text in it, details) that its description doesn't "
            "already answer."
        ),
        parameters={
            "type": "object",
            "properties": {
                "image": {
                    "type": "string",
                    "description": "The image's document id, or its filename.",
                },
                "question": {
                    "type": "string",
                    "description": "What to find out about the image, self-contained.",
                },
            },
            "required": ["image", "question"],
        },
        async_function=_run,
    )


def get_thread_file(doc: EntryDocument, thread_id: str) -> dict[str, str]:
    """Document of the thread."""
    return {
        "documentId": str(doc.id),
        "filename": doc.file_name,
        "mediaType": doc.content_type,
        "url": get_thread_file_url(doc, thread_id),
    }


async def _get_image(
    arguments: dict[str, Any], result: str, thread_id: str
) -> dict[str, list[dict[str, str]]] | None:
    doc = await get_thread_image(thread_id, str(arguments.get("image", "")))
    return {"files": [get_thread_file(doc, thread_id)]} if doc else None


#: Show the image `ask_image` looked at, as a file artifact:
#: ``tool_artifacts = {ASK_IMAGE_TOOL: ASK_IMAGE_ARTIFACT}``.
ASK_IMAGE_ARTIFACT = ToolArtifact(FileArtifact, _get_image)
