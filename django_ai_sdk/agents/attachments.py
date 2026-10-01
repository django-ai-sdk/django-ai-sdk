from __future__ import annotations

from typing import TYPE_CHECKING, Any

from django_ai_sdk.utils import resolve_setting

if TYPE_CHECKING:
    from django_ai_sdk.memories.models import EntryDocument


class InlineFileCapability:
    """Files attached to user messages: what the model gets to see of them.

    Every attachment becomes a context line (`format_attachment` /
    `format_image_attachment`). Images go inline as pixels when the agent
    `has_vision()`; otherwise the model gets their extracted description plus,
    with a vision agent configured, the `ask_image` tool. To make the
    model look a file up first, override `Agent.get_run_required_tools`.
    """

    # Provided by Agent.
    file_upload: bool

    # The model accepts images: image attachments on the latest user message are
    # sent as pixels.
    vision: bool = False

    def has_vision(self) -> bool:
        """Whether this agent's own model gets image attachments as pixels."""
        return self.vision

    def get_attachment_tools(self, thread_id: str) -> list[Any]:
        """`ask_image`, when this agent takes uploads and a vision agent is set."""
        from django_ai_sdk.files.processors import has_vision_support
        from django_ai_sdk.memories.tools import ask_image_tool

        if self.file_upload and thread_id and has_vision_support():
            return [ask_image_tool(thread_id)]
        return []

    def format_attachment(self, doc: EntryDocument) -> str:
        """Context line the model gets for a file attached to a user message.

        Override to point the model at your own file tools.
        """
        memory = doc.memory.description if doc.memory else ""
        line = (
            f'[Attached file "{doc.file_name}" ({doc.content_type or "unknown type"}), '
            f"document id {doc.id}, stored in {memory or 'thread memory'}."
        )
        summary = (doc.entry.data or {}).get("summary") if doc.entry else None
        if summary:
            line += f" Summary: {summary}"
        return line + " Use search_uploaded_documents to query its content.]"

    def format_image_attachment(self, doc: EntryDocument, *, inline: bool) -> str:
        """Context line for an image attached to a user message.

        `inline` means the model gets the pixels with this message; otherwise it
        gets the description from memory extraction and, when a vision agent
        is configured, is pointed at the `ask_image` tool.
        """
        from django_ai_sdk.files.processors import has_vision_support
        from django_ai_sdk.memories.tools import ASK_IMAGE_TOOL

        line = f'[Attached image "{doc.file_name}", document id {doc.id}'
        if inline:
            return line + ", included in this message.]"
        caption = (doc.entry.content if doc.entry else "").strip()
        limit = resolve_setting("AI_SDK_IMAGE_CAPTION_LIMIT", 1500)
        if limit and limit > 0 and len(caption) > limit:
            caption = caption[:limit].rstrip() + "…"
        if caption:
            line += f". You can't see it; auto-generated description: {caption}"
        else:
            line += ". You can't see it and it has no description yet."
        if has_vision_support():
            line += (
                f" For anything the description doesn't answer, call {ASK_IMAGE_TOOL}"
                f" with document id {doc.id}."
            )
        return line + "]"
