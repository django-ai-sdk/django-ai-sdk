from __future__ import annotations

from typing import TYPE_CHECKING, Any

from django_ai_sdk.utils import resolve_setting

if TYPE_CHECKING:
    from django_ai_sdk.memories.models import EntryDocument


class InlineFileCapability:
    file_upload: bool

    vision: bool = False

    def has_vision(self) -> bool:
        return self.vision

    def get_attachment_tools(self, thread_id: str) -> list[Any]:
        from django_ai_sdk.files.processors import has_vision_support
        from django_ai_sdk.memories.tools import ask_image_tool

        if self.file_upload and thread_id and has_vision_support():
            return [ask_image_tool(thread_id)]
        return []

    def format_attachment(self, doc: EntryDocument) -> str:
        content_type = doc.content_type or "unknown type"
        stored_in = (doc.memory.description if doc.memory else "") or "thread memory"
        summary = (doc.entry.data or {}).get("summary") if doc.entry else None

        sentences = [
            f'Attached file "{doc.file_name}" ({content_type}), document id {doc.id}, '
            f"stored in {stored_in}."
        ]
        if summary:
            sentences.append(f"Summary: {summary}")
        sentences.append("Use `search_uploaded_documents` to query its content.")
        return get_context_line(sentences)

    def format_image_attachment(self, doc: EntryDocument, *, inline: bool) -> str:
        from django_ai_sdk.files.processors import has_vision_support
        from django_ai_sdk.memories.tools import ASK_IMAGE_TOOL

        sentences = [f'Attached image "{doc.file_name}", document id {doc.id}.']
        if inline:
            sentences.append("It is included in this message.")
            return get_context_line(sentences)

        if caption := get_image_caption(doc):
            sentences.append(f"You can't see it; auto-generated description: {caption}")
        else:
            sentences.append("You can't see it and it has no description yet.")
        if has_vision_support():
            sentences.append(
                f"For anything the description doesn't answer, call {ASK_IMAGE_TOOL} "
                f"with document id {doc.id}."
            )
        return get_context_line(sentences)


def get_context_line(sentences: list[str]) -> str:
    return f"[{' '.join(sentences)}]"


def get_image_caption(doc: EntryDocument) -> str:
    caption = (doc.entry.content if doc.entry else "").strip()
    limit = resolve_setting("AI_SDK_IMAGE_CAPTION_LIMIT", 1500)
    if limit and limit > 0 and len(caption) > limit:
        caption = caption[:limit].rstrip() + "…"
    return caption
