from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from django_ai_sdk.common import ChatMessage
    from django_ai_sdk.memories.models import EntryDocument


class InlineFileCapability:
    """Files attached to user messages: what the model gets to see of them.

    Every attachment becomes a context line (`format_attachment` /
    `format_image_attachment`). Images go inline as pixels when the agent
    `has_vision()`; otherwise the model gets their extracted description plus,
    with `AI_SDK_VISION_MODEL` set, the `ask_image` tool.
    """

    # Provided by Agent.
    file_upload: bool

    def get_model(self) -> str:
        raise NotImplementedError

    # The model accepts images: image attachments on the latest user message are
    # sent as pixels. Also true without this flag when the agent runs on
    # AI_SDK_VISION_MODEL (see has_vision).
    vision: bool = False

    # Tools the agent must call on a run whose latest user message has
    # attachments other than images (images are covered by pixels or caption).
    attachment_tools: list[str] = []

    def has_vision(self) -> bool:
        """Whether this agent's own model gets image attachments as pixels."""
        from django_ai_sdk.files.processors import get_vision_model

        vision_model = get_vision_model()
        return self.vision or bool(vision_model and self.get_model() == vision_model)

    def get_run_required_tools(self, messages: list[ChatMessage]) -> list[str]:
        """Tools this run must call on top of `required_tools` (enforced by the tool agent)."""
        last_user = next((m for m in reversed(messages) if m.role == "user"), None)
        # Images are covered by their pixels or caption; ask_image stays optional.
        if last_user and any(not a.media_type.startswith("image/") for a in last_user.attachments):
            return list(self.attachment_tools)
        return []

    def get_attachment_tools(self, thread_id: str) -> list[Any]:
        """`ask_image`, when this agent takes uploads and a vision model is set."""
        from django_ai_sdk.files.processors import get_vision_model
        from django_ai_sdk.memories.tools import ask_image_tool

        if self.file_upload and thread_id and get_vision_model():
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
        gets the description from memory extraction and, when a vision model is
        configured, is pointed at the `ask_image` tool.
        """
        from django_ai_sdk.files.processors import get_vision_model
        from django_ai_sdk.memories.tools import ASK_IMAGE_TOOL

        line = f'[Attached image "{doc.file_name}", document id {doc.id}'
        if inline:
            return line + ", included in this message.]"
        # ponytail: fixed cap on the caption, make it a setting if it's too short
        caption = (doc.entry.content if doc.entry else "")[:1500]
        if caption:
            line += f". You can't see it; auto-generated description: {caption}"
        else:
            line += ". You can't see it and it has no description yet."
        if get_vision_model():
            line += (
                f" For anything the description doesn't answer, call {ASK_IMAGE_TOOL}"
                f" with document id {doc.id}."
            )
        return line + "]"
