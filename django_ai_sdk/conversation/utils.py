from django_ai_sdk.views.schemas import Message


# TODO: will do for now, but not the cleanest.
# but we could ask LLM as helper, to generate a proper title.
def generate_thread_title(messages: list[Message]) -> str:
    for message in messages:
        if message.role != "user":
            continue
        for part in message.parts or []:
            if part.type == "text" and part.text:
                text = part.text
                title = text[:50].strip()
                return title + "..." if len(text) > 50 else title
    return "New Conversation"
