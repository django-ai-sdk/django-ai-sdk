from django_ai_sdk.common import Prompt, prompt

# Prompt template used for thread title generation.
TITLE_GENERATION_PROMPT: Prompt = prompt("""\
    ### Task:
    Generate a concise, 3-5 word title summarizing the chat history.

    ### Guidelines:
    - The title should clearly represent the main theme or subject of the conversation.
    - Start the title with a single emoji that enhances understanding of the topic.
    - Write the title in the same language as the user's messages.
    - default to English if multilingual or unclear.
    - Match the tone and register of the user (formal, casual, technical, etc.).
    - Prioritize accuracy over creativity; keep it clear and simple.

    ### Output rules (strict):
    - Return ONLY the title. No preamble, no explanation, no commentary.
    - No markdown, no quotes, no backticks, no code fences.
    - A single line of plain text.
    - Maximum 60 characters including the emoji.

    ### Examples:
    - 📉 Stock Market Trends
    - 🍪 Perfect Chocolate Chip Recipe
    - 🎮 Video Game Development Insights
""")
