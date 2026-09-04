from __future__ import annotations

from typing import TYPE_CHECKING

from django.conf import settings
from django_ai_sdk import Agent
from django_ai_sdk.adapters.base import Run, Stream
from django_ai_sdk.agents import LogToolCallsHook, ToolAgent, ToolAgentConfig, auto_register
from django_ai_sdk.citations import DefaultCitationFormatter
from django_ai_sdk.common import prompt
from django_ai_sdk.generators import openai_responses_chat
from django_ai_sdk.suggestions import DefaultSuggestionGenerator

from apps.agents.tools import fetch_page_tool, search_web_tool

if TYPE_CHECKING:
    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser


@auto_register
class ResearchPlannerAgent(Agent):
    """Research subagent that searches the web and builds a report."""

    name = "Research Planner"
    description = (
        "Researches a topic by searching the web, fetching pages, "
        "and returning a structured report."
    )
    model = settings.AI_SDK_DEFAULT_MODEL
    llm = openai_responses_chat
    llm_kwargs = {"reasoning": {"effort": "high", "summary": "detailed"}}
    hidden = True
    tools: list = [search_web_tool, fetch_page_tool]

    max_agent_steps = 3
    max_tool_calls = 2

    instructions = prompt("""\
        You are a research subagent.

        Given a delegated research task:
        1. Run up to 2 web searches with search_web.
        2. Fetch the most relevant pages with fetch_page to get details.
        3. Synthesize findings into a structured report with sections and
           a short summary at the top. Cite sources by URL in each section.

        Research the exact wording of the task. Do not autocorrect proper nouns,
        company names, or typos. If the task mentions a specific string (e.g. a
        company name like "Amazon"), search for that literal string and note
        ambiguity if search turns up nothing obvious.

        Never fabricate search results, URLs, or sources.

        If a search_web call fails with an error (e.g. rate limit, timeout, or
        any other problem), you must NOT continue as if you searched. State
        clearly that web search failed and that the report therefore cannot
        contain verified live facts, versions, dates, or URLs. Do not provide
        specific version numbers, release dates, or URLs unless they came from a
        successful search_web result. Do not retry the same failed search
        endlessly.

        If a search returns no results, say the topic could not be verified via
        live search rather than inventing citations.

        Reply with only the final report text.
    """)


@auto_register
class DeepResearchAgent(Agent):
    """
    Deep research agent that delegates to a hidden research subagent.
    """

    name = "Deep Research Agent"
    description = "Delegates multi-step web research to a hidden research subagent."
    model = settings.AI_SDK_DEFAULT_MODEL
    llm = openai_responses_chat
    llm_kwargs = {"reasoning": {"effort": "low", "summary": "auto"}}
    instructions = prompt("""\
        You are a deep research coordinator.

        When the user asks a research question:
        - Delegate the full research task to the Research Planner subagent in a
          SINGLE call. Copy the user's question verbatim into the `task`
          parameter: never paraphrase, autocorrect, split it, or expand it.
        - Present the returned report to the user as-is, formatted in Markdown.
          Do not invent sources or facts the subagent did not include.
        - For non-research questions, answer directly.
    """)

    agents = [ResearchPlannerAgent]

    citation_formatter_class = DefaultCitationFormatter
    suggestion_generator = DefaultSuggestionGenerator

    async def get_run_adapter(
        self,
        thread_id: str | None = None,
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> Run:
        return Run(generator=self.get_llm())

    async def get_pipeline_adapter(
        self,
        thread_id: str | None = None,
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> Stream:
        """Create Haystack pipeline adapter with the research subagent."""

        generator = self.get_llm()
        storage_adapter = await self.get_storage_adapter(thread_id)

        tool_agent = ToolAgent(
            config=ToolAgentConfig(
                model=self.get_model(),
                system_prompt=self.get_system_prompt(),
                tools=await self.get_tools(thread_id=thread_id or "", user=user),
                exit_conditions=["text"],
                max_agent_steps=10,
                hooks={"before_tool": [LogToolCallsHook()]},
            ),
            generator=generator,
        )

        pipeline = tool_agent.pipeline()

        return Stream(
            pipeline=pipeline,
            generator=generator,
            storage_adapter=storage_adapter,
            citation_registry=self.get_citation_registry(),
            suggestion_generator=self.get_suggestion_generator(),
        )
