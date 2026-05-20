---
title: Tools and Agents
type: docs
prev: building-assistants
next: views-and-routing
weight: 3
---

This page covers how to give your assistants tools -- functions they can call during a conversation.

## OpenAI Agent with Co-located Tools

The `OpenAIAgentAdapter` uses the `agents` library for function calling. You define tools as methods on your assistant class with the `@function_tool` decorator.

From the demo (`demo/piratespeak/assistants/pirate_agent.py`):

```python
import random
from agents import Agent, function_tool, set_default_openai_client
from django.conf import settings
from openai import AsyncOpenAI
from django_ai_sdk import Assistant
from django_ai_sdk.adapters.openai import OpenAIAgentAdapter
from django_ai_sdk.protocols.vercel import VercelProtocolHandler

class PirateAgentAssistant(Assistant):
    name = "Captain Blackbeard Bot"
    model = "gpt-4"
    instructions = [
        "You are Captain Blackbeard Bot, a swashbuckling pirate AI assistant!",
        "",
        "Always respond in character as a pirate captain.",
        "Use your tools for pirate tasks and entertainment.",
    ]
    protocol = VercelProtocolHandler

    def get_tools(self):
        return [
            self.get_pirate_insult,
            self.get_treasure_location,
            self.get_weather_forecast,
            self.tell_pirate_joke,
            self.get_ship_status,
        ]

    @function_tool(name_override="pirate_insult")
    def get_pirate_insult(self) -> str:
        """Get a creative pirate insult."""
        insults = [
            "Ye scurvy bilge rat!",
            "Arr, ye landlubber!",
            "Ye mangy sea dog!",
        ]
        return random.choice(insults)

    @function_tool(name_override="treasure_location")
    def get_treasure_location(self) -> str:
        """Get information about a mysterious treasure location."""
        locations = [
            "Buried beneath the twisted oak on Dead Man's Island",
            "Hidden in the caves of Skull Rock, past the third waterfall",
        ]
        return f"Arr! The treasure be {random.choice(locations)}!"

    @function_tool(name_override="weather_forecast")
    def get_weather_forecast(self) -> str:
        """Get a pirate-themed weather forecast for sailing."""
        conditions = [
            "Perfect sailing weather with fair winds and following seas!",
            "Storm clouds brewing on the horizon - batten down the hatches!",
        ]
        return f"Arr! {random.choice(conditions)}"

    async def get_pipeline_adapter(self, thread_id=None):
        # Get storage adapter for this thread
        storage_adapter = await self.get_storage_adapter(thread_id)
        
        client = AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=getattr(settings, "OPENAI_API_URL", None),
        )
        set_default_openai_client(client)

        agent = Agent(
            name=self.name,
            model=self.get_model(),
            instructions=self.get_instructions(),
            tools=self.get_tools(),
        )
        return OpenAIAgentAdapter(agent=agent, storage_adapter=storage_adapter)
```

Key things to note:

- `@function_tool` makes a method callable by the agent. The docstring becomes the tool description the model sees.
- `name_override` controls the tool name exposed to the model (otherwise it uses the method name).
- `get_tools()` returns the list of tools -- the agent gets these when it's created in `get_pipeline_adapter()`.
- You need to call `set_default_openai_client()` to configure the `agents` library with your API client.

When the model calls a tool, the stream includes tool call events:

```
data: {"type":"tool-input-start","toolCallId":"tool_abc","toolName":"pirate_insult"}
data: {"type":"tool-input-available","toolCallId":"tool_abc","toolName":"pirate_insult","input":{}}
data: {"type":"tool-output-available","toolCallId":"tool_abc","output":"Ye scurvy bilge rat!"}
data: {"type":"text-delta","id":"text_1","delta":"Arr! Here be an insult for ye..."}
```

## Agent Swarm with Haystack

For more complex setups where you want specialist tools coordinated by a triage agent, use Haystack's `Tool` objects with `HaystackAdapter`.

From the demo (`demo/piratespeak/assistants/agent_swarm.py`):

```python
import random
from typing import Annotated
from django.conf import settings
from haystack import Pipeline
from haystack.components.agents import Agent as HaystackAgent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.tools import Tool
from haystack.utils import Secret
from django_ai_sdk import Assistant
from django_ai_sdk.adapters.haystack import HaystackAdapter
from django_ai_sdk.protocols.vercel import VercelProtocolHandler

# Tool functions must be standalone (not methods) for Haystack serialization
def pirate_boat_expert(topic: Annotated[str, "Topic about pirate boats"]):
    """Provide expert lore about pirate boats."""
    return f"Aye! On pirate boats: {topic}! They be swift and full o' tales."

def find_treasure(location: Annotated[str, "Location to search for treasure"]):
    """Simulate treasure discovery."""
    treasure_id = random.randint(1000, 9999)
    value = random.randint(5000, 50000)
    return {
        "location": location,
        "treasure_id": treasure_id,
        "value": f"{value} gold coins",
    }

class AgentSwarmAssistant(Assistant):
    name = "Pirate Agent Swarm"
    model = "gpt-4"
    instructions = [
        "You are a Triage Agent for a crew of pirate specialists.",
        "",
        "Decide whether the user wants:",
        "- pirate boat expertise (call pirate_boat_expert)",
        "- treasure finding (call find_treasure)",
        "- or general pirate help",
    ]
    protocol = VercelProtocolHandler

    def get_tools(self):
        return [
            Tool(
                name="pirate_boat_expert",
                description="Expert knowledge about pirate boats",
                parameters={
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string", "description": "Subject about pirate boats"}
                    },
                    "required": ["topic"],
                },
                function=pirate_boat_expert,
            ),
            Tool(
                name="find_treasure",
                description="Find treasure at a given location",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "Location to search"}
                    },
                    "required": ["location"],
                },
                function=find_treasure,
            ),
        ]

    async def get_pipeline_adapter(self, thread_id=None):
        storage_adapter = await self.get_storage_adapter(thread_id)
        pipeline = Pipeline()

        triage_agent = HaystackAgent(
            chat_generator=OpenAIChatGenerator(
                model=self.get_model(),
                api_key=Secret.from_env_var("OPENAI_API_KEY"),
                api_base_url=getattr(settings, "OPENAI_API_URL", None),
            ),
            tools=self.get_tools(),
            system_prompt=self.get_system_prompt(),
            exit_conditions=["text"],
        )
        pipeline.add_component("triage_agent", triage_agent)

        return HaystackAdapter(
            pipeline=pipeline,
            generator_component=triage_agent.chat_generator,
            storage_adapter=storage_adapter,
        )
```

The key differences from the OpenAI agent pattern:

- Tool functions are **standalone** (not methods on the class) -- Haystack needs this for serialization.
- You wrap them in `Tool()` objects with explicit JSON Schema `parameters`.
- The triage agent decides which specialist tool to call based on the user's request.

## Which Tool Pattern Should I Use?

| Pattern | Adapter | How tools are defined | Good for |
|---------|---------|----------------------|----------|
| Co-located `@function_tool` | `OpenAIAgentAdapter` | Methods on the assistant class | Most use cases, keeps code organized |
| Standalone `Tool()` objects | `HaystackAdapter` | Functions outside the class | Haystack pipelines, shared tools across assistants |
| No tools | `OpenAIAdapter` | N/A | Simple conversational AI |
