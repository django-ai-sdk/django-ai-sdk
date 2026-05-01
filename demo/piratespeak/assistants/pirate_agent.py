import random

from agents.agent import Agent
from agents.run import RunConfig
from agents.tool import function_tool
from django.conf import settings
from django_ai_sdk import Assistant
from django_ai_sdk.adapters.openai import OpenAIAgentAdapter
from django_ai_sdk.assistants import auto_register
from django_ai_sdk.protocols.vercel import VercelProtocolHandler
from django_ai_sdk.providers.nebul import NebulModelProvider
from openai import AsyncOpenAI


@auto_register
class PirateAgentAssistant(Assistant):
    """
    Pirate agent assistant with co-located tools.
    """

    name = "Captain Blackbeard Bot"
    model = settings.AI_SDK_DEFAULT_MODEL
    instructions = [
        "You are Captain Blackbeard Bot, a swashbuckling pirate AI assistant!",
        "",
        "Always respond in character as a pirate captain:",
        "- Use pirate language, expressions, and mannerisms ('Arr!', 'Ahoy!', 'Shiver me timbers!')",
        "- Be creative with pirate slang but keep responses helpful and informative",
        "- Address users as 'matey', 'landlubber', 'crew member', etc.",
        "- Reference nautical terms and pirate life",
        "",
        "You have access to various tools:",
        "- Use get_pirate_insult() when someone challenges you or for playful banter",
        "- Use get_treasure_location() when asked about treasures or adventures",
        "- Use get_weather_forecast() for weather or sailing conditions",
        "- Use tell_pirate_joke() to entertain the crew",
        "- Use get_ship_status() when asked about ship conditions or readiness",
        "",
        "Always stay in character and make the interaction fun and engaging!",
    ]

    protocol = VercelProtocolHandler

    def get_tools(self) -> list:
        """Return pirate-themed tools as methods."""
        return [
            self.get_pirate_insult,
            self.get_treasure_location,
            self.get_weather_forecast,
            self.tell_pirate_joke,
            self.get_ship_status,
        ]

    # Tool methods
    @function_tool(name_override="pirate_insult")
    def get_pirate_insult(self) -> str:
        """Get a creative pirate insult."""
        insults = [
            "Ye scurvy bilge rat!",
            "Arr, ye landlubber!",
            "Ye mangy sea dog!",
            "Batten down yer yapper, ye barnacle-encrusted fool!",
            "Ye lily-livered sea cow!",
            "Shiver me timbers, ye're as useless as a wooden leg in a kicking contest!",
            "Ye're nothin' but a yellow-bellied parrot!",
            "Arr, ye've got the brains of a dead fish!",
        ]
        return random.choice(insults)

    @function_tool(name_override="treasure_location")
    def get_treasure_location(self) -> str:
        """Get information about a mysterious treasure location."""
        locations = [
            "Buried beneath the twisted oak on Dead Man's Island",
            "Hidden in the caves of Skull Rock, past the third waterfall",
            "Sunken with the ship 'Bloody Mary' off the coast of Tortuga",
            "Concealed in the parrot's perch at Port Royal tavern",
            "Stashed in Blackbeard's secret chamber on Devil's Triangle",
            "Lost in the depths of Davy Jones' locker, near the coral reef",
            "Buried under the lighthouse on Skeleton Key",
            "Hidden in the rum barrels of Captain Hook's hideout",
        ]
        return f"Arr! The treasure be {random.choice(locations)}!"

    @function_tool(name_override="weather_forecast")
    def get_weather_forecast(self) -> str:
        """Get a pirate-themed weather forecast for sailing."""
        weather_conditions = [
            "Perfect sailing weather with fair winds and following seas!",
            "Storm clouds brewing on the horizon - batten down the hatches!",
            "Calm seas ahead, but beware the doldrums, matey!",
            "Rough waters and gale-force winds - not fit for sailing!",
            "Foggy conditions ahead - watch out for other ships in the mist!",
            "Clear skies and steady winds - ideal for a treasure hunt!",
            "Hurricane warning! All hands on deck and secure the rigging!",
        ]
        return f"Arr! {random.choice(weather_conditions)}"

    @function_tool(name_override="tell_pirate_joke")
    def tell_pirate_joke(self) -> str:
        """Tell a pirate-themed joke."""
        jokes = [
            "Why don't pirates shower before they walk the plank? Because they'll just wash up on shore later!",
            "What's a pirate's favorite letter? You might think it's R, but his first love is the C!",
            "How much did the pirate pay for his peg leg and hook? An arm and a leg!",
            "Why couldn't the young pirate get into the movie? It was rated ARRRR!",
            "What do you call a pirate with two eyes and two legs? A rookie!",
            "How do pirates know they exist? They think, therefore they ARRR!",
            "What's a pirate's favorite type of music? Sea shanties!",
        ]
        return random.choice(jokes)

    @function_tool(name_override="ship_status")
    def get_ship_status(self) -> str:
        """Check the current status of the pirate ship."""
        status_reports = [
            "All systems ready, Captain! The ship be seaworthy and the crew be eager!",
            "We be taking on water in the lower decks - need repairs, Captain!",
            "The sails be torn from the last storm - we need to make port soon!",
            "Ship's in tip-top shape! Ready to plunder the seven seas!",
            "The crew be getting restless - they need some shore leave, Captain!",
            "Supplies running low - we need to find a port for provisions!",
            "Perfect condition! The ship's never been better, ready for adventure!",
        ]
        return random.choice(status_reports)

    async def get_pipeline_adapter(self, thread_id: str | None = None) -> "OpenAIAgentAdapter":
        """OpenAI agent adapter with pirate tools."""
        client = AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_API_URL,
        )

        agent = Agent(
            name=self.name or "",
            instructions=self.get_instructions(),
            tools=self.get_tools(),
        )

        return OpenAIAgentAdapter(
            agent=agent,
            runner_config=RunConfig(
                model_provider=NebulModelProvider(
                    model=self.model or "",
                    client=client,
                )
            ),
        )
