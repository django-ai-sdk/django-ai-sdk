from django.conf import settings
from django_ai_sdk import Assistant
from django_ai_sdk.adapters.openai import OpenAIAdapter
from django_ai_sdk.assistants import auto_register
from django_ai_sdk.common import prompt
from django_ai_sdk.rags import (
    BM25RAG,
    BM25Config,
    RagDocument,
    RAGProvider,
)
from openai import AsyncOpenAI


@auto_register
class PirateOpenAIAssistant(Assistant):
    name = "OpenAI Pirate with RAG"
    model = settings.AI_SDK_DEFAULT_MODEL
    instructions = prompt("""\
        You are a knowledgeable pirate AI assistant.
        Use pirate language, expressions, and mannerisms in all your responses.
        Address users as 'matey', 'landlubber', or 'crew member'.
        You have access to a knowledge base about pirate life, rules, treasure, and weapons.
        Use the context provided to answer questions accurately.
    """)

    # protocol = VercelProtocolHandler
    # storage_adapter = MemoryStorageAdapter
    rag_provider = RAGProvider()

    def get_example_documents(self) -> list[RagDocument]:
        """
        Get 5 pirate-themed documents
        """
        return [
            RagDocument(
                id="pirate_code",
                content=(
                    "The Pirate Code of Conduct includes several important rules that all crew members must follow: "
                    "1) Every pirate has a vote in affairs of the moment and equal title to fresh provisions and liquors. "
                    "2) No pirate shall gamble with cards or dice for money while on board the ship. "
                    "3) The lights and candles must be put out by 8 PM to prevent fire. "
                    "4) Each pirate shall keep his piece (gun), pistols, and cutlass clean and fit for service. "
                ),
                metadata={
                    "source": "pirate_lore",
                    "topic": "rules",
                    "category": "governance",
                },
            ),
            RagDocument(
                id="treasure_map",
                content=(
                    "X marks the spot on the old island where Captain Blackbeard buried his treasure. "
                    "Follow these directions: Start at the skull-shaped rock on the north shore. "
                    "Walk 20 paces east toward the three palm trees. Turn south and walk until you reach the black rocks. "
                    "From there, go 15 paces west to the large cave entrance. The chest is buried 3 feet deep, "
                    "marked by a red cloth. Inside you'll find 500 gold doubloons, Spanish silver pieces of eight, "
                    "jewels from the Spanish Main, and a golden chalice. Beware the trap!"
                ),
                metadata={
                    "source": "pirate_lore",
                    "topic": "treasure",
                    "category": "navigation",
                },
            ),
            RagDocument(
                id="ship_commands",
                content=(
                    "Common pirate ship commands and their meanings: 'Avast ye!' means stop and pay attention immediately. "
                    "'Aye aye, Captain!' is the proper response when you understand an order. "
                    "'Shiver me timbers!' is an expression of surprise or disbelief. "
                    "'Weigh anchor and hoist the mizzen!' means prepare to set sail. "
                    "'Clear the deck!' means remove all obstacles and prepare for battle. "
                    "'Man the cannons!' is the call to prepare for combat. "
                    "'Scuttle the ship!' means intentionally sink the vessel. "
                    "'Walk the plank!' is the punishment where a person must walk off a board into the sea."
                ),
                metadata={
                    "source": "pirate_lore",
                    "topic": "language",
                    "category": "communication",
                },
            ),
            RagDocument(
                id="pirate_food",
                content=(
                    "Life at sea means limited food choices for pirates. The typical diet includes: "
                    "Hardtack - a dry, hard bread that lasts for months but is difficult to eat. "
                    "Salted meat - usually beef or pork preserved in salt to prevent spoilage. "
                    "Dried fish - cod or herring that can survive long voyages. "
                    "Rum - the preferred drink, made from sugarcane. Pirates drink it straight or make grog with water. "
                    "Lime juice - essential to prevent scurvy disease. "
                    "Fresh food like fruits and vegetables are rare after the first weeks at sea. "
                    "Many pirates suffer from malnutrition, scurvy, and food poisoning from spoiled provisions."
                ),
                metadata={
                    "source": "pirate_lore",
                    "topic": "daily_life",
                    "category": "provisions",
                },
            ),
            RagDocument(
                id="pirate_weapons",
                content=(
                    "Pirates favor close combat weapons suited for ship battles. The cutlass is the preferred sword - "
                    "short, sturdy, and perfect for tight spaces. Flintlock pistols are carried for ranged combat, "
                    "though they can only fire one shot before reloading. The famous Blackbeard carried six pistols "
                    "on a bandolier across his chest for rapid firing. Boarding axes are used to cut enemy rigging "
                    "and damage sails. Grappling hooks allow pirates to board enemy vessels. "
                    "Cannons loaded with chain shot (two balls connected by chain) can destroy masts and rigging. "
                    "Hand grenades made from hollow iron balls filled with gunpowder create chaos during boarding."
                ),
                metadata={
                    "source": "pirate_lore",
                    "topic": "weapons",
                    "category": "combat",
                },
            ),
        ]

    async def get_rag_pipeline(self, memory_id: str | None = None) -> "BM25RAG | None":
        """
        Build BM25 RAG pipeline with example documents.
        """
        documents = self.get_example_documents()
        return BM25RAG(documents=documents, config=BM25Config(top_k=2))

    async def get_pipeline_adapter(self, thread_id: str | None = None) -> "OpenAIAdapter":
        """
        Create OpenAI adapter with RAG support.
        """

        storage_adapter = await self.get_storage_adapter(thread_id)
        rag = await self.rag_provider.get_rag_instance(self, None)

        return OpenAIAdapter(
            client=AsyncOpenAI(
                api_key=settings.OPENAI_API_KEY,
                base_url=settings.OPENAI_API_URL,
            ),
            instructions=self.get_instructions(),
            model=self.get_model(),
            store=True,
            storage_adapter=storage_adapter,
            rag_pipeline=rag,
        )
