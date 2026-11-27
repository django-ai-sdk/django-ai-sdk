import uuid

import pytest
from django_ai_sdk import Agent
from django_ai_sdk.agents.registry import (
    AgentRegistry,
    AgentRegistrationError,
    auto_register,
    registry,
)


def is_valid_uuid(value: str) -> bool:
    try:
        uuid.UUID(value)
        return True
    except ValueError:
        return False


class TestAgentRegistry:
    @pytest.fixture(autouse=True)
    def reset_registry(self):
        registry._reset()
        yield
        registry._reset()

    def test_an_agent_without_name_or_description_still_describes_itself(self):
        class Bare(Agent):
            async def get_pipeline_adapter(self, thread_id=None):
                pass

        agent = Bare()

        assert agent.get_name() == "Unnamed Agent"
        assert agent.description is None
        assert agent.info().name == "Unnamed Agent"

    def test_the_storage_attribute_picks_the_storage_adapter(self):
        from django_ai_sdk.storage.db import DbStorageAdapter

        class Persistent(Agent):
            storage = DbStorageAdapter

            async def get_pipeline_adapter(self, thread_id=None):
                pass

        assert Persistent().storage_adapter is DbStorageAdapter

    def test_auto_registration(self):
        @auto_register
        class TestBot(Agent):
            name = "Test Bot"

            async def get_pipeline_adapter(self, thread_id=None):
                pass

        ids = registry.ids()
        assert len(ids) == 1
        assert is_valid_uuid(ids[0])
        assert ids[0] in registry

    def test_id_is_deterministic_uuid_v5(self):
        registry._reset()

        @auto_register
        class DeterministicAgent(Agent):
            async def get_pipeline_adapter(self, thread_id=None):
                pass

        first_id = DeterministicAgent._agent_id

        registry._reset()

        @auto_register
        class DeterministicAgent(Agent):
            async def get_pipeline_adapter(self, thread_id=None):
                pass

        second_id = DeterministicAgent._agent_id

        assert first_id == second_id
        assert is_valid_uuid(first_id)

    def test_different_classes_get_different_ids(self):
        @auto_register
        class FirstAgent(Agent):
            async def get_pipeline_adapter(self, thread_id=None):
                pass

        @auto_register
        class SecondAgent(Agent):
            async def get_pipeline_adapter(self, thread_id=None):
                pass

        ids = registry.ids()
        assert len(ids) == 2
        assert ids[0] != ids[1]
        assert all(is_valid_uuid(id_) for id_ in ids)

    def test_id_override_allowed(self):
        class OverrideAgent(Agent):
            async def get_pipeline_adapter(self, thread_id=None):
                pass

        first_id = OverrideAgent._agent_id
        registry.register(OverrideAgent)

        class OverrideAgent(Agent):
            async def get_pipeline_adapter(self, thread_id=None):
                pass

        second_id = OverrideAgent._agent_id
        assert first_id == second_id

        result = registry.register(OverrideAgent)
        assert result is OverrideAgent
        assert first_id in registry.ids()

    def test_base_agent_not_registered(self):
        assert len(registry.ids()) == 0

    def test_setup_instantiates_agents(self):
        @auto_register
        class SetupAgent(Agent):
            async def get_pipeline_adapter(self, thread_id=None):
                pass

        agent_id = SetupAgent._agent_id
        registry.setup()

        agent = registry.get(agent_id)
        assert agent is not None
        assert isinstance(agent, SetupAgent)
        assert agent.agent_id == agent_id

    def test_setup_is_idempotent(self):
        @auto_register
        class IdempotentAgent(Agent):
            async def get_pipeline_adapter(self, thread_id=None):
                pass

        agent_id = IdempotentAgent._agent_id
        registry.setup()
        first_instance = registry.get(agent_id)

        registry.setup()
        second_instance = registry.get(agent_id)

        assert first_instance is second_instance

    def test_a_wrong_path_in_the_agents_setting_is_refused(self, settings):
        from django.core.exceptions import ImproperlyConfigured

        settings.AI_SDK_AGENTS = ["myapp.agents.Missing"]

        with pytest.raises(ImproperlyConfigured, match="myapp"):
            registry.setup()

    def test_get_before_setup_raises(self):
        @auto_register
        class EarlyAgent(Agent):
            async def get_pipeline_adapter(self, thread_id=None):
                pass

        agent_id = EarlyAgent._agent_id

        with pytest.raises(RuntimeError) as exc_info:
            registry.get(agent_id)

        assert "not initialized" in str(exc_info.value).lower()

    def test_get_unknown_returns_none(self):
        @auto_register
        class KnownAgent(Agent):
            async def get_pipeline_adapter(self, thread_id=None):
                pass

        registry.setup()

        unknown_id = str(uuid.uuid4())
        assert registry.get(unknown_id) is None

    def test_all_returns_all_instances(self):
        @auto_register
        class FirstAgent(Agent):
            async def get_pipeline_adapter(self, thread_id=None):
                pass

        @auto_register
        class SecondAgent(Agent):
            async def get_pipeline_adapter(self, thread_id=None):
                pass

        registry.setup()

        all_agents = registry.all()
        assert len(all_agents) == 2
        for agent_id, agent in all_agents.items():
            assert is_valid_uuid(agent_id)
            assert isinstance(agent, Agent)

    def test_all_before_setup_raises(self):
        @auto_register
        class AnyAgent(Agent):
            async def get_pipeline_adapter(self, thread_id=None):
                pass

        with pytest.raises(RuntimeError):
            registry.all()

    def test_ids_available_before_setup(self):
        @auto_register
        class PreSetupAgent(Agent):
            async def get_pipeline_adapter(self, thread_id=None):
                pass

        agent_id = PreSetupAgent._agent_id

        assert agent_id in registry.ids()
        assert is_valid_uuid(agent_id)

        with pytest.raises(RuntimeError):
            registry.get(agent_id)

    def test_in_operator(self):
        @auto_register
        class InOperatorAgent(Agent):
            async def get_pipeline_adapter(self, thread_id=None):
                pass

        agent_id = InOperatorAgent._agent_id

        assert agent_id in registry
        assert "not-a-real-uuid" not in registry

    def test_agent_id_set_on_class(self):
        @auto_register
        class IdAgent(Agent):
            async def get_pipeline_adapter(self, thread_id=None):
                pass

        assert is_valid_uuid(IdAgent._agent_id)

    def test_singleton_instance(self):
        reg1 = AgentRegistry()
        reg2 = AgentRegistry()
        assert reg1 is reg2
        assert reg1 is registry


class TestAgentInfoMixin:
    @pytest.fixture(autouse=True)
    def reset_registry(self):
        registry._reset()
        yield
        registry._reset()

    def test_info_returns_metadata(self):
        @auto_register
        class InfoAgent(Agent):
            name = "Info Bot"
            model = "gpt-4"
            description = "An info bot"

            async def get_pipeline_adapter(self, thread_id=None):
                pass

        agent_id = InfoAgent._agent_id
        registry.setup()
        agent = registry.get(agent_id)

        info = agent.info()
        assert info.id == agent_id
        assert is_valid_uuid(info.id)
        assert info.name == "Info Bot"
        assert info.model == "gpt-4"
        assert info.description == "An info bot"
        assert info.class_name == "InfoAgent"

    def test_agent_id_property(self):
        @auto_register
        class PropAgent(Agent):
            async def get_pipeline_adapter(self, thread_id=None, run_id=None):
                pass

        registry.setup()
        agent_id = PropAgent._agent_id
        agent = registry.get(agent_id)

        assert agent.agent_id == agent_id
        assert is_valid_uuid(agent.agent_id)

    def test_agent_id_fallback(self):
        @auto_register
        class FallbackAgent(Agent):
            async def get_pipeline_adapter(self, thread_id=None, run_id=None):
                pass

        agent = FallbackAgent()

        first_id = agent.agent_id
        assert is_valid_uuid(first_id)

        second_id = agent.agent_id
        assert first_id == second_id

    def test_info_with_get_name_method(self):
        @auto_register
        class DynamicNameAgent(Agent):
            name = None

            def get_name(self):
                return "Dynamic Name"

            async def get_pipeline_adapter(self, thread_id=None, run_id=None):
                pass

        agent_id = DynamicNameAgent._agent_id
        registry.setup()
        agent = registry.get(agent_id)

        info = agent.info()
        assert info.name == "Dynamic Name"
        assert is_valid_uuid(info.id)

    def test_info_rag_false_without_provider(self):
        @auto_register
        class NoRagAgent(Agent):
            name = "No RAG"

            async def get_pipeline_adapter(self, thread_id=None, run_id=None):
                pass

        registry.setup()
        agent = registry.get(NoRagAgent._agent_id)

        info = agent.info()
        assert info.rag is False

    def test_info_rag_true_with_provider(self):
        from django_ai_sdk.rags.provider import RAGProvider

        @auto_register
        class RagAgent(Agent):
            name = "Has RAG"
            rag_provider = RAGProvider()

            async def get_pipeline_adapter(self, thread_id=None, run_id=None):
                pass

        registry.setup()
        agent = registry.get(RagAgent._agent_id)

        info = agent.info()
        assert info.rag is True


class TestAbstractAgents:
    @pytest.fixture(autouse=True)
    def reset_registry(self):
        registry._reset()
        yield
        registry._reset()

    def test_an_abstract_base_is_not_registered(self):
        class SharedBase(Agent):
            abstract = True
            name = "Shared Base"

        assert SharedBase._agent_id == ""
        assert SharedBase not in registry._classes.values()

    def test_a_concrete_subclass_of_an_abstract_base_is_registered(self):
        class SharedBase(Agent):
            abstract = True
            name = "Shared Base"

        class RealBot(SharedBase):
            name = "Real Bot"

        assert RealBot._agent_id
        assert registry._classes[RealBot._agent_id] is RealBot

    def test_abstract_also_blocks_explicit_registration(self):
        @auto_register
        class DecoratedBase(Agent):
            abstract = True
            name = "Decorated Base"

        assert DecoratedBase not in registry._classes.values()

    def test_setup_does_not_instantiate_an_abstract_base(self):
        class SharedBase(Agent):
            abstract = True
            name = "Shared Base"

        class RealBot(SharedBase):
            name = "Real Bot"

        registry.setup()

        assert [type(i) for i in registry._instances.values()] == [RealBot]
 
