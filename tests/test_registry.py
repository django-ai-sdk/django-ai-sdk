"""Tests for the AgentRegistry."""

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
    """Check if a string is a valid UUID."""
    try:
        uuid.UUID(value)
        return True
    except ValueError:
        return False


class TestAgentRegistry:
    """Test suite for AgentRegistry singleton."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset global registry singleton before each test."""
        registry._reset()
        yield
        registry._reset()

    def test_auto_registration(self):
        """Test that Agent subclasses are auto-registered with UUID v5 IDs."""

        @auto_register
        class TestBot(Agent):
            name = "Test Bot"

            async def get_pipeline_adapter(self, thread_id=None):
                pass

        # Check that an ID was registered and it's a valid UUID
        ids = registry.ids()
        assert len(ids) == 1
        assert is_valid_uuid(ids[0])
        assert ids[0] in registry

    def test_id_is_deterministic_uuid_v5(self):
        """Test that the same class always generates the same UUID v5 ID."""
        registry._reset()

        # First registration
        @auto_register
        class DeterministicAgent(Agent):
            async def get_pipeline_adapter(self, thread_id=None):
                pass

        first_id = DeterministicAgent._agent_id

        # Reset and re-register (simulating restart)
        registry._reset()

        # Re-define same class with same decorator
        @auto_register
        class DeterministicAgent(Agent):  # noqa: F811
            async def get_pipeline_adapter(self, thread_id=None):
                pass

        second_id = DeterministicAgent._agent_id

        # Same class path should generate same UUID
        assert first_id == second_id
        assert is_valid_uuid(first_id)

    def test_different_classes_get_different_ids(self):
        """Test that different classes get unique UUIDs."""

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
        """Test that a different class with same ID can override (for testing)."""
        # This allows test classes with same names to work properly

        class OverrideAgent(Agent):
            async def get_pipeline_adapter(self, thread_id=None):
                pass

        first_id = OverrideAgent._agent_id
        registry.register(OverrideAgent)

        # Create another class with same module.name (simulates test scenario)
        class OverrideAgent(Agent):  # noqa: F811
            async def get_pipeline_adapter(self, thread_id=None):
                pass

        second_id = OverrideAgent._agent_id
        # Should be same ID (based on module.name)
        assert first_id == second_id

        # Register again - should succeed (replaces previous)
        result = registry.register(OverrideAgent)
        assert result is OverrideAgent
        assert first_id in registry.ids()

    def test_base_agent_not_registered(self):
        """Test that the base Agent class is not auto-registered."""
        # The base Agent class should not appear in registry
        assert len(registry.ids()) == 0

    def test_setup_instantiates_agents(self):
        """Test that setup() creates instances of all agents."""

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
        """Test that calling setup() multiple times doesn't duplicate."""

        @auto_register
        class IdempotentAgent(Agent):
            async def get_pipeline_adapter(self, thread_id=None):
                pass

        agent_id = IdempotentAgent._agent_id
        registry.setup()
        first_instance = registry.get(agent_id)

        registry.setup()  # Call again
        second_instance = registry.get(agent_id)

        assert first_instance is second_instance

    def test_get_before_setup_raises(self):
        """Test that get() before setup() raises RuntimeError."""

        @auto_register
        class EarlyAgent(Agent):
            async def get_pipeline_adapter(self, thread_id=None):
                pass

        agent_id = EarlyAgent._agent_id

        with pytest.raises(RuntimeError) as exc_info:
            registry.get(agent_id)

        assert "not initialized" in str(exc_info.value).lower()

    def test_get_unknown_returns_none(self):
        """Test that get() returns None for unknown ID."""

        @auto_register
        class KnownAgent(Agent):
            async def get_pipeline_adapter(self, thread_id=None):
                pass

        registry.setup()

        # Random valid UUID that doesn't exist
        unknown_id = str(uuid.uuid4())
        assert registry.get(unknown_id) is None

    def test_all_returns_all_instances(self):
        """Test that all() returns dict of all instances."""

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
        # All values should be instances, all keys should be valid UUIDs
        for agent_id, agent in all_agents.items():
            assert is_valid_uuid(agent_id)
            assert isinstance(agent, Agent)

    def test_all_before_setup_raises(self):
        """Test that all() before setup() raises RuntimeError."""

        @auto_register
        class AnyAgent(Agent):
            async def get_pipeline_adapter(self, thread_id=None):
                pass

        with pytest.raises(RuntimeError):
            registry.all()

    def test_ids_available_before_setup(self):
        """Test that ids() works even before setup()."""

        @auto_register
        class PreSetupAgent(Agent):
            async def get_pipeline_adapter(self, thread_id=None):
                pass

        agent_id = PreSetupAgent._agent_id

        # Can get IDs before setup
        assert agent_id in registry.ids()
        assert is_valid_uuid(agent_id)

        # But can't get instances
        with pytest.raises(RuntimeError):
            registry.get(agent_id)

    def test_in_operator(self):
        """Test the 'in' operator for checking registration."""

        @auto_register
        class InOperatorAgent(Agent):
            async def get_pipeline_adapter(self, thread_id=None):
                pass

        agent_id = InOperatorAgent._agent_id

        assert agent_id in registry
        assert "not-a-real-uuid" not in registry

    def test_agent_id_set_on_class(self):
        """Test that _agent_id is set on the registered class."""

        @auto_register
        class IdAgent(Agent):
            async def get_pipeline_adapter(self, thread_id=None):
                pass

        assert is_valid_uuid(IdAgent._agent_id)

    def test_singleton_instance(self):
        """Test that registry is a singleton."""
        reg1 = AgentRegistry()
        reg2 = AgentRegistry()
        assert reg1 is reg2
        assert reg1 is registry


class TestAgentInfoMixin:
    """Test suite for AgentInfoMixin."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset global registry singleton before each test."""
        registry._reset()
        yield
        registry._reset()

    def test_info_returns_metadata(self):
        """Test that info() returns correct metadata with UUID id."""

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
        """Test the agent_id property returns UUID."""

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
        """Test agent_id fallback generates consistent UUID."""

        # Create instance without going through registry
        @auto_register
        class FallbackAgent(Agent):
            async def get_pipeline_adapter(self, thread_id=None, run_id=None):
                pass

        # Manually create instance (won't have _agent_id cached yet)
        agent = FallbackAgent()

        # Should generate UUID from class path
        first_id = agent.agent_id
        assert is_valid_uuid(first_id)

        # Second call should return cached value
        second_id = agent.agent_id
        assert first_id == second_id

    def test_info_with_get_name_method(self):
        """Test info() uses get_name() when name attribute is None."""

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
        """Test info().rag is False when no rag_provider set."""

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
        """Test info().rag is True when rag_provider is set."""

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
    """`abstract = True` marks a shared base meant only to be subclassed. It exists so
    a project can factor common config (model, permissions, storage) into a base class
    without that base showing up as a usable agent.
    """

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        registry._reset()
        yield
        registry._reset()

    def test_an_abstract_base_is_not_registered(self):
        class SharedBase(Agent):
            abstract = True
            name = "Shared Base"

        # register() never ran, so the class-level default is untouched.
        assert SharedBase._agent_id == ""
        assert SharedBase not in registry._classes.values()

    def test_a_concrete_subclass_of_an_abstract_base_is_registered(self):
        """`abstract` is read off the class's own __dict__, so subclasses don't inherit
        it and don't have to restate `abstract = False`."""

        class SharedBase(Agent):
            abstract = True
            name = "Shared Base"

        class RealBot(SharedBase):
            name = "Real Bot"

        assert RealBot._agent_id
        assert registry._classes[RealBot._agent_id] is RealBot

    def test_abstract_also_blocks_explicit_registration(self):
        """Checked in register() rather than __init_subclass__, so @auto_register and
        AI_SDK_AGENTS skip it too — one rule, every path."""

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
 
