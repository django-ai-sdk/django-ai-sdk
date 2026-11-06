"""Declaring workflows, and the wiring rules that catch a typo before a worker does."""

from __future__ import annotations

import pytest
from django.core.exceptions import ImproperlyConfigured
from django.test import override_settings

from django_ai_sdk.workflows.listeners import RunListener
from django_ai_sdk.workflows.checks import check_workflows
from django_ai_sdk.workflows.models import Workflow
from django_ai_sdk.workflows.schemas import (
    ListenerDefinition,
    FieldDefinition,
    StepDefinition,
    WorkflowDefinition,
)
from django_ai_sdk.workflows.registry import (
    aget_workflow,
    get_invalid_workflows,
    is_registered,
    register_workflow,
    reset_registry,
    validate_definition,
    validate_name,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    reset_registry()
    yield
    reset_registry()


LISTENERS = {"loud": "tests.unit.test_workflow_registry.LoudListener"}


class LoudListener(RunListener):
    pass


def definition(name="example", **kwargs) -> WorkflowDefinition:
    kwargs.setdefault("steps", [StepDefinition(name="result", agent_id="a")])
    return WorkflowDefinition(name=name, **kwargs)


class TestRegistration:
    def test_registering_makes_it_reachable_by_name(self):
        assert is_registered(register_workflow(definition("digest")))

    def test_register_returns_the_definition_untouched(self):
        original = definition("digest")
        assert register_workflow(original) is original

    def test_registering_the_same_definition_twice_is_not_a_collision(self, caplog):
        register_workflow(definition("digest"))
        with caplog.at_level("WARNING"):
            register_workflow(definition("digest"))
        assert "declared twice" not in caplog.text

    def test_a_different_definition_under_one_name_warns(self, caplog):
        register_workflow(definition("digest"))
        with caplog.at_level("WARNING"):
            register_workflow(definition("digest", steps=[StepDefinition(name="r", agent_id="b")]))
        assert "declared twice" in caplog.text


class TestABrokenDeclarationDoesNotStopTheSite:
    """A typo in one app's workflows module must not take Django's boot with it."""

    def test_it_does_not_raise(self, caplog):
        with caplog.at_level("WARNING"):
            register_workflow(definition("broken", steps=[]))
        assert "not registered" in caplog.text

    def test_it_is_not_reachable(self):
        assert not is_registered(register_workflow(definition("broken", steps=[])))

    def test_it_is_recorded_for_the_system_check(self):
        register_workflow(definition("broken", steps=[]))
        assert "no steps" in get_invalid_workflows()["broken"]

    def test_the_system_check_reports_it_as_an_error(self):
        register_workflow(definition("broken", steps=[]))
        errors = check_workflows()
        assert [e.id for e in errors] == ["ai_sdk.workflows.E001"]
        assert "no steps" in errors[0].msg

    def test_the_system_check_is_silent_when_every_declaration_is_valid(self):
        register_workflow(definition("fine"))
        assert check_workflows() == []

    def test_registering_a_fixed_definition_clears_the_error(self):
        register_workflow(definition("digest", steps=[]))
        register_workflow(definition("digest"))
        assert get_invalid_workflows() == {}


class TestTheNameIsTheRegistryKey:
    def test_a_name_is_required(self):
        with pytest.raises(ImproperlyConfigured, match="non-empty `name`"):
            validate_name("")

    def test_a_slug_is_accepted(self):
        validate_name("weekly-triage")

    def test_a_non_slug_name_is_rejected_and_the_message_carries_the_slug(self):
        # Held to slug form so a declaration and a row cannot sit under two keys.
        with pytest.raises(ImproperlyConfigured, match="'weekly-review'"):
            validate_name("Weekly Review")

    def test_a_name_longer_than_the_slug_column_is_rejected(self):
        with pytest.raises(ImproperlyConfigured, match="not a slug"):
            validate_name("a" * 101)


class TestValidation:
    """What one step cannot see alone: the registries, and its neighbours' names."""

    def test_at_least_one_step(self):
        with pytest.raises(ImproperlyConfigured, match="no steps"):
            validate_definition(WorkflowDefinition(name="empty", steps=[]))

    def test_a_step_needs_an_agent_id(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="agent_id"):
            StepDefinition(name="result", agent_id="")

    def test_a_registered_step_takes_no_output_fields(self):
        """Only an agent step runs with structured output; declaring it elsewhere is dropped."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="output_fields"):
            StepDefinition(
                name="gather", type="gather_thread", output_fields={"title": FieldDefinition()}
            )

    def test_a_step_needs_a_name(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            StepDefinition(name="", agent_id="a")

    def test_a_duplicate_step_name_is_refused(self):
        """The one name rule left: a step's name is the key its result is filed under."""
        with pytest.raises(ImproperlyConfigured, match="duplicate step name"):
            validate_definition(
                definition(
                    steps=[
                        StepDefinition(name="same", agent_id="a"),
                        StepDefinition(name="same", agent_id="b"),
                    ]
                )
            )

    def test_a_step_may_not_require_a_name_no_earlier_step_produces(self):
        """The typo case, caught where it is written rather than by a worker."""
        with pytest.raises(ImproperlyConfigured, match="sumary"):
            validate_definition(
                definition(
                    steps=[
                        StepDefinition(name="summary", agent_id="a"),
                        StepDefinition(name="verdict", agent_id="b", requires=["sumary"]),
                    ]
                )
            )

    def test_a_step_may_not_require_a_later_one(self):
        """Order is the author's: a list that cannot run says so up front."""
        with pytest.raises(ImproperlyConfigured, match="later"):
            validate_definition(
                definition(
                    steps=[
                        StepDefinition(name="first", agent_id="a", requires=["later"]),
                        StepDefinition(name="later", agent_id="b"),
                    ]
                )
            )

    def test_a_step_name_may_match_a_run_input_name(self):
        """Two namespaces, so neither has to be renamed for the other's sake."""
        validate_definition(
            definition(
                input_fields={"summary": FieldDefinition()},
                steps=[StepDefinition(name="summary", agent_id="a")],
            )
        )

    def test_an_unregistered_listener_type_is_refused(self):
        with pytest.raises(ImproperlyConfigured, match="not in AI_SDK_WORKFLOWS"):
            validate_definition(definition(listeners=[ListenerDefinition(type="carrier_pigeon")]))

    def test_an_unregistered_listener_on_a_step_is_refused(self):
        with pytest.raises(ImproperlyConfigured, match="not in AI_SDK_WORKFLOWS"):
            validate_definition(
                definition(
                    steps=[
                        StepDefinition(
                            name="x", agent_id="a", listeners=[ListenerDefinition(type="carrier_pigeon")]
                        )
                    ]
                )
            )

    @override_settings(AI_SDK_WORKFLOWS={"LISTENERS": LISTENERS})
    def test_a_registered_listener_is_accepted_in_both_places(self):
        validate_definition(
            definition(
                steps=[StepDefinition(name="x", agent_id="a", listeners=[ListenerDefinition(type="loud")])],
                listeners=[ListenerDefinition(type="loud")],
            )
        )

    @override_settings(AI_SDK_WORKFLOWS={"LISTENERS": LISTENERS})
    def test_a_valid_chain_passes(self):
        validate_definition(
            definition(
                input_fields={"history": FieldDefinition(type="list")},
                steps=[
                    StepDefinition(name="summary", agent_id="a", history=["history"]),
                    StepDefinition(name="verdict", agent_id="b", requires=["summary"]),
                ],
                listeners=[ListenerDefinition(type="loud", config={"step": "verdict"})],
            )
        )

    def test_history_a_definition_does_not_declare_is_refused(self):
        with pytest.raises(ImproperlyConfigured, match="does not declare"):
            validate_definition(
                definition(steps=[StepDefinition(name="x", agent_id="a", history=["history"])])
            )

    def test_input_fields_that_cannot_compile_are_refused(self):
        """`validate_definition` promises a check where it is written, so it builds the model."""
        with pytest.raises(ImproperlyConfigured, match="input_fields cannot compile"):
            validate_definition(definition(input_fields={"model_dump": FieldDefinition()}))

    def test_output_fields_that_cannot_compile_are_refused(self):
        with pytest.raises(ImproperlyConfigured, match="output_fields cannot compile"):
            validate_definition(
                definition(
                    steps=[
                        StepDefinition(
                            name="x", agent_id="a", output_fields={"model_dump": FieldDefinition()}
                        )
                    ]
                )
            )

    def test_a_stored_definition_needs_no_name(self):
        # A row is keyed by its `slug` column, so the name inside the JSON is unused.
        validate_definition(definition(""))


# transaction=True throughout: an async ORM write is not rolled back by plain django_db.
@pytest.mark.django_db(transaction=True)
class TestStoredWorkflows:
    async def test_a_row_is_reachable_by_slug(self):
        await Workflow.objects.acreate(
            name="Weekly Review", definition=definition("weekly").model_dump()
        )
        assert await aget_workflow("weekly-review") is not None

    def test_one_name_reads_one_row_not_every_row(self):
        from asgiref.sync import async_to_sync
        from django.db import connection
        from django.test.utils import CaptureQueriesContext

        for name in ("one", "two", "three"):
            Workflow.objects.create(name=name, definition=definition(name).model_dump())

        with CaptureQueriesContext(connection) as queries:
            found = async_to_sync(aget_workflow)("two")

        assert found is not None
        assert len(queries) == 1
        assert "LIMIT 1" in queries[0]["sql"]

    async def test_an_inactive_row_is_not(self):
        await Workflow.objects.acreate(
            name="Retired", definition=definition("retired").model_dump(), active=False
        )
        assert await aget_workflow("retired") is None

    async def test_code_wins_a_slug_collision(self):
        register_workflow(definition("shared"))
        await Workflow.objects.acreate(
            name="shared",
            definition=definition(
                "shared", steps=[StepDefinition(name="r", agent_id="db")]
            ).model_dump(),
        )
        # A database row adds a workflow where there is no code; it never overrides one.
        assert (await aget_workflow("shared")).steps[0].agent_id == "a"

    async def test_a_deleted_row_is_gone_on_the_next_lookup(self):
        row = await Workflow.objects.acreate(
            name="temporary", definition=definition("temporary").model_dump()
        )
        assert await aget_workflow("temporary") is not None
        await Workflow.objects.filter(pk=row.pk).adelete()
        assert await aget_workflow("temporary") is None

    async def test_an_unparseable_row_raises(self):
        from pydantic import ValidationError

        await Workflow.objects.acreate(name="broken", definition={"nonsense": True})
        with pytest.raises(ValidationError):
            await aget_workflow("broken")

    async def test_a_row_that_fails_the_wiring_checks_raises(self):
        # Rows get the same wiring checks a declaration does (a duplicate step name).
        broken = definition(
            "broken",
            steps=[
                StepDefinition(name="summary", agent_id="a"),
                StepDefinition(name="summary", agent_id="b"),
            ],
        )
        await Workflow.objects.acreate(name="broken", definition=broken.model_dump())
        with pytest.raises(ImproperlyConfigured, match="duplicate step name"):
            await aget_workflow("broken")


@pytest.mark.django_db(transaction=True)
class TestSlug:
    def test_it_is_derived_from_the_name(self):
        row = Workflow.objects.create(name="Weekly Review", definition={})
        assert row.slug == "weekly-review"

    def test_a_colliding_name_gets_a_suffix(self):
        # Names are not unique and the API takes free text, so this has to be a working
        # create rather than an integrity error.
        first = Workflow.objects.create(name="Weekly Review", definition={})
        second = Workflow.objects.create(name="Weekly Review", definition={})
        assert (first.slug, second.slug) == ("weekly-review", "weekly-review-2")

    def test_an_explicit_slug_is_kept(self):
        row = Workflow.objects.create(name="Weekly Review", slug="wr", definition={})
        assert row.slug == "wr"
