"""Declaring workflows, and the wiring rules that catch a typo before a worker does."""

from __future__ import annotations

import pytest
from django.core.exceptions import ImproperlyConfigured
from django.db import OperationalError

from django_ai_sdk.workflows import (
    WorkflowAction,
    WorkflowDefinition,
    WorkflowSettings,
    WorkflowStep,
)
from django_ai_sdk.workflows.checks import check_workflows
from django_ai_sdk.workflows.registry import (
    aget_workflow,
    aget_workflows,
    get_declared_workflows,
    get_invalid_workflows,
    register,
    reset_registry,
    validate,
    validate_name,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    reset_registry()
    yield
    reset_registry()


def definition(name="example", **kwargs) -> WorkflowDefinition:
    kwargs.setdefault("steps", [WorkflowStep(agent_id="a", output_key="result")])
    return WorkflowDefinition(name=name, **kwargs)


class TestRegistration:
    def test_registering_makes_it_reachable_by_name(self):
        register(definition("digest"))
        assert set(get_declared_workflows()) == {"digest"}

    def test_register_returns_the_definition_untouched(self):
        original = definition("digest")
        assert register(original) is original

    def test_registering_the_same_definition_twice_is_not_a_collision(self, caplog):
        register(definition("digest"))
        with caplog.at_level("WARNING"):
            register(definition("digest"))
        assert "declared twice" not in caplog.text

    def test_a_different_definition_under_one_name_warns(self, caplog):
        register(definition("digest"))
        with caplog.at_level("WARNING"):
            register(definition("digest", steps=[WorkflowStep(agent_id="b", output_key="r")]))
        assert "declared twice" in caplog.text

    def test_an_unknown_name_is_absent(self):
        assert "nope" not in get_declared_workflows()


class TestABrokenDeclarationDoesNotStopTheSite:
    """A typo in one app's workflows module must not take Django's boot with it."""

    def test_it_does_not_raise(self, caplog):
        with caplog.at_level("WARNING"):
            register(definition("broken", steps=[]))
        assert "not registered" in caplog.text

    def test_it_is_not_reachable(self):
        register(definition("broken", steps=[]))
        assert get_declared_workflows() == {}

    def test_it_is_recorded_for_the_system_check(self):
        register(definition("broken", steps=[]))
        assert "no steps" in get_invalid_workflows()["broken"]

    def test_the_system_check_reports_it_as_an_error(self):
        register(definition("broken", steps=[]))
        errors = check_workflows()
        assert [e.id for e in errors] == ["ai_sdk.workflows.E001"]
        assert "no steps" in errors[0].msg

    def test_the_system_check_is_silent_when_every_declaration_is_valid(self):
        register(definition("fine"))
        assert check_workflows() == []

    def test_registering_a_fixed_definition_clears_the_error(self):
        register(definition("digest", steps=[]))
        register(definition("digest"))
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
    def test_at_least_one_step(self):
        with pytest.raises(ImproperlyConfigured, match="no steps"):
            validate(WorkflowDefinition(name="empty", steps=[]))

    def test_a_step_needs_an_agent_id(self):
        with pytest.raises(ImproperlyConfigured, match="no `agent_id`"):
            validate(definition(steps=[WorkflowStep(agent_id="", output_key="result")]))

    def test_a_step_needs_an_output_key(self):
        with pytest.raises(ImproperlyConfigured, match="output_key"):
            validate(definition(steps=[WorkflowStep(agent_id="a", output_key="")]))

    def test_a_duplicate_output_key_would_overwrite(self):
        with pytest.raises(ImproperlyConfigured, match="reuses"):
            validate(
                definition(
                    steps=[
                        WorkflowStep(agent_id="a", output_key="same"),
                        WorkflowStep(agent_id="b", output_key="same"),
                    ]
                )
            )

    def test_a_step_reading_an_output_no_earlier_step_produces(self):
        with pytest.raises(ImproperlyConfigured, match="which no earlier step produces"):
            validate(
                definition(
                    steps=[
                        WorkflowStep(agent_id="a", output_key="summary"),
                        WorkflowStep(agent_id="b", input_key="sumary", output_key="verdict"),
                    ]
                )
            )

    def test_a_step_cannot_read_its_own_output(self):
        with pytest.raises(ImproperlyConfigured, match="no earlier step produces"):
            validate(
                definition(steps=[WorkflowStep(agent_id="a", input_key="r", output_key="r")])
            )

    def test_a_step_cannot_read_a_later_step(self):
        with pytest.raises(ImproperlyConfigured, match="no earlier step produces"):
            validate(
                definition(
                    steps=[
                        WorkflowStep(agent_id="a", input_key="later", output_key="first"),
                        WorkflowStep(agent_id="b", output_key="later"),
                    ]
                )
            )

    def test_an_action_reading_an_output_nothing_produces(self):
        with pytest.raises(ImproperlyConfigured, match="which no step produces"):
            validate(
                definition(actions=[WorkflowAction(type="thread_message", input_key="nope")])
            )

    def test_a_valid_chain_passes(self):
        validate(
            definition(
                steps=[
                    WorkflowStep(agent_id="a", output_key="summary"),
                    WorkflowStep(agent_id="b", input_key="summary", output_key="verdict"),
                ],
                actions=[WorkflowAction(type="thread_message", input_key="verdict")],
            )
        )

    def test_an_action_with_no_input_key_takes_everything(self):
        validate(definition(actions=[WorkflowAction(type="thread_message")]))

    def test_a_stored_definition_needs_no_name(self):
        # A row is keyed by its `slug` column, so the name inside the JSON is unused.
        validate(definition(""))


# transaction=True throughout: an async ORM write is not rolled back by plain django_db.
@pytest.mark.django_db(transaction=True)
class TestDatabaseMerge:
    async def test_a_row_is_reachable_by_slug(self):
        await WorkflowSettings.objects.acreate(
            name="Weekly Review", definition=definition("weekly").model_dump()
        )
        assert await aget_workflow("weekly-review") is not None

    async def test_an_inactive_row_is_not(self):
        await WorkflowSettings.objects.acreate(
            name="Retired", definition=definition("retired").model_dump(), active=False
        )
        assert await aget_workflow("retired") is None

    async def test_code_wins_a_slug_collision(self, caplog):
        register(definition("shared"))
        await WorkflowSettings.objects.acreate(
            name="shared",
            definition=definition(
                "shared", steps=[WorkflowStep(agent_id="db", output_key="r")]
            ).model_dump(),
        )

        # A database row adds a workflow where there is no code; it never overrides one.
        assert (await aget_workflow("shared")).steps[0].agent_id == "a"

        # The single lookup short-circuits on the registry and never reads the row.
        with caplog.at_level("WARNING"):
            merged = await aget_workflows()
        assert merged["shared"].steps[0].agent_id == "a"
        assert "shadowed" in caplog.text

    async def test_the_merged_view_holds_both(self):
        register(definition("from-code"))
        await WorkflowSettings.objects.acreate(
            name="from-db", definition=definition("from-db").model_dump()
        )

        assert set(await aget_workflows()) == {"from-code", "from-db"}

    async def test_the_sync_view_never_touches_the_database(self):
        await WorkflowSettings.objects.acreate(
            name="from-db", definition=definition("from-db").model_dump()
        )
        # Management commands and the system check rely on this staying code-only.
        assert get_declared_workflows() == {}


@pytest.mark.django_db(transaction=True)
class TestABrokenRowIsSkipped:
    """A definition edited into an unrunnable state must not take out the rest."""

    async def test_an_unparseable_row_is_skipped(self):
        register(definition("healthy"))
        await WorkflowSettings.objects.acreate(name="broken", definition={"nonsense": True})

        assert set(await aget_workflows()) == {"healthy"}

    async def test_a_broken_input_chain_is_rejected(self):
        # Rows get the same wiring checks a declaration does, and need them more: this
        # one was filled in through a form, not written and reviewed.
        broken = WorkflowDefinition(
            name="broken",
            steps=[
                WorkflowStep(agent_id="a", output_key="summary"),
                WorkflowStep(agent_id="b", input_key="sumary", output_key="verdict"),
            ],
        )
        await WorkflowSettings.objects.acreate(name="broken", definition=broken.model_dump())

        assert await aget_workflow("broken") is None

    async def test_it_is_logged_once_rather_than_on_every_dispatch(self, caplog):
        await WorkflowSettings.objects.acreate(name="broken", definition={"nonsense": True})

        with caplog.at_level("WARNING"):
            await aget_workflows()
            await aget_workflows()
        assert caplog.text.count("cannot run") == 1

    async def test_a_valid_row_still_resolves(self):
        await WorkflowSettings.objects.acreate(
            name="fine", definition=definition("fine").model_dump()
        )
        assert await aget_workflow("fine") is not None


@pytest.mark.django_db(transaction=True)
class TestAnUnreadableDatabaseIsNotSilence:
    """An outage reaches the caller, rather than reading as "no such workflow"."""

    async def test_it_propagates(self, monkeypatch):
        # A dispatch that swallowed this would skip a run a retry would have completed.
        def boom(*args, **kwargs):
            raise OperationalError("connection gone")

        monkeypatch.setattr(WorkflowSettings.objects, "filter", boom)
        register(definition("from-code"))

        with pytest.raises(OperationalError):
            await aget_workflows()


@pytest.mark.django_db(transaction=True)
class TestSlug:
    def test_it_is_derived_from_the_name(self):
        row = WorkflowSettings.objects.create(name="Weekly Review", definition={})
        assert row.slug == "weekly-review"

    def test_a_colliding_name_gets_a_suffix(self):
        # Names are not unique and the API takes free text, so this has to be a working
        # create rather than an integrity error.
        first = WorkflowSettings.objects.create(name="Weekly Review", definition={})
        second = WorkflowSettings.objects.create(name="Weekly Review", definition={})
        assert (first.slug, second.slug) == ("weekly-review", "weekly-review-2")

    def test_an_explicit_slug_is_kept(self):
        row = WorkflowSettings.objects.create(name="Weekly Review", slug="wr", definition={})
        assert row.slug == "wr"


@pytest.mark.django_db(transaction=True)
class TestRowsAreReadFresh:
    """A change to a row has to be visible on the next dispatch, not the one after."""

    async def test_deleting_a_row_takes_effect_immediately(self):
        row = await WorkflowSettings.objects.acreate(
            name="temporary", definition=definition("temporary").model_dump()
        )
        assert await aget_workflow("temporary") is not None

        await WorkflowSettings.objects.filter(pk=row.pk).adelete()

        assert await aget_workflow("temporary") is None

    async def test_a_row_added_later_is_picked_up(self):
        await aget_workflows()
        await WorkflowSettings.objects.acreate(
            name="added later", definition=definition("added").model_dump()
        )

        assert await aget_workflow("added-later") is not None
