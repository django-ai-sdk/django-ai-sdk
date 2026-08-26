"""Registration-time validation, and the input an automation hands its workflow.

A half-specified declaration stays out of the registry and is named by `manage.py check`.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from django.core.exceptions import ImproperlyConfigured

from django_ai_sdk.automations import Audience, Automation
from django_ai_sdk.automations.registry import (
    get_automation,
    get_automations,
    get_invalid_automations,
    register,
    reset_registry,
    validate,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    reset_registry()
    yield
    reset_registry()


def make(**attrs) -> Automation:
    """An Automation subclass built from keyword attributes, not registered."""
    defaults = {"name": "example", "cron": "0 9 * * *", "workflow": "some-workflow"}
    return type("Example", (Automation,), {**defaults, **attrs})()


class TestValidation:
    def test_registering_makes_it_reachable_by_name(self):
        @register
        class Example(Automation):
            name = "example"
            cron = "0 9 * * *"
            workflow = "some-workflow"

        assert get_automation("example") is not None
        assert set(get_automations()) == {"example"}

    def test_register_returns_the_class_untouched(self):
        @register
        class Example(Automation):
            name = "example"
            cron = "0 9 * * *"
            workflow = "some-workflow"

        # Still an ordinary class, so it can be imported and unit-tested directly.
        assert Example().name == "example"

    def test_name_is_required(self):
        with pytest.raises(ImproperlyConfigured):
            validate(make(name=""))

    def test_no_workflow_named(self):
        with pytest.raises(ImproperlyConfigured, match="names no workflow"):
            validate(make(workflow=""))

    def test_no_schedule(self):
        with pytest.raises(ImproperlyConfigured, match="must set `cron`"):
            validate(make(cron=""))

    def test_a_refused_declaration_does_not_stop_the_import(self, caplog):
        # One app's typo would otherwise take down every page, not just its automation.
        with caplog.at_level("WARNING"):

            @register
            class Broken(Automation):
                name = "broken"
                cron = "not a cron"
                workflow = "some-workflow"

        assert get_automation("broken") is None
        assert "not a valid 5-field cron expression" in get_invalid_automations()["broken"]
        assert "Automation not registered" in caplog.text

    def test_a_refused_class_is_still_returned_unchanged(self):
        @register
        class Broken(Automation):
            name = "broken"
            workflow = ""

        assert Broken().name == "broken"

    def test_an_unnamed_declaration_is_recorded_under_its_class_name(self):
        @register
        class Nameless(Automation):
            cron = "0 9 * * *"
            workflow = "some-workflow"

        assert "Nameless" in get_invalid_automations()

    def test_fixing_it_clears_the_reason(self):
        @register
        class Example(Automation):
            name = "example"
            workflow = ""

        assert "example" in get_invalid_automations()

        @register
        class Fixed(Automation):
            name = "example"
            cron = "0 9 * * *"
            workflow = "some-workflow"

        assert get_invalid_automations() == {}
        assert get_automation("example") is not None

    def test_colliding_names_warn_rather_than_silently_shadow(self, caplog):
        @register
        class First(Automation):
            name = "dup"
            cron = "0 9 * * *"
            workflow = "some-workflow"

        with caplog.at_level("WARNING"):

            @register
            class Second(Automation):
                name = "dup"
                cron = "0 9 * * *"
                workflow = "another-workflow"

        assert "declared by both" in caplog.text

    def test_reimporting_the_same_class_is_not_a_collision(self, caplog):
        class Example(Automation):
            name = "example"
            cron = "0 9 * * *"
            workflow = "some-workflow"

        register(Example)
        with caplog.at_level("WARNING"):
            register(Example)
        assert "declared by both" not in caplog.text


class TestScheduleResolution:
    def test_settings_override_the_class_default(self, settings):
        settings.AI_SDK_AUTOMATIONS = {"example": {"CRON": "*/5 * * * *"}}
        assert str(make().get_schedule()) == str(make(cron="*/5 * * * *").get_schedule())


class TestInput:
    """The automation supplies the turn its workflow starts from, not the behaviour."""

    def test_placeholders_are_filled(self):
        automation = make(input="Everything since {last_run_at} for {name}.")
        rendered = automation.render_input(last_run_at=datetime(2026, 8, 16, tzinfo=UTC))
        assert "2026-08-16" in rendered
        assert "example" in rendered

    def test_a_first_run_says_the_beginning_rather_than_none(self):
        # "since None" would reach the model verbatim and mean nothing to it.
        assert "the beginning" in make(input="Since {last_run_at}.").render_input()

    def test_unknown_placeholders_survive(self):
        # Prose contains braces. Raising here would turn a legitimate sentence into a
        # boot failure.
        assert make(input="Use {json} format.").render_input() == "Use {json} format."

    def test_dotted_and_positional_braces_survive_too(self):
        # A regex on bare {word} tokens can never reach str.format's attribute-access
        # syntax, so `{name.__class__}`-style text can't raise or be resolved as
        # anything but literal prose.
        automation = make(input="See {name.__class__} and {0} and {}.")
        assert automation.render_input() == "See {name.__class__} and {0} and {}."

    def test_there_is_a_usable_default(self):
        assert "example" in make().render_input()


class TestAudienceDefault:
    def test_the_default_is_the_app_itself(self):
        assert make().audience is Audience.APP
