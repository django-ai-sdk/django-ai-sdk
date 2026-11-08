"""The oversight system checks and the settings schema."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from django.test import override_settings
from django_ai_sdk.agents import Agent
from django_ai_sdk.oversight import (
    Alert,
    JudgeLens,
    L,
    LLMJudge,
    Mode,
    Patterns,
    Policy,
    Rule,
    register_policy,
)
from django_ai_sdk.oversight.actions import Block
from django_ai_sdk.oversight.checks import check_oversight, check_oversight_deploy
from django_ai_sdk.oversight.conf import setting
from django_ai_sdk.oversight.types import Stage, Subject
from django_ai_sdk.permissions import AllowAll

from tests.oversight.helpers import (
    ENFORCE,
    policy,
)


class NotAStream:
    """A chat adapter that is not the SDK's Stream."""


class CustomAdapterAgent(Agent):
    _skip_auto_register = True
    permissions = [AllowAll]
    name = "custom-adapter"
    oversight = "test-guard"

    async def get_pipeline_adapter(self, thread_id=None, user=None) -> NotAStream:  # noqa: ANN001
        return NotAStream()


def check_ids(**settings: Any) -> set[str]:
    from django_ai_sdk.oversight.checks import check_oversight

    with override_settings(**settings):
        return {m.id for m in check_oversight()}


def keys(**configured: Any) -> dict[str, str]:
    with override_settings(AI_SDK_OVERSIGHT=configured):
        return {key: problem for key, problem, _ in setting.problems()}


class TestSettingsSchema:
    """E012 names the key, what is wrong, and how to fix it."""

    def test_a_valid_configuration_has_no_problems(self):
        assert not keys(
            FAIL_MODE="open",
            DEADLINE_MS={"gate": 2000, "finish": None},
            CAPTURE="metadata",
            RETENTION={"content": 30},
            MAX_INPUT_CHARS=None,
            JUDGES={"judge": {"class": "LLMJudge", "model": "m"}},
        )

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("FAIL_MODE", {"gate": "closed"}),
            ("FAIL_MODE", "closd"),
            ("CAPTURE", "everything"),
            ("STREAM_MODE", "hold"),
            ("CLIENT_DETAIL", "staff"),
            ("ENABLED", "yes"),
            ("DEADLINE_MS", {"watch": 300}),
            ("DEADLINE_MS", {"gate": "1s"}),
            ("DEADLINE_MS", 1000),
            ("MAX_INPUT_CHARS", -1),
            ("RETENTION", {"content": 0}),
            ("JUDGES", ["judge"]),
            ("NOTIFIERS", "myapp.Slack"),
        ],
    )
    def test_a_value_of_the_wrong_kind_is_an_error(self, key, value):
        assert key in keys(**{key: value})

    def test_validators_are_registered_in_code_not_settings(self):
        # register_validator is the one way to name a validator.
        assert "VALIDATORS" in keys(VALIDATORS={"even": "tests.oversight.test_checks.keys"})

    def test_a_key_that_is_not_a_setting_names_it_and_the_keys(self):
        with override_settings(AI_SDK_OVERSIGHT={"GATE_DEADLINE_MS": 500}):
            found = [m for m in check_oversight() if m.id == "ai_sdk.oversight.E012"]
        assert len(found) == 1
        assert "GATE_DEADLINE_MS" in found[0].msg and "DEADLINE_MS" in found[0].hint

    def test_an_unknown_key_lists_the_keys(self):
        with override_settings(AI_SDK_OVERSIGHT={"DEFAULT_POLCY": "x"}):
            found = [m for m in check_oversight() if m.id == "ai_sdk.oversight.E012"]
        assert "DEFAULT_POLICY" in found[0].hint


class TestPolicyAndDeployChecks:
    def test_an_agent_naming_an_unregistered_policy_is_an_error(self):
        from django_ai_sdk.oversight.checks.agents import agent_checks

        cls = type(
            "Typo",
            (Agent,),
            {"_skip_auto_register": True, "permissions": [AllowAll], "oversight": "suport"},
        )
        with patch("django_ai_sdk.agents.registry.registry.all", lambda: {"a": cls()}):
            found = agent_checks()
        assert [m.id for m in found] == ["ai_sdk.oversight.E009"]

    def _deploy_ids(self, name: str, **configured: Any) -> list[str]:
        settings = {"DEFAULT_POLICY": name, **configured}
        with (
            override_settings(AI_SDK_OVERSIGHT=settings),
            patch("django_ai_sdk.agents.registry.registry.all", dict),
        ):
            return [m.id for m in check_oversight_deploy()]

    def test_an_enforced_gate_rule_only_a_slow_judge_answers_is_a_deploy_error(self):
        name = policy(
            "fc-judge-only",
            [JudgeLens(name="j", stages=[Stage.GATE], judges=[LLMJudge(model="m")])],
            [Rule("r", when=L("j").is_true, then=Block(), mode=ENFORCE)],
        )
        assert "ai_sdk.oversight.E013" in self._deploy_ids(name)
        # A deadline the judge can meet is fine.
        assert "ai_sdk.oversight.E013" not in self._deploy_ids(name, DEADLINE_MS={"gate": 60_000})

    def test_a_slow_judge_behind_an_unsure_pattern_miss_is_a_deploy_error(self):
        # A miss below min_confidence is unsure, so the judge is asked on every miss.
        name = self._chain("fc-judge-behind-patterns", miss_confidence=0.5)
        assert "ai_sdk.oversight.E013" in self._deploy_ids(name)

    def test_a_judge_after_a_sure_miss_is_never_asked_and_not_flagged(self):
        name = self._chain("fc-judge-after-sure-miss", miss_confidence=0.9)
        assert "ai_sdk.oversight.E013" not in self._deploy_ids(name)

    def test_a_local_classifier_is_not_a_slow_judge(self):
        from django_ai_sdk.oversight.judges import LocalClassifier

        name = policy(
            "fc-local",
            [JudgeLens(name="j", stages=[Stage.GATE], judges=[LocalClassifier()])],
            [Rule("r", when=L("j").is_true, then=Block(), mode=ENFORCE)],
        )
        assert "ai_sdk.oversight.E013" not in self._deploy_ids(name)

    def test_a_slow_custom_external_judge_is_a_deploy_error(self):
        from django_ai_sdk.oversight.judges import Judge

        class Remote(Judge):
            external = "my service"
            timeout = 30.0

            async def answer(self, lens, view):
                return None

        name = policy(
            "fc-custom-external",
            [JudgeLens(name="j", stages=[Stage.GATE], judges=[Remote()])],
            [Rule("r", when=L("j").is_true, then=Block(), mode=ENFORCE)],
        )
        assert "ai_sdk.oversight.E013" in self._deploy_ids(name)

    def test_a_slow_judge_on_a_tool_call_lens_is_a_deploy_error(self):
        name = policy(
            "fc-tool-judge",
            [
                JudgeLens(
                    name="j",
                    subject=Subject.TOOL_CALL,
                    stages=[Stage.TOOL],
                    judges=[LLMJudge(model="m")],
                )
            ],
            [Rule("r", when=L("j").is_true, then=Block(), mode=ENFORCE)],
        )
        assert "ai_sdk.oversight.E013" in self._deploy_ids(name)
        assert "ai_sdk.oversight.E013" not in self._deploy_ids(name, DEADLINE_MS={"tool": 60_000})

    def _chain(self, name: str, *, miss_confidence: float) -> str:
        return policy(
            name,
            [
                JudgeLens(
                    name="j",
                    stages=[Stage.GATE],
                    judges=[
                        Patterns([r"ignore"], miss_confidence=miss_confidence),
                        LLMJudge(model="m"),
                    ],
                )
            ],
            [Rule("r", when=L("j").is_true, then=Block(), mode=ENFORCE)],
        )

    def test_a_report_or_alert_rule_on_a_slow_judge_is_not_flagged(self):
        name = policy(
            "fc-judge-report",
            [JudgeLens(name="j", stages=[Stage.GATE], judges=[LLMJudge(model="m")])],
            [
                Rule("r", when=L("j").is_true, then=Block(), mode=Mode.REPORT),
                Rule("a", when=L("j").is_true, then=Alert(), mode=ENFORCE),
            ],
        )
        assert "ai_sdk.oversight.E013" not in self._deploy_ids(name)


@pytest.mark.usefixtures("no_reflect_queue")
class TestChecks:
    def test_every_id_is_namespaced(self):
        from django_ai_sdk.oversight.checks import check_oversight

        with override_settings(AI_SDK_OVERSIGHT={"DEFAULT_POLICY": "test-guard", "X": 1}):
            ids = {m.id for m in check_oversight()}
        assert ids and all(i.startswith("ai_sdk.oversight.") for i in ids)

    def test_unknown_settings_keys_are_errors(self):
        assert "ai_sdk.oversight.E012" in check_ids(AI_SDK_OVERSIGHT={"DEFAULT_POLCY": "x"})
        assert "ai_sdk.oversight.E012" not in check_ids(AI_SDK_OVERSIGHT={"DEFAULT_POLICY": ""})

    def test_a_live_stream_with_an_enforced_finish_mask_warns(self):
        live = check_ids(AI_SDK_OVERSIGHT={"DEFAULT_POLICY": "test-guard", "STREAM_MODE": "live"})
        assert "ai_sdk.oversight.W011" in live
        # The default holds the answer exactly while such a rule is enforced.
        assert "ai_sdk.oversight.W011" not in check_ids(
            AI_SDK_OVERSIGHT={"DEFAULT_POLICY": "test-guard"}
        )

    def test_unanswered_lenses_only_for_policies_in_use(self):
        no_agents = patch("django_ai_sdk.agents.registry.registry.all", dict)
        with no_agents:
            assert "ai_sdk.oversight.W003" not in check_ids(AI_SDK_OVERSIGHT={})
            assert "ai_sdk.oversight.W003" in check_ids(
                AI_SDK_OVERSIGHT={"DEFAULT_POLICY": "default"}
            )

    def test_a_lens_left_with_patterns_alone_is_named(self):
        from django_ai_sdk.oversight.checks.policies import policy_checks

        with override_settings(AI_SDK_OVERSIGHT={"DEFAULT_POLICY": "default"}):
            [warning] = [m for m in policy_checks({"default"}) if m.id.endswith(".W003")]
        # Skipped outright, and answered by its patterns with no model left to ask.
        assert "helpfulness" in warning.msg
        assert "prompt_injection (patterns only)" in warning.msg
        assert "'classifier'" in warning.hint
        with override_settings(
            AI_SDK_OVERSIGHT={
                "DEFAULT_POLICY": "default",
                "JUDGES": {"classifier": {"class": "LLMJudge"}, "judge": "classifier"},
            }
        ):
            assert not [m for m in policy_checks({"default"}) if m.id.endswith(".W003")]

    def test_a_judge_after_a_sure_pattern_miss_is_unreachable(self):
        from django_ai_sdk.oversight.checks.policies import policy_checks

        name = self._sure_miss("fc-unreachable")
        found = [m for m in policy_checks(set()) if m.id.endswith(".W004")]
        [warning] = [m for m in found if repr(name) in m.msg]
        assert "judge:m" in warning.msg

    def _sure_miss(self, name: str) -> str:
        return policy(
            name,
            [JudgeLens(name="j", judges=[Patterns([r"x"], miss_confidence=0.9), LLMJudge("m")])],
            [],
        )

    def test_a_policy_name_declared_twice_is_an_error(self, registry_snapshot: Any):
        class First(Policy):
            name = "test-twice"
            lenses = []  # noqa: RUF012
            rules = []  # noqa: RUF012

        class Second(Policy):
            name = "test-twice"
            lenses = []  # noqa: RUF012
            rules = []  # noqa: RUF012

        register_policy(First)
        register_policy(Second)
        assert registry_snapshot.get_policy_class("test-twice") is First
        assert "ai_sdk.oversight.E010" in check_ids()

    def test_a_chat_adapter_that_is_not_a_stream_warns(self):
        from django_ai_sdk.oversight.checks.agents import agent_checks

        with patch(
            "django_ai_sdk.agents.registry.registry.all", lambda: {"a": CustomAdapterAgent()}
        ):
            found = agent_checks()
        assert [m.id for m in found] == ["ai_sdk.oversight.W013"]

        class OptedOut(CustomAdapterAgent):
            oversight = False

        with patch("django_ai_sdk.agents.registry.registry.all", lambda: {"a": OptedOut()}):
            assert agent_checks() == []

    def test_the_default_capture_is_masked(self):
        assert setting.defaults["CAPTURE"] == "masked"


def test_the_manual_lists_every_check_id():
    """Check ids are public: the manual's table is the ids the code raises."""
    import re
    from pathlib import Path

    from django_ai_sdk.checks import SETTINGS_CHECK

    root = Path(__file__).resolve().parents[2]
    # The opt-in apps raise their own ids, under their own PREFIX.
    opt_in = {"ledger", "switches"}
    code = "".join(
        path.read_text(encoding="utf-8")
        for path in (root / "django_ai_sdk/oversight").rglob("*.py")
        if not opt_in & set(path.relative_to(root).parts)
    )
    raised = set(re.findall(r'id=f"\{PREFIX\}\.([EW]\d{3})"', code)) | {SETTINGS_CHECK}
    manual = (root / "docs/content/manual/oversight.md").read_text(encoding="utf-8")
    section = manual.split("\n## System checks\n", 1)[1].split("\n## ", 1)[0]
    listed = set(re.findall(r"^\| `ai_sdk\.oversight\.([EW]\d{3})`", section, re.M))
    assert listed == raised
