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
    Mask,
    Mode,
    Maybe,
    PatternJudge,
    PIILens,
    Policy,
    Rule,
    register_policy,
)
from django_ai_sdk.oversight.rules import Block
from django_ai_sdk.oversight.checks import check_oversight, check_oversight_deploy
from django_ai_sdk.oversight.conf import DEFAULTS
from django_ai_sdk.oversight.types import Stage, Subject
from django_ai_sdk.permissions import AllowAll

from tests.oversight.helpers import (
    ENFORCE,
    policy,
)


def check_ids(**settings: Any) -> set[str]:
    from django_ai_sdk.oversight.checks import check_oversight

    with override_settings(**settings):
        return {m.id for m in check_oversight()}


def keys(**configured: Any) -> str:
    with override_settings(AI_SDK_OVERSIGHT=configured):
        from django_ai_sdk.oversight.checks import settings_checks

        return " ".join(error.msg for error in settings_checks())


class TestSettingsSchema:
    def test_a_valid_configuration_has_no_problems(self):
        assert not keys(
            FAIL_MODE="open",
            DEADLINE_MS={"gate": 2000, "finish": None},
            CAPTURE="metadata",
            RETENTION={"content": 30},
            MAX_INPUT_CHARS=None,
            MAX_OTHER_TEXT_CHUNKS=4,
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
            ("MAX_OTHER_TEXT_CHUNKS", 0),
            ("MAX_OTHER_TEXT_CHUNKS", None),
            ("RETENTION", {"content": 0}),
            ("JUDGES", ["judge"]),
            ("NOTIFIERS", "myapp.Slack"),
        ],
    )
    def test_a_value_of_the_wrong_kind_is_an_error(self, key, value):
        assert key in keys(**{key: value})

    def test_the_message_says_what_the_key_accepts(self):
        assert keys(FAIL_MODE="closd") == "AI_SDK_OVERSIGHT['FAIL_MODE'] must be one of 'closed', 'open'."

    def test_every_setting_has_a_rule(self):
        from django_ai_sdk.oversight.checks import ACCEPTS

        assert list(ACCEPTS) == list(DEFAULTS)

    def test_validators_are_registered_in_code_not_settings(self):
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
        from django_ai_sdk.oversight.checks import agent_checks

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

    def test_an_enforced_gate_rule_only_a_slow_judge_answers_is_a_deploy_warning(self):
        name = policy(
            "fc-judge-only",
            [JudgeLens(name="j", stages=[Stage.GATE], judges=[LLMJudge(model="m")])],
            [Rule("r", when=L("j").is_true, then=Block(), mode=ENFORCE)],
        )
        assert "ai_sdk.oversight.E013" in self._deploy_ids(name)
        assert "ai_sdk.oversight.E013" not in self._deploy_ids(name, DEADLINE_MS={"gate": 60_000})

    def test_a_slow_judge_behind_an_unsure_pattern_miss_is_a_deploy_warning(self):
        name = self._chain("fc-judge-behind-patterns", miss_confidence=0.5)
        assert "ai_sdk.oversight.E013" in self._deploy_ids(name)

    def test_a_local_classifier_is_not_a_slow_judge(self):
        from django_ai_sdk.oversight import LocalClassifier

        name = policy(
            "fc-local",
            [JudgeLens(name="j", stages=[Stage.GATE], judges=[LocalClassifier()])],
            [Rule("r", when=L("j").is_true, then=Block(), mode=ENFORCE)],
        )
        assert "ai_sdk.oversight.E013" not in self._deploy_ids(name)

    def test_a_slow_custom_external_judge_is_a_deploy_warning(self):
        from django_ai_sdk.oversight import Judge

        class Remote(Judge):
            external = "my service"
            timeout = 30.0

            async def answer(self, lens, context):
                return None

        name = policy(
            "fc-custom-external",
            [JudgeLens(name="j", stages=[Stage.GATE], judges=[Remote()])],
            [Rule("r", when=L("j").is_true, then=Block(), mode=ENFORCE)],
        )
        assert "ai_sdk.oversight.E013" in self._deploy_ids(name)

    def test_a_slow_judge_on_a_tool_call_lens_is_a_deploy_warning(self):
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
                        PatternJudge([r"ignore"], miss_confidence=miss_confidence),
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
        assert "ai_sdk.oversight.W011" not in check_ids(
            AI_SDK_OVERSIGHT={"DEFAULT_POLICY": "test-guard"}
        )

    def test_an_enforced_rule_no_judge_answers(self):
        from django.core.checks import ERROR, WARNING

        from django_ai_sdk.oversight.checks import policy_checks

        def level(judge: str | Maybe) -> list[int]:
            name = policy(
                f"fc-blind-{type(judge).__name__}",
                [JudgeLens(name="j", judges=[judge])],
                [Rule("r", when=L("j").is_true, then=Block(), mode=ENFORCE)],
            )
            found = policy_checks(set())
            return [
                (m.id.rsplit(".", 1)[1], m.level)
                for m in found
                if m.id.endswith((".E007", ".W008")) and repr(name) in m.msg
            ]

        assert level(Maybe("fc-nobody")) == [("W008", WARNING)]
        assert level("fc-nobody") == [("E007", ERROR)]

    def test_an_optional_judge_that_is_defined_but_cannot_run_is_an_error(self):
        from django_ai_sdk.oversight import TypeSafe
        from django_ai_sdk.oversight.checks import lens_checks
        from django_ai_sdk.oversight.registry import build_policy

        name = policy(
            "fc-maybe-defined",
            [JudgeLens(name="risky", question="Risky?", judges=[Maybe("classifier")])],
            [Rule("r", when=L("risky").is_true, then=Block(), mode=ENFORCE)],
        )
        with (
            override_settings(AI_SDK_OVERSIGHT={"JUDGES": {"classifier": TypeSafe()}}),
            patch.object(TypeSafe, "missing", return_value="no key"),
        ):
            found = lens_checks(name, build_policy(name))
        assert [m.id.rsplit(".", 1)[1] for m in found] == ["E007"]

    def test_an_enforced_rule_on_a_lens_its_judges_cannot_answer(self):
        from django_ai_sdk.oversight import Entities, TypeSafe
        from django_ai_sdk.oversight.checks import lens_checks
        from django_ai_sdk.oversight.registry import build_policy

        name = policy(
            "fc-entities-classifier",
            [JudgeLens(name="names", answer=Entities, question="Which?", judges=[TypeSafe()])],
            [Rule("r", when=L("names").is_true, then=Block(), mode=ENFORCE)],
        )
        with patch.object(TypeSafe, "missing", return_value=""):
            found = lens_checks(name, build_policy(name))
        assert [m.id.rsplit(".", 1)[1] for m in found] == ["E007"]

    def test_a_notify_that_reaches_no_one_is_flagged(self):
        from django_ai_sdk.oversight.rules import Notify
        from django_ai_sdk.oversight.checks import policy_checks

        lens = JudgeLens(name="j", judges=[PatternJudge([r"x"])])
        when = L("j").is_true
        typo = policy("fc-notify-typo", [lens], [Rule("r", when=when, then=Notify(["siem"]))])
        loud = policy("fc-notify-none", [lens], [Rule("r", when=when, then=Notify(), mode=ENFORCE)])

        def found(id: str, in_use: set[str], **configured: Any) -> list[str]:
            with override_settings(AI_SDK_OVERSIGHT=configured):
                return [m.msg for m in policy_checks(in_use) if m.id.endswith(id)]

        notifiers = {"NOTIFIERS": ["django_ai_sdk.oversight.notifiers.LogNotifier"]}
        assert any(repr(typo) in msg for msg in found(".E014", set(), **notifiers))
        assert not any(repr(loud) in msg for msg in found(".E014", set(), **notifiers))
        assert any(repr(loud) in msg for msg in found(".W015", {loud}))
        assert not found(".W015", {loud}, **notifiers)

    def test_a_block_that_waits_for_a_reflect_lens_warns(self):
        from django_ai_sdk.oversight.checks import policy_checks

        gate = JudgeLens(name="g", stages=[Stage.GATE], judges=[PatternJudge([r"x"])])
        later = JudgeLens(name="later", stages=[Stage.REFLECT], judges=[PatternJudge([r"y"])])
        both = L("g").is_true & L("later").is_true
        mixed = policy(
            "fc-reflect-block",
            [gate, later],
            [Rule("r", when=both, then=Block(), mode=ENFORCE)],
        )
        reported = policy(
            "fc-reflect-report", [gate, later], [Rule("r", when=both, then=Block())]
        )
        alone = policy(
            "fc-gate-block",
            [gate, later],
            [
                Rule("r", when=L("g").is_true, then=Block(), mode=ENFORCE),
                Rule("a", when=both, then=Alert("info"), mode=ENFORCE),
            ],
        )
        warned = [m.msg for m in policy_checks(set()) if m.id.endswith(".W018")]
        assert any(repr(mixed) in msg and "'r'" in msg for msg in warned)
        assert not any(repr(reported) in msg or repr(alone) in msg for msg in warned)

    def test_a_mask_that_waits_past_the_gate_warns(self):
        from django_ai_sdk.oversight.checks import policy_checks

        asked = PIILens(name="asked", stages=[Stage.GATE], kinds=["email"])
        said = PIILens(name="said", subject=Subject.ANSWER, stages=[Stage.FINISH], kinds=["email"])
        either = L("asked").is_true | L("said").is_true
        waits = policy(
            "fc-mask-waits", [asked, said], [Rule("m", when=either, then=Mask(), mode=ENFORCE)]
        )
        split = policy(
            "fc-mask-split",
            [asked, said],
            [
                Rule("m", when=L("asked").is_true, then=Mask(), mode=ENFORCE),
                Rule("n", when=L("said").is_true, then=Mask(), mode=ENFORCE),
            ],
        )
        warned = [m.msg for m in policy_checks(set()) if m.id.endswith(".W019")]
        assert any(repr(waits) in msg and "'m'" in msg for msg in warned)
        assert not any(repr(split) in msg for msg in warned)

    def test_a_policy_name_declared_twice_is_an_error(self, registry_snapshot: Any):
        class First(Policy):
            name = "test-twice"
            lenses = []
            rules = []

        class Second(Policy):
            name = "test-twice"
            lenses = []
            rules = []

        register_policy(First)
        register_policy(Second)
        assert registry_snapshot.get_policy_class("test-twice") is First
        assert "ai_sdk.oversight.E010" in check_ids()

    def test_the_default_capture_is_masked(self):
        assert DEFAULTS["CAPTURE"] == "masked"


def test_the_manual_lists_every_check_id():
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    code = "".join(
        p.read_text(encoding="utf-8") for p in (root / "django_ai_sdk/oversight").rglob("*.py")
    )
    raised = set(re.findall(r'id=f"\{PREFIX\}\.([EW]\d{3})"', code)) | {"E012"}
    manual = (root / "docs/content/manual/oversight.md").read_text(encoding="utf-8")
    section = manual.split("\n## System checks\n", 1)[1].split("\n## ", 1)[0]
    listed = set(re.findall(r"^\| `ai_sdk\.oversight\.([EW]\d{3})`", section, re.M))
    assert listed == raised
