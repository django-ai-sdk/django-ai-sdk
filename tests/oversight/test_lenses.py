"""Lenses: pattern, check and judge lenses, their presets, and the rules they follow."""

from __future__ import annotations

import re
import time

import pytest
from django.core.exceptions import ImproperlyConfigured
from django_ai_sdk.oversight import (
    CheckLens,
    Choice,
    Entities,
    Flag,
    JudgeLens,
    L,
    PatternLens,
    Score,
    Violation,
    register_validator,
)
from django_ai_sdk.oversight.judges.base import ask
from django_ai_sdk.oversight.lenses import lens_from_dict
from django_ai_sdk.oversight.patterns import (
    COUNTRY_PII_KINDS,
    DEFAULT_PII_KINDS,
    MAX_SCAN_CHARS,
    PII_PATTERNS,
    bsn_ok,
    iban_ok,
    luhn_ok,
    mask,
    mask_all,
    mask_secrets,
)
from django_ai_sdk.oversight.presets import (
    EmptyRetrievalLens,
    LanguageLens,
    LengthLens,
    PIILens,
    RefusalLens,
    SecretsLens,
    ToolErrorsLens,
)
from django_ai_sdk.oversight.types import (
    Answer,
    Source,
    Stage,
    Subject,
    TextSpan,
    ToolCall,
    TurnContext,
)

from tests.oversight.helpers import FnLens


def view(text: str = "", output: str = "", sources: list[str] | None = None) -> TurnContext:
    return TurnContext(
        turn_id="t1",
        input=text,
        output=output,
        sources=[Source(title=f"s{i}", content=c) for i, c in enumerate(sources or [])],
    )


async def answer(lens, turn: TurnContext) -> Answer | None:
    """What the lens's one step answers about `turn`."""
    [step] = lens.steps()
    return await ask(step, lens, turn)


class TestChecksums:
    def test_bsn_eleven_check(self):
        assert bsn_ok("111222333")
        assert bsn_ok("1112.22.333")
        assert not bsn_ok("111222334")
        assert not bsn_ok("000000000")

    def test_iban_mod_97(self):
        assert iban_ok("NL91ABNA0417164300")
        assert iban_ok("NL91 ABNA 0417 1643 00")
        assert not iban_ok("NL92ABNA0417164300")

    def test_luhn(self):
        assert luhn_ok("4111 1111 1111 1111")
        assert not luhn_ok("4111 1111 1111 1112")


class TestPII:
    async def test_finds_valid_kinds_only(self):
        text = "Mail a@b.nl, IBAN NL91ABNA0417164300, bsn 111222333, fake bsn 111222334"
        found = await answer(PIILens(kinds=[*DEFAULT_PII_KINDS, "bsn"]), view(text))
        assert found.value == ["bsn", "email", "iban"]
        assert found.confidence == 1.0
        assert len(found.spans) == 3
        assert found.answerer == "patterns"

    async def test_nothing_found_is_a_quiet_no(self):
        found = await answer(PIILens(), view("How do I reset my password?"))
        assert found.value == []
        # What patterns find is a fact, so a miss is as sure as a hit.
        assert found.confidence == 1
        assert found.rationale == "found none of: email, iban, credit card, phone, ip address"

    async def test_an_address_has_no_format_to_find(self):
        # Code finds what has a fixed shape; a street address or a name needs a model.
        assert (await answer(PIILens(), view("Mijn adres is Jan Voornstraat 8"))).value == []

    async def test_a_bsn_must_pass_the_eleven_check(self):
        bsn = PIILens(kinds=["bsn"])
        assert (await answer(bsn, view("Mijn BSN is 123456789"))).value == []
        assert (await answer(bsn, view("Mijn BSN is 111222333"))).value == ["bsn"]

    def test_an_unknown_kind_is_refused(self):
        with pytest.raises(ImproperlyConfigured):
            PIILens(kinds=["shoe_size"]).validate()

    async def test_secrets(self):
        text = "my key is sk-proj-abcdefghijklmnopqrstuvwxyz1234 and AKIAABCDEFGHIJKLMNOP"
        assert (await answer(SecretsLens(), view(text))).value == ["api_key", "aws_key"]

    def test_email_is_linear_on_hostile_input(self):
        email = re.compile(PII_PATTERNS["email"])
        started = time.perf_counter()
        assert email.search("a." * 20_000) is None
        assert time.perf_counter() - started < 0.05
        for address in ("jan.de+x@example.co.uk", "foo.bar@sub.example.com"):
            assert email.search(address).group() == address
        assert email.search("a@b.nl.").group() == "a@b.nl"

    async def test_patterns_read_all_of_a_long_text(self):
        # Padding must not hide what comes after it, nor a match on a window's edge.
        late = "x " * MAX_SCAN_CHARS + "a@b.nl"
        found = await answer(PIILens(), view(late))
        assert found.value == ["email"] and late[found.spans[0].start :].startswith("a@b.nl")
        edge = "x" * (MAX_SCAN_CHARS - 4) + " a@b.nl " + "y" * MAX_SCAN_CHARS
        found = await answer(PIILens(), view(edge))
        assert [edge[span.start : span.end] for span in found.spans] == ["a@b.nl"]

    async def test_mask_replaces_spans_by_label(self):
        text = "write to a@b.nl now"
        found = await answer(PIILens(), view(text))
        assert mask(text, found.spans) == "write to [EMAIL] now"

    def test_mask_all_masks_every_group_on_the_original_offsets(self):
        text = "abcdefghij"
        groups = [
            ([TextSpan(start=1, end=4, label="one")], "<{label}>"),
            # Overlaps the first: merged into it. Adjacent to it: masked on its own.
            (
                [TextSpan(start=3, end=6, label="two"), TextSpan(start=6, end=8, label="three")],
                "[{label}]",
            ),
        ]
        assert mask_all(text, groups) == "a<ONE>[THREE]ij"
        assert mask_all(text, []) == text

    def test_mask_secrets(self):
        key = "sk-proj-abcdefghijklmnopqrstuvwxyz1234"
        assert mask_secrets(f"use {key} now") == "use [API_KEY] now"
        assert mask_secrets("nothing here") == "nothing here"


class TestPatternLens:
    async def test_patterns_label_by_group(self):
        lens = PatternLens(name="x", patterns={"greeting": [r"\bhello\b"], "bye": r"\bbye\b"})
        assert (await answer(lens, view("hello and bye"))).value == ["bye", "greeting"]

    async def test_a_plain_list_labels_hits_with_the_lens_name(self):
        lens = PatternLens(name="competitor", keywords=["acme", "globex"])
        assert (await answer(lens, view("acme or Globex"))).value == ["competitor"]
        assert (await answer(lens, view("Is it cheaper?"))).value == []

    async def test_refusal_is_answered_by_its_patterns_without_a_classifier(self):
        lens = RefusalLens()
        lens.validate()
        found = await answer(lens, view(output="Sorry, I can't help with that."))
        assert (found.value, found.confidence) == (True, 1.0)

    def test_it_answers_entities_only(self):
        with pytest.raises(ImproperlyConfigured, match="Entities only"):
            PatternLens(name="x", patterns=["x"], answer=Flag).validate()

    def test_it_needs_patterns_or_keywords(self):
        with pytest.raises(ImproperlyConfigured, match="no `patterns`"):
            PatternLens(name="x").validate()

    def test_a_bad_regex_is_refused(self):
        with pytest.raises(ImproperlyConfigured, match="bad regex"):
            PatternLens(name="x", patterns=["(unclosed"]).validate()

    def test_is_true_means_it_found_something(self):
        found, none = Answer(value=["email"]), Answer(value=[])
        assert L("x").is_true.evaluate({"x": found})
        assert not L("x").is_true.evaluate({"x": none})
        assert L("x").is_false.evaluate({"x": none})
        assert not L("x").is_false.evaluate({"x": found})


class TestPatternsInTheWild:
    """What real text looks like: a payment line, a server spec, a question about settings."""

    @pytest.mark.parametrize(
        "text",
        [
            "Pay BE68539007547034 EUR 25",
            "Pay BE68 5390 0754 7034 EUR 25",
            "Pay to BE68539007547034 GKCCBEBB today",
            "Pay to NL91 ABNA 0417 1643 00 ABNANL2A today",
        ],
    )
    async def test_an_iban_followed_by_a_currency_or_a_bic_is_masked(self, text):
        found = await answer(PIILens(kinds=["iban"]), view(text))
        assert found.value == ["iban"]
        masked = mask(text, found.spans)
        assert "[IBAN]" in masked and "0754" not in masked and "0417" not in masked
        # Only the IBAN: what follows it stays.
        assert masked.split("[IBAN]")[1].split()[0] in {"EUR", "GKCCBEBB", "ABNANL2A"}

    async def test_a_card_followed_by_its_expiry_is_masked(self):
        text = "card 4111 1111 1111 1111 2027 please"
        found = await answer(PIILens(kinds=["credit_card"]), view(text))
        assert mask(text, found.spans) == "card [CREDIT_CARD] 2027 please"

    async def test_a_shape_that_fails_its_checksum_stays_unmasked(self):
        found = await answer(PIILens(kinds=["iban"]), view("BE69539007547034 EUR"))
        assert found.value == []

    @pytest.mark.parametrize("text", ["4096 MB of memory", "HTTP 8080 OK", "In 1984 WE met"])
    async def test_a_number_and_a_unit_is_not_a_postcode(self, text):
        assert (await answer(PIILens(kinds=["postcode_nl"]), view(text))).value == []

    @pytest.mark.parametrize("text", ["Damrak 1, 1012 LG Amsterdam", "postcode 1012LG"])
    async def test_a_postcode_still_is(self, text):
        found = await answer(PIILens(kinds=["postcode_nl"]), view(text))
        assert found.value == ["postcode_nl"]

    @pytest.mark.parametrize(
        "text",
        [
            "Where is my password: settings or profile?",
            "I forgot my password: changed it yesterday",
            "Wachtwoord: vergeten, wat nu?",
        ],
    )
    async def test_prose_about_a_password_is_not_one(self, text):
        assert (await answer(SecretsLens(), view(text))).value == []

    @pytest.mark.parametrize(
        "text", ["password=hunter2", 'password: "correcthorse"', "wachtwoord: Welkom2024!"]
    )
    async def test_a_password_assignment_is_one(self, text):
        assert (await answer(SecretsLens(), view(text))).value == ["password_assignment"]


class TestPIIKinds:
    """Formats that mean the same everywhere by default; a country's own when asked for."""

    TEXT = "BSN 111222333, bel 06-12345678 of +31 6 12345678, Damrak 1, 1012 LG Amsterdam"

    async def test_country_kinds_are_off_by_default(self):
        assert not set(COUNTRY_PII_KINDS) & set(PIILens().finder.labels())
        assert (await answer(PIILens(), view(self.TEXT))).value == ["phone"]

    async def test_naming_them_turns_them_on(self):
        lens = PIILens(kinds=[*DEFAULT_PII_KINDS, "bsn", "phone_nl", "postcode_nl"])
        found = await answer(lens, view(self.TEXT))
        assert found.value == ["bsn", "phone", "phone_nl", "postcode_nl"]

    def test_no_kinds_picked_means_the_defaults(self):
        assert PIILens(kinds=[]).finder.labels() == list(DEFAULT_PII_KINDS)

    def test_the_docstring_lists_every_kind(self):
        assert all(kind in PIILens.__doc__ for kind in PII_PATTERNS)

    @pytest.mark.parametrize(
        ("key", "label"),
        [
            ("sk-ant-api03-" + "a" * 40, "api_key"),
            ("sk-proj-" + "b" * 40, "api_key"),
            ("sk_live_" + "c" * 24, "stripe_key"),
            ("rk_live_" + "d" * 24, "stripe_key"),
            ("AIza" + "e" * 35, "google_api_key"),
        ],
    )
    async def test_keys(self, key, label):
        assert (await answer(SecretsLens(), view(f"use {key} now"))).value == [label]


def even(text):
    if int(text) % 2:
        raise Violation("Odd amounts are not allowed.")


def short_answers_only(text, view):
    if len(view.output) > 5:
        raise Violation("The answer is long.")


class TestCheckLens:
    async def test_a_violation_is_a_yes_with_its_message_as_the_reason(self):
        lens = CheckLens(name="even", validators=[even])
        odd = await answer(lens, view("3"))
        assert (odd.value, odd.rationale, odd.answerer) == (
            True,
            "Odd amounts are not allowed.",
            "check",
        )
        assert (await answer(lens, view("4"))).value is False

    async def test_a_validator_may_read_the_turn(self):
        lens = CheckLens(name="short", subject=Subject.OUTPUT, validators=[short_answers_only])
        assert (await answer(lens, view(output="a long answer"))).value is True
        assert (await answer(lens, view(output="ok"))).value is False

    async def test_validators_by_name(self):
        register_validator("test_even")(even)
        lens = CheckLens(name="even", validators=["test_even"])
        assert (await answer(lens, view("5"))).value is True

    def test_an_unknown_validator_is_refused(self):
        with pytest.raises(ImproperlyConfigured, match="No validator"):
            CheckLens(name="x", validators=["nobody_registered_this"]).validate()

    def test_validators_answer_a_flag(self):
        with pytest.raises(ImproperlyConfigured, match="validators answer a Flag"):
            CheckLens(name="x", validators=[even], answer=Score).validate()
        FnLens(name="x", answer=Score, fn=lambda text, view: 0.5).validate()

    def test_a_check_lens_needs_validators_or_its_own_check(self):
        with pytest.raises(ImproperlyConfigured, match="no `validators`"):
            CheckLens(name="x").validate()

    async def test_presets_answer_facts_about_the_turn(self):
        long = await answer(LengthLens(max_chars=3), view("four"))
        assert (long.value, long.rationale) == (True, "4 characters")
        mismatch = await answer(
            LanguageLens(),
            view(
                "Wat is het adres van de winkel?", "The shop is on the main street and it is open."
            ),
        )
        assert mismatch.value is True
        looked = TurnContext(tool_calls=[ToolCall(name="search", result=None)])
        assert (await answer(EmptyRetrievalLens(), looked)).value is True
        # An empty search leaves no grounding, and the lens still runs.
        assert EmptyRetrievalLens().applies(looked) and not EmptyRetrievalLens().applies(TurnContext())
        calls = [ToolCall(name="a", error=True), ToolCall(name="b")]
        assert (await answer(ToolErrorsLens(), TurnContext(tool_calls=calls))).value == 0.5
        assert await answer(ToolErrorsLens(), TurnContext()) is None


class TestFlag:
    def test_a_dutch_word_is_not_a_yes(self):
        with pytest.raises(ValueError):
            Flag.parse(JudgeLens(name="x"), "ja")


class TestAttributesBelongWhereTheyAreRead:
    """An attribute a lens does not read is refused, never silently ignored."""

    def test_a_judge_attribute_on_a_pattern_lens(self):
        class Asks(PatternLens):
            name = "asks"
            patterns = ["x"]  # noqa: RUF012
            question = "Is it x?"

        with pytest.raises(ImproperlyConfigured, match="question"):
            Asks().validate()

    def test_an_option_of_another_answer_type(self):
        with pytest.raises(ImproperlyConfigured, match="labels"):
            JudgeLens(name="f", judges=["judge"], labels=["a"]).validate()
        with pytest.raises(ImproperlyConfigured, match="yes"):
            JudgeLens(name="s", judges=["judge"], answer=Score, yes="Yes!").validate()

    def test_an_attribute_no_lens_of_its_kind_has(self):
        with pytest.raises(ImproperlyConfigured, match="judges"):
            CheckLens(name="x", validators=[even], judges=["judge"])

    def test_options_left_out_take_the_answer_types_defaults(self):
        flag = JudgeLens(name="f", judges=["judge"])
        assert (flag.yes, flag.no) == ("Yes.", "No.")
        score = JudgeLens(name="s", judges=["judge"], answer=Score, low="none")
        assert (score.low, score.middle, score.high) == ("none", "partly", "completely")
        choice = JudgeLens(name="c", judges=["judge"], answer=Choice, labels=["a", "other"])
        assert (choice.default, choice.criteria) == ("", {})
        assert not hasattr(flag, "labels")

    def test_a_choice_set_default_must_be_among_its_labels(self):
        with pytest.raises(ImproperlyConfigured, match="default"):
            JudgeLens(
                name="c", judges=["j"], answer=Choice, labels=["a", "b"], default="other"
            ).validate()

    def test_a_judge_only_choice_needs_no_other_label(self):
        lens = JudgeLens(name="c", judges=["j"], answer=Choice, labels=["a", "b"])
        lens.validate()
        # Patterns that find none give the last label.
        assert Choice.from_detections(lens, []) == "b"


class TestFromDict:
    def test_a_pattern_lens(self):
        lens = lens_from_dict({"kind": "pattern", "name": "x", "keywords": ["acme"]})
        assert isinstance(lens, PatternLens) and lens.answer is Entities

    def test_a_check_lens_names_registered_validators(self):
        register_validator("test_even")(even)
        lens = lens_from_dict({"kind": "check", "name": "even", "validators": ["test_even"]})
        assert isinstance(lens, CheckLens)
        with pytest.raises(ImproperlyConfigured):
            lens_from_dict({"kind": "check", "name": "x", "validators": ["os.system"]})

    def test_a_judge_lens_with_its_answer(self):
        lens = lens_from_dict(
            {
                "kind": "judge",
                "name": "topic",
                "answer": "choice",
                "labels": ["hr", "other"],
                "judges": ["judge?"],
                "stages": ["reflect"],
            }
        )
        assert isinstance(lens, JudgeLens) and lens.answer is Choice
        assert lens.stages == (Stage.REFLECT,)

    def test_a_preset(self):
        lens = lens_from_dict({"kind": "pattern", "preset": "pii", "name": "p", "kinds": ["bsn"]})
        assert isinstance(lens, PIILens) and lens.kinds == ["bsn"]
        with pytest.raises(ImproperlyConfigured, match="not a check one"):
            lens_from_dict({"kind": "check", "preset": "pii", "name": "p"})

    def test_an_unknown_kind_or_answer(self):
        with pytest.raises(ImproperlyConfigured, match="lens kind"):
            lens_from_dict({"kind": "flag", "name": "x"})
        with pytest.raises(ImproperlyConfigured, match="Unknown answer"):
            lens_from_dict({"kind": "judge", "name": "x", "answer": "maybe", "judges": ["j"]})
