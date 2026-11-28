from __future__ import annotations

import re
import time

import pytest
from django.core.exceptions import ImproperlyConfigured
from django_ai_sdk.oversight import (
    Maybe,
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
from django_ai_sdk.oversight.judges.base import ask_judge
from django_ai_sdk.oversight.lenses import lens_from_dict
from django_ai_sdk.oversight.patterns import (
    DEFAULT_PII_KINDS,
    PII_NOT_AFTER,
    PII_PATTERNS,
    PII_VERIFY,
    finds_personal_or_secret,
    iban_ok,
    luhn_ok,
    mask,
    mask_all,
    mask_secrets,
    register_pii_kind,
)
from django_ai_sdk.oversight.lenses import (
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
    Stage,
    Subject,
    TextSpan,
    ToolCall,
    TurnContext,
)

from tests.oversight.helpers import FnLens, turn_context


async def answer(lens, turn: TurnContext) -> Answer | None:
    [step] = lens.steps()
    return await ask_judge(step, lens, turn)


class TestChecksums:
    def test_iban_mod_97(self):
        assert iban_ok("NL91ABNA0417164300")
        assert iban_ok("NL91 ABNA 0417 1643 00")
        assert not iban_ok("NL92ABNA0417164300")

    def test_luhn(self):
        assert luhn_ok("4111 1111 1111 1111")
        assert not luhn_ok("4111 1111 1111 1112")


class TestPII:
    async def test_finds_valid_kinds_only(self):
        text = "Mail a@b.nl, IBAN NL91ABNA0417164300, fake IBAN NL91ABNA0417164301"
        found = await answer(PIILens(), turn_context(text))
        assert found.value == ["email", "iban"]
        assert found.confidence == 1.0
        assert len(found.spans) == 2
        assert found.answerer == "patterns"

    async def test_nothing_found_is_a_quiet_no(self):
        found = await answer(PIILens(), turn_context("How do I reset my password?"))
        assert found.value == []
        assert found.confidence == 1
        assert found.rationale == "found none of: email, iban, credit card, phone, ip address"

    async def test_an_address_has_no_format_to_find(self):
        assert (
            await answer(PIILens(), turn_context("Mijn adres is Jan Voornstraat 8"))
        ).value == []

    def test_an_unknown_kind_is_refused(self):
        with pytest.raises(ImproperlyConfigured):
            PIILens(kinds=["shoe_size"]).validate()

    async def test_secrets(self):
        text = "my key is sk-proj-abcdefghijklmnopqrstuvwxyz1234 and AKIAABCDEFGHIJKLMNOP"
        assert (await answer(SecretsLens(), turn_context(text))).value == ["api_key", "aws_key"]

    def test_email_is_linear_on_hostile_input(self):
        email = re.compile(PII_PATTERNS["email"])
        for hostile in ("a." * 20_000, "a-" * 20_000, "a [dot] " * 5_000, "a " * 20_000 + "[at]"):
            started = time.perf_counter()
            assert email.search(hostile) is None
            assert time.perf_counter() - started < 0.1
        for address in ("jan.de+x@example.co.uk", "foo.bar@sub.example.com"):
            assert email.search(address).group() == address
        assert email.search("a@b.nl.").group() == "a@b.nl"

    async def test_patterns_read_all_of_a_long_text(self):
        late = "x " * 50_000 + "a@b.nl"
        found = await answer(PIILens(), turn_context(late))
        assert found.value == ["email"] and late[found.spans[0].start :].startswith("a@b.nl")

    async def test_a_long_private_key_is_masked_whole(self):
        key = "-----BEGIN RSA PRIVATE KEY-----\n" + ("A" * 64 + "\n") * 40
        key += "-----END RSA PRIVATE KEY-----"
        text = "x" * 48_000 + key + " tail"
        found = await answer(SecretsLens(), turn_context(text))
        assert mask(text, found.spans) == "x" * 48_000 + "[PRIVATE_KEY] tail"

    async def test_mask_replaces_spans_by_label(self):
        text = "write to a@b.nl now"
        found = await answer(PIILens(), turn_context(text))
        assert mask(text, found.spans) == "write to [EMAIL] now"

    def test_mask_all_masks_every_group_on_the_original_offsets(self):
        text = "abcdefghij"
        groups = [
            ([TextSpan(start=1, end=4, label="one")], "<{label}>"),
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
        assert (await answer(lens, turn_context("hello and bye"))).value == ["bye", "greeting"]

    async def test_a_plain_list_labels_hits_with_the_lens_name(self):
        lens = PatternLens(name="competitor", keywords=["acme", "globex"])
        assert (await answer(lens, turn_context("acme or Globex"))).value == ["competitor"]
        assert (await answer(lens, turn_context("Is it cheaper?"))).value == []

    async def test_refusal_is_answered_by_its_patterns_without_a_classifier(self):
        lens = RefusalLens()
        lens.validate()
        found = await answer(lens, turn_context(output="Sorry, I can't help with that."))
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
        assert L("x").is_true.matches({"x": found})
        assert not L("x").is_true.matches({"x": none})
        assert L("x").is_false.matches({"x": none})
        assert not L("x").is_false.matches({"x": found})


class TestPatternsInTheWild:
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
        found = await answer(PIILens(kinds=["iban"]), turn_context(text))
        assert found.value == ["iban"]
        masked = mask(text, found.spans)
        assert "[IBAN]" in masked and "0754" not in masked and "0417" not in masked
        assert masked.split("[IBAN]")[1].split()[0] in {"EUR", "GKCCBEBB", "ABNANL2A"}

    @pytest.mark.parametrize(
        ("text", "kind", "secret"),
        [
            ("rekening nl91abna0417164300 graag", "iban", "0417"),
            ("Duitse rekening de89-3704-0044-0532-0130-00 ok", "iban", "0532"),
            ("IBAN NL91\u200bABNA\u200b0417\u200b1643\u200b00 ok", "iban", "0417"),
            ("iban mt84malt011000012345mtlcast001s ok", "iban", "0123"),
            ("mail sanne\uff20gmail.com ok", "email", "gmail"),
            ("mail mehmet [at] ziggo [dot] nl ok", "email", "ziggo"),
            ("mail jan (AT) example.co.uk ok", "email", "example"),
        ],
    )
    async def test_numbers_in_other_formats_are_masked(self, text, kind, secret):
        found = await answer(PIILens(kinds=["iban", "email"]), turn_context(text))
        assert found.value == [kind]
        masked = mask(text, found.spans)
        assert secret not in masked and masked.endswith(" ok") or masked.endswith("graag")

    @pytest.mark.parametrize(
        "text",
        [
            "offerte 2026-1187",
            "Totaal 1.170,00",
            "op 2026/11/17 om 14:00",
            "INC-0152",
            "Garage invoice no. 202611188 for the door",
            "factuurnummer: 202611188",
            "Order #202611188",
            "bestelnummer 202611188",
            "afspraak 03-11-2026 14:00",
            "op 01.02.2026 09:00",
            "Ik ben (at) home om 5 uur",
            "we meet at the office dot com later",
            "kijk op punt 3 en punt 4",
        ],
    )
    async def test_dates_amounts_and_numbers_are_not_masked(self, text):
        found = await answer(PIILens(kinds=["iban", "email"]), turn_context(text))
        assert found.value == []

    async def test_a_card_followed_by_its_expiry_is_masked(self):
        text = "card 4111 1111 1111 1111 2027 please"
        found = await answer(PIILens(kinds=["credit_card"]), turn_context(text))
        assert mask(text, found.spans) == "card [CREDIT_CARD] 2027 please"

    async def test_a_shape_that_fails_its_checksum_stays_unmasked(self):
        found = await answer(PIILens(kinds=["iban"]), turn_context("BE69539007547034 EUR"))
        assert found.value == []

    @pytest.mark.parametrize(
        "text",
        [
            "Where is my password: settings or profile?",
            "I forgot my password: changed it yesterday",
            "Wachtwoord: vergeten, wat nu?",
        ],
    )
    async def test_prose_about_a_password_is_not_one(self, text):
        assert (await answer(SecretsLens(), turn_context(text))).value == []

    @pytest.mark.parametrize(
        "text",
        [
            "password=hunter2",
            'password: "correcthorse"',
            '{"password": "hunter2!"}',
            '{"pwd": "s3cret!"}',
            '{"db_password":"abcdef"}',
            'password: "correct horse battery staple"',
        ],
    )
    async def test_a_password_assignment_is_one(self, text):
        assert (await answer(SecretsLens(), turn_context(text))).value == ["password_assignment"]

    async def test_a_private_key_is_masked_whole(self):
        from django_ai_sdk.oversight.patterns import mask_secrets

        pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIBOgIBAAJBAKj\n-----END RSA PRIVATE KEY-----"
        assert mask_secrets(f"key: {pem} ok") == "key: [PRIVATE_KEY] ok"
        assert mask_secrets("-----BEGIN OPENSSH PRIVATE KEY-----\nb3Blbn") == "[PRIVATE_KEY]"


class TestPIIKinds:
    TEXT = "BSN 111222333, bel 06-12345678 of +31 6 12345678, Damrak 1, 1012 LG Amsterdam"

    async def test_a_national_number_needs_a_kind_of_its_own(self):
        assert (await answer(PIILens(), turn_context(self.TEXT))).value == ["phone"]

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
        assert (await answer(SecretsLens(), turn_context(f"use {key} now"))).value == [label]


@pytest.fixture
def member_id_kind():
    from django_ai_sdk.oversight import patterns

    tables = (PII_PATTERNS, PII_VERIFY, PII_NOT_AFTER)
    saved = [dict(table) for table in tables]
    saved_compiled = list(patterns._PII)
    register_pii_kind(
        "member_id",
        r"\bM-\d{4}\b",
        verify=lambda value: value.endswith("7"),
        not_after=r"(?i)\bticket\s*$",
    )
    yield
    for table, copy in zip(tables, saved, strict=True):
        table.clear()
        table.update(copy)
    patterns._PII[:] = saved_compiled


@pytest.mark.usefixtures("member_id_kind")
class TestRegisterPIIKind:
    async def test_a_named_kind_is_found_when_it_passes_its_check(self):
        lens = PIILens(kinds=["member_id"])
        found = await answer(lens, turn_context("member M-1237, not M-1234"))
        assert found.value == ["member_id"]
        assert len(found.spans) == 1

    async def test_a_label_right_before_it_makes_it_no_match(self):
        found = await answer(PIILens(kinds=["member_id"]), turn_context("ticket M-1237"))
        assert found.value == []

    def test_it_stays_off_until_a_lens_names_it(self):
        assert "member_id" not in PIILens().finder.labels()

    def test_a_stored_definition_may_name_it(self):
        lens = lens_from_dict(
            {"kind": "pattern", "preset": "pii", "name": "p", "kinds": ["member_id"]}
        )
        assert lens.finder.labels() == ["member_id"]

    def test_receipts_and_the_record_see_it(self):
        assert finds_personal_or_secret("member M-1237")
        assert not finds_personal_or_secret("member M-1234")


def even(text):
    if int(text) % 2:
        raise Violation("Odd amounts are not allowed.")


def short_answers_only(text, context):
    if len(context.output) > 5:
        raise Violation("The answer is long.")


class TestCheckLens:
    async def test_a_violation_is_a_yes_with_its_message_as_the_reason(self):
        lens = CheckLens(name="even", validators=[even])
        odd = await answer(lens, turn_context("3"))
        assert (odd.value, odd.rationale, odd.answerer) == (
            True,
            "Odd amounts are not allowed.",
            "check",
        )
        assert (await answer(lens, turn_context("4"))).value is False

    async def test_a_validator_may_read_the_turn(self):
        lens = CheckLens(name="short", subject=Subject.ANSWER, validators=[short_answers_only])
        assert (await answer(lens, turn_context(output="a long answer"))).value is True
        assert (await answer(lens, turn_context(output="ok"))).value is False

    async def test_validators_by_name(self):
        register_validator("test_even")(even)
        lens = CheckLens(name="even", validators=["test_even"])
        assert (await answer(lens, turn_context("5"))).value is True

    def test_an_unknown_validator_is_refused(self):
        with pytest.raises(ImproperlyConfigured, match="No validator"):
            CheckLens(name="x", validators=["nobody_registered_this"]).validate()

    def test_validators_answer_a_flag(self):
        with pytest.raises(ImproperlyConfigured, match="validators answer a Flag"):
            CheckLens(name="x", validators=[even], answer=Score).validate()
        FnLens(name="x", answer=Score, fn=lambda text, context: 0.5).validate()

    def test_a_check_lens_needs_validators_or_its_own_check(self):
        with pytest.raises(ImproperlyConfigured, match="no `validators`"):
            CheckLens(name="x").validate()

    async def test_presets_answer_facts_about_the_turn(self):
        long = await answer(LengthLens(max_chars=3), turn_context("four"))
        assert (long.value, long.rationale) == (True, "4 characters")
        mismatch = await answer(
            LanguageLens(),
            turn_context(
                "Wat is het adres van de winkel?", "The shop is on the main street and it is open."
            ),
        )
        assert mismatch.value is True
        looked = TurnContext(tool_calls=[ToolCall(name="search", result=None)])
        assert (await answer(EmptyRetrievalLens(), looked)).value is True
        assert EmptyRetrievalLens().applies(looked) and not EmptyRetrievalLens().applies(
            TurnContext()
        )
        calls = [ToolCall(name="a", error=True), ToolCall(name="b")]
        assert (await answer(ToolErrorsLens(), TurnContext(tool_calls=calls))).value == 0.5
        assert await answer(ToolErrorsLens(), TurnContext()) is None


class TestFlag:
    def test_a_dutch_word_is_not_a_yes(self):
        with pytest.raises(ValueError):
            Flag.parse(JudgeLens(name="x"), "ja")


class TestAttributesBelongWhereTheyAreRead:
    def test_an_attribute_no_lens_of_its_kind_has(self):
        with pytest.raises(ImproperlyConfigured, match="judges"):
            CheckLens(name="x", validators=[even], judges=["judge"])

    def test_an_option_of_another_answer_type(self):
        with pytest.raises(ValueError, match="labels"):
            JudgeLens(name="f", judges=["judge"], labels=["a"])

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
                "judges": [{"alias": "judge", "optional": True}],
                "stages": ["reflect"],
            }
        )
        assert isinstance(lens, JudgeLens) and lens.answer is Choice
        assert lens.stages == (Stage.REFLECT,)

    def test_an_optional_alias_has_one_json_form(self):
        lens = lens_from_dict(
            {"kind": "judge", "name": "t", "judges": [{"alias": "judge", "optional": True}]}
        )
        assert lens.judges == [Maybe("judge")]
        with pytest.raises(ImproperlyConfigured, match="optional alias"):
            lens_from_dict(
                {"kind": "judge", "name": "t", "judges": [{"alias": "judge", "optional": False}]}
            )

    def test_a_preset(self):
        lens = lens_from_dict({"kind": "pattern", "preset": "pii", "name": "p", "kinds": ["iban"]})
        assert isinstance(lens, PIILens) and lens.kinds == ["iban"]
        with pytest.raises(ImproperlyConfigured, match="not a check one"):
            lens_from_dict({"kind": "check", "preset": "pii", "name": "p"})

    def test_an_unknown_kind_or_answer(self):
        with pytest.raises(ImproperlyConfigured, match="lens kind"):
            lens_from_dict({"kind": "flag", "name": "x"})
        with pytest.raises(ImproperlyConfigured, match="Unknown answer"):
            lens_from_dict({"kind": "judge", "name": "x", "answer": "maybe", "judges": ["j"]})
