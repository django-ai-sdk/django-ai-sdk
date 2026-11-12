"""What oversight keeps where: everything that can quote a turn lives with its content.

Per capture level, the rows (turn, lens answer, rule match) and the review items
hold no text of the turn; the content holds it as capture says;
forgetting a thread takes it, and who asked."""

from __future__ import annotations

import base64
import json

import pytest
from asgiref.sync import sync_to_async
from django.test import override_settings
from django_ai_sdk.common import ChatMessage
from django_ai_sdk.oversight import (
    Entities,
    JudgeLens,
    L,
    Mask,
    Mode,
    PIILens,
    Policy,
    Queue,
    Rule,
    register_policy,
)
from django_ai_sdk.oversight.content import (
    FOUND,
    load_content,
    masker_for_capture,
    stored_value,
    unmasked_literals,
)
from django_ai_sdk.oversight.guard import TurnGuard
from django_ai_sdk.oversight.models import LensAnswer, ReviewItem, RuleMatch, Turn
from django_ai_sdk.oversight.patterns import DEFAULT_PII_KINDS
from django_ai_sdk.oversight.persist import split_answer
from django_ai_sdk.oversight.reflect import areflect
from django_ai_sdk.oversight.types import Answer, Stage, Subject, TextSpan, ToolCall
from haystack.dataclasses import ChatMessage as HaystackChatMessage

from tests.oversight.helpers import (
    FnLens,
    Storage,
    chat,
    closed_turn,
    make_agent,
    oversight_parts,
)

EMAIL = "jan@example.com"


BSN = "111222333"  # passes the 11-check: a country kind no lens of the policy names


KEY = "sk-proj-abcdefghijklmnopqrstuvwxyz0123456789ABCD"


NAME = "Jan Jansen"  # found by a judge, not by any detector


IBAN = "NL91ABNA0417164300"


SECRETS = (EMAIL, BSN, KEY, NAME, "Jansen")


MASTER = base64.b64encode(b"k" * 32).decode()


OTHER = base64.b64encode(b"o" * 32).decode()


def _naming_judge(text, view):
    # A judge that names who it found in its own words (no `labels`), and says where.
    at = view.input.find(NAME)
    if at < 0:
        return Answer(value=[])
    return Answer(value=[NAME], spans=[TextSpan(start=at, end=at + len(NAME), label="person")])


def _judged(text, view):
    return Answer(value=True, confidence=0.9, rationale=f"It quotes {NAME}, {EMAIL}, {BSN}, {KEY}")


def _taken(text, view):
    # Like TypeSafe.per_source: the titles of the sources, each one's score by its key.
    return Answer(
        value=[s.title for s in view.sources],
        confidence=0.9,
        items={s.key: 0.9 for s in view.sources},
    )


def _later(text, view):
    return Answer(value=True, rationale=f"Later: {EMAIL} again")


@register_policy
class PrivacyPolicy(Policy):
    name = "test-privacy"
    lenses = [  # noqa: RUF012
        PIILens(name="pii_in", stages=[Stage.GATE], kinds=[*DEFAULT_PII_KINDS, "bsn"]),
        FnLens(name="who", answer=Entities, stages=[Stage.GATE], fn=_naming_judge),
        FnLens(name="judged", subject=Subject.OUTPUT, stages=[Stage.FINISH], fn=_judged),
        FnLens(
            name="taken", answer=Entities, subject=Subject.OUTPUT, stages=[Stage.FINISH], fn=_taken
        ),
        FnLens(name="later", subject=Subject.OUTPUT, stages=[Stage.REFLECT], fn=_later),
    ]
    rules = [  # noqa: RUF012
        # Report mode: nothing is masked by a rule, so capture alone decides.
        Rule("mask_pii_in", when=L("pii_in").is_true, then=Mask(), mode=Mode.REPORT),
        Rule("queue_judged", when=L("judged").is_true, then=Queue("q"), mode=Mode.ENFORCE),
    ]


@register_policy
class NamingPolicy(Policy):
    name = "test-privacy-naming"
    lenses = [  # noqa: RUF012
        FnLens(
            name="who",
            answer=Entities,
            stages=[Stage.GATE],
            fn=lambda text, v: (
                Answer(
                    value=[EMAIL],
                    spans=[
                        TextSpan(
                            start=v.input.find(EMAIL),
                            end=v.input.find(EMAIL) + len(EMAIL),
                            label="email",
                        )
                    ],
                )
                if EMAIL in v.input
                else Answer(value=[])
            ),
        )
    ]
    rules = [Rule("mask_who", when=L("who").is_true, then=Mask(), mode=Mode.ENFORCE)]  # noqa: RUF012


async def run_turn(
    capture: str = "full", *, thread_id: str | None = None, user=None, **settings
) -> Turn:
    with override_settings(AI_SDK_OVERSIGHT={"CAPTURE": capture, **settings}):
        messages = [
            ChatMessage(role="user", content=f"Earlier: I am {NAME}, mail {EMAIL}, key {KEY}"),
            ChatMessage(role="assistant", content=f"Noted, {NAME}, BSN {BSN}"),
            ChatMessage(role="user", content=f"Now: {NAME} {EMAIL} BSN {BSN}"),
        ]
        guard = await TurnGuard.start(
            make_agent([], None, policy="test-privacy"),
            channel="api",
            messages=messages,
            thread_id=thread_id,
            user=user,
        )
        assert guard.active
        await guard.gate()
        guard.set_sources(
            [
                {
                    "key": f"https://crm/{EMAIL}",
                    "title": f"File of {NAME} {EMAIL}",
                    "url": f"https://crm/{EMAIL}",
                    "content": f"{NAME}: mail {EMAIL}, BSN {BSN}, {KEY}",
                    "kind": "doc",
                }
            ]
        )
        guard.add_tool_call(
            ToolCall(
                id="c1",
                name="lookup",
                arguments={"q": f"{NAME} {EMAIL} {BSN}"},
                result=f"Found {NAME}, {EMAIL}, BSN {BSN}, key {KEY}",
            )
        )
        await guard.finish(f"Answer for {NAME}: {EMAIL}, BSN {BSN}")
        return await closed_turn(guard)


def _rows(turn: Turn) -> str:
    """Every row of the record, and the review items, as one string."""
    found = [
        *Turn.objects.filter(pk=turn.pk).values(),
        *LensAnswer.objects.filter(turn=turn).values(),
        *RuleMatch.objects.filter(turn=turn).values(),
        *ReviewItem.objects.filter(turn=turn).values(),
    ]
    return json.dumps(found, default=str)


def _leaks(text: str) -> list[str]:
    return [secret for secret in SECRETS if secret in text]


async def _kept(turn: Turn) -> dict:
    """What the turn's content holds now, read again."""
    fresh = await Turn.objects.select_related("content").aget(pk=turn.pk)
    return await sync_to_async(load_content)(fresh) or {}


def _by_lens(turn: Turn) -> dict[str, LensAnswer]:
    return {o.lens: o for o in LensAnswer.objects.filter(turn=turn)}


@pytest.mark.django_db(transaction=True)
class TestWhatCaptureKeeps:
    @pytest.mark.parametrize("capture", ["none", "metadata", "masked", "full"])
    async def test_the_rows_hold_no_text_of_the_turn(self, capture):
        turn = await run_turn(capture)
        assert _leaks(await sync_to_async(_rows)(turn)) == []
        answer_rows = await sync_to_async(_by_lens)(turn)
        # A detector's kinds stay; names a judge gave in its own words are counted.
        assert answer_rows["pii_in"].value == ["bsn", "email"]
        assert answer_rows["who"].value == [FOUND]
        assert answer_rows["taken"].value == [FOUND]
        assert turn.labels["who"] == turn.labels["taken"] == [FOUND]

    @pytest.mark.parametrize("capture", ["none", "metadata"])
    async def test_none_and_metadata_keep_no_text_anywhere(self, capture):
        turn = await run_turn(capture)
        kept = await _kept(turn)
        assert _leaks(json.dumps(kept)) == []
        assert "File of" not in json.dumps(kept) and "crm" not in json.dumps(kept)
        assert "texts" not in kept
        if capture == "metadata":
            assert kept["sources"] == [{"kind": "doc", "seeded": False}]

    async def test_masked_capture_masks_every_string_it_keeps(self):
        turn = await run_turn("masked")
        kept = await _kept(turn)
        assert _leaks(json.dumps(kept)) == []
        assert kept["input"] == "Now: [PERSON] [EMAIL] BSN [BSN]"
        assert (
            kept["history"][0]["content"] == "Earlier: I am [PERSON], mail [EMAIL], key [API_KEY]"
        )
        [source] = kept["sources"]
        assert (source["key"], source["title"]) == (
            "https://crm/[EMAIL]",
            "File of [PERSON] [EMAIL]",
        )
        [call] = kept["tool_calls"]
        assert call["result"] == "Found [PERSON], [EMAIL], BSN [BSN], key [API_KEY]"
        assert call["arguments"] == {"q": "[PERSON] [EMAIL] [BSN]"}
        answer_rows = await sync_to_async(_by_lens)(turn)
        texts = kept["texts"]
        assert texts[str(answer_rows["judged"].id)] == {
            "rationale": "It quotes [PERSON], [EMAIL], [BSN], [API_KEY]"
        }
        assert texts[str(answer_rows["who"].id)]["value"] == ["[PERSON]"]
        assert texts[str(answer_rows["taken"].id)] == {
            "value": ["File of [PERSON] [EMAIL]"],
            "items": {"https://crm/[EMAIL]": 0.9},
        }

    async def test_full_capture_keeps_the_words_with_the_content_only(self):
        turn = await run_turn("full")
        kept = await _kept(turn)
        answer_rows = await sync_to_async(_by_lens)(turn)
        texts = kept["texts"]
        assert texts[str(answer_rows["who"].id)]["value"] == [NAME]
        assert texts[str(answer_rows["judged"].id)]["rationale"].startswith(f"It quotes {NAME}")

    async def test_reflect_keeps_its_words_like_close_did(self):
        turn = await run_turn("masked")
        assert await areflect(str(turn.id)) == 1
        later = await LensAnswer.objects.aget(turn=turn, lens="later")
        assert later.value is True
        assert (await _kept(turn))["texts"][str(later.id)] == {"rationale": "Later: [EMAIL] again"}

    def test_what_masking_replaced_is_read_back_from_the_masked_copy(self):
        masked, shown = "Mail [EMAIL] or [PERSON] [1]", f"Mail {EMAIL} or {NAME} [1]"
        assert unmasked_literals(masked, shown) == {EMAIL: "email", NAME: "person"}
        assert unmasked_literals("Hi [PERSON]", "Hi [PERSON]") == {}
        assert unmasked_literals("Hi [PERSON]", "Bye") == {}


@pytest.mark.django_db(transaction=True)
class TestWhatTheClientSees:
    async def test_client_detail_names_nothing_a_mask_took_out(self):
        with override_settings(AI_SDK_OVERSIGHT={"CLIENT_DETAIL": "full"}):
            storage = Storage()
            agent = make_agent(
                [HaystackChatMessage.from_assistant("ok")], storage, policy="test-privacy-naming"
            )
            parts = await chat(agent, f"Mail {EMAIL} please")
        [gate, *_] = oversight_parts(parts)
        assert gate["input"] == "Mail [EMAIL] please"
        assert gate["detail"]["answers"][0]["value"] == [FOUND]
        assert EMAIL not in json.dumps(parts)
        assert EMAIL not in json.dumps([m.model_dump() for m in storage.stored], default=str)


class TestTheRecordsValue:
    def test_an_entities_value_keeps_labels_and_counts_names(self):
        from django_ai_sdk.oversight.packs.quality import TakenFrom
        from django_ai_sdk.oversight.packs.security import PIIInput

        titles = Answer(value=[f"Dossier {EMAIL} (BSN {BSN})"], answerer="typesafe")
        assert stored_value(TakenFrom(), titles) == [FOUND]
        # Patterns report kinds; a person, like a judge, may type anything.
        assert stored_value(PIIInput(), Answer(value=["email"], answerer="patterns")) == ["email"]
        assert stored_value(PIIInput(), Answer(value=["email"], answerer="human")) == [FOUND]
        labelled = JudgeLens(name="kind", answer=Entities, labels=["hr", "pay"])
        assert stored_value(labelled, Answer(value=["hr", NAME], answerer="judge")) == ["hr", FOUND]
        assert stored_value(JudgeLens(name="f"), Answer(value=True)) is True

    def test_a_reason_and_its_scores_are_kept_as_capture_says(self):
        lens = JudgeLens(name="f")
        answer = Answer(
            value=True,
            confidence=0.9,
            rationale=f"It quotes {IBAN}.",
            items={f"https://crm/{EMAIL}": 0.95},
        )
        columns, masked = split_answer(lens, "finish", answer, keep=masker_for_capture("masked"))
        assert "rationale" not in columns and "items" not in columns
        assert masked == {"rationale": "It quotes [IBAN].", "items": {"https://crm/[EMAIL]": 0.95}}
        assert split_answer(lens, "finish", answer, keep=masker_for_capture("full"))[1] == {
            "rationale": f"It quotes {IBAN}.",
            "items": {f"https://crm/{EMAIL}": 0.95},
        }
        for capture in ("metadata", "none"):
            assert split_answer(lens, "finish", answer, keep=masker_for_capture(capture))[1] == {}

    def test_a_found_name_is_masked_before_the_detectors_cut_it_apart(self):
        keep = masker_for_capture("masked", {f"Jan Jansen <{EMAIL}>": "person"})
        assert keep.text(f"Mail Jan Jansen <{EMAIL}> today.") == "Mail [PERSON] today."

    def test_an_earlier_answerers_names_stay_out_of_the_trail(self):
        lens = JudgeLens(name="who", answer=Entities)
        earlier = Answer(value=[NAME], confidence=0.4, answerer="judge")
        answer = Answer(value=[NAME, "Piet"], answerer="strong", trail=[earlier])
        columns, kept = split_answer(lens, "reflect", answer, keep=masker_for_capture("full"))
        assert [step["value"] for step in columns["trail"]] == [[FOUND]]
        assert kept == {"value": [NAME, "Piet"], "trail": [[NAME]]}


class TestDeployCheck:
    @staticmethod
    def capture_ids() -> list[str]:
        from django_ai_sdk.oversight.checks import check_oversight_deploy

        # The test settings run tasks inline (W014): only what capture says counts here.
        return [m.id for m in check_oversight_deploy() if m.id != "ai_sdk.oversight.W014"]

    def test_full_capture_warns(self):
        with override_settings(AI_SDK_OVERSIGHT={"CAPTURE": "full"}):
            assert self.capture_ids() == ["ai_sdk.oversight.W001"]

    def test_masked_capture_says_nothing(self):
        assert self.capture_ids() == []


def test_the_admin_shows_no_content_and_nobody():
    from django.contrib import admin
    from django.contrib.auth.models import AnonymousUser
    from django.test import RequestFactory, modify_settings

    with modify_settings(INSTALLED_APPS={"append": "django.contrib.admin"}):
        from django_ai_sdk.oversight.admin import ReviewItemAdmin, TurnAdmin

        request = RequestFactory().get("/")
        request.user = AnonymousUser()
        turn = TurnAdmin(Turn, admin.site).get_fields(request)
        review = ReviewItemAdmin(ReviewItem, admin.site).get_fields(request)
    for field in ("content", "actor_id", "thread_id", "user_message_id", "message_id"):
        assert field not in turn
    assert "note" not in review and "resolved_by_id" not in review
