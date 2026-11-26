from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, Mock, patch

import pytest
from django.core.cache import cache
from django.http import HttpResponse, JsonResponse
from django.test import override_settings
from django_ai_sdk.integrations.mcp.loader import DynamicMCPIntegration
from django_ai_sdk.integrations.mcp.schemas import StaticMCPIntegrationConfig
from django_ai_sdk.integrations.registry import register, reset_registry
from django_ai_sdk.integrations.webhooks.base import (
    InboundEvent,
    WebhookIntegration,
    current_event,
)
from django_ai_sdk.integrations.webhooks.tasks import ERROR_MESSAGE, TIMEOUT_MESSAGE, run_inbound
from django_ai_sdk.integrations.webhooks.views import receive


@pytest.fixture(autouse=True)
def _clean_registry():
    reset_registry()
    cache.clear()
    yield
    reset_registry()
    cache.clear()


@pytest.fixture
def enqueue():
    stub = Mock(aenqueue=AsyncMock())
    with patch("django_ai_sdk.integrations.webhooks.views.handle_inbound", stub):
        yield stub.aenqueue


class DemoWebhookIntegration(WebhookIntegration):
    name = "demo"
    label = "Demo"
    agent = "tests.unit.test_webhook_integrations.DemoAgent"

    def __init__(self) -> None:
        self.verified = True
        self.parsed: object = "event"
        self.replies: list[str] = []

    def verify(self, request):
        return self.verified

    def parse(self, request):
        if self.parsed == "event":
            body = json.loads(request.body)
            return InboundEvent(
                integration=self.name,
                event_id=body["event_id"],
                text=body["text"],
                conversation_id="C1",
                external_user_id=body["sender"],
                workspace_id=body["workspace"],
            )
        return self.parsed

    async def reply(self, event, text):
        self.replies.append(text)


class FakeAgent:
    _agent_id = "fake-agent"


def _post(rf, name="demo", event_id="Ev1", text="hello", sender="U1", workspace="T1"):
    return rf.post(
        f"/api/integrations/{name}/webhook/",
        data=json.dumps(
            {"event_id": event_id, "text": text, "sender": sender, "workspace": workspace}
        ),
        content_type="application/json",
    )


class TestReceiveGuard:
    async def test_an_unknown_name_is_not_found(self, rf):
        response = await receive(_post(rf, name="nope"), "nope")
        assert response.status_code == 404

    async def test_a_database_backed_mcp_row_cannot_be_reached(self, rf):
        row = DynamicMCPIntegration(
            "demo", StaticMCPIntegrationConfig(url="https://example.com/mcp")
        )
        register(row)
        response = await receive(_post(rf), "demo")
        assert response.status_code == 404

    async def test_an_unconfigured_integration_is_not_found(self, rf):
        integration = DemoWebhookIntegration()
        integration.agent = ""
        register(integration)
        response = await receive(_post(rf), "demo")
        assert response.status_code == 404

    async def test_a_bad_signature_is_unauthorized(self, rf):
        integration = DemoWebhookIntegration()
        integration.verified = False
        register(integration)
        response = await receive(_post(rf), "demo")
        assert response.status_code == 401

    async def test_a_get_is_not_found(self, rf):
        register(DemoWebhookIntegration())
        response = await receive(rf.get("/api/integrations/demo/webhook/"), "demo")
        assert response.status_code == 404


class TestParseOutcomes:
    async def test_none_acknowledges_without_queueing(self, rf, enqueue):
        integration = DemoWebhookIntegration()
        integration.parsed = None
        register(integration)
        response = await receive(_post(rf), "demo")
        assert response.status_code == 200
        enqueue.assert_not_awaited()

    async def test_a_response_is_returned_verbatim(self, rf):
        # Slack's url_verification handshake wants a body, not a bare 200.
        integration = DemoWebhookIntegration()
        integration.parsed = JsonResponse({"challenge": "abc"})
        register(integration)
        response = await receive(_post(rf), "demo")
        assert isinstance(response, HttpResponse)
        assert json.loads(response.content) == {"challenge": "abc"}

    async def test_an_event_is_queued(self, rf, enqueue):
        register(DemoWebhookIntegration())
        response = await receive(_post(rf), "demo")
        assert response.status_code == 200
        payload = enqueue.await_args.args[0]
        assert payload["text"] == "hello"
        assert payload["integration"] == "demo"

    @pytest.mark.parametrize("text", ["", "   "])
    async def test_an_event_without_text_is_not_queued(self, rf, enqueue, text):
        register(DemoWebhookIntegration())
        response = await receive(_post(rf, text=text), "demo")
        assert response.status_code == 200
        enqueue.assert_not_awaited()

    async def test_the_acknowledgement_body_is_the_integrations_own(self, rf, enqueue):
        # A platform that shows the asker a placeholder needs a body in the ack.
        integration = DemoWebhookIntegration()
        integration.ack = lambda event: JsonResponse({"type": 5})
        register(integration)
        response = await receive(_post(rf), "demo")
        assert json.loads(response.content) == {"type": 5}


class TestAccess:
    async def test_an_unset_allowlist_answers_anyone(self, rf, enqueue):
        register(DemoWebhookIntegration())
        await receive(_post(rf), "demo")
        enqueue.assert_awaited_once()

    async def test_a_listed_sender_is_answered(self, rf, enqueue):
        register(DemoWebhookIntegration())
        with override_settings(AI_SDK_INTEGRATIONS={"demo": {"ALLOW_FROM": ["U1"]}}):
            await receive(_post(rf, sender="U1"), "demo")
        enqueue.assert_awaited_once()

    async def test_an_unlisted_sender_is_acknowledged_and_dropped(self, rf, enqueue):
        # 200, not 403: a refusal the platform reads as failure earns a redelivery.
        register(DemoWebhookIntegration())
        with override_settings(AI_SDK_INTEGRATIONS={"demo": {"ALLOW_FROM": ["U1"]}}):
            response = await receive(_post(rf, sender="U2"), "demo")
        assert response.status_code == 200
        enqueue.assert_not_awaited()

    async def test_the_any_sentinel_answers_anyone(self, rf, enqueue):
        register(DemoWebhookIntegration())
        with override_settings(AI_SDK_INTEGRATIONS={"demo": {"ALLOW_FROM": "*"}}):
            await receive(_post(rf, sender="U9"), "demo")
        enqueue.assert_awaited_once()

    async def test_the_any_sentinel_inside_a_list_answers_anyone(self, rf, enqueue):
        register(DemoWebhookIntegration())
        with override_settings(AI_SDK_INTEGRATIONS={"demo": {"ALLOW_FROM": ["*"]}}):
            await receive(_post(rf, sender="U9"), "demo")
        enqueue.assert_awaited_once()

    async def test_a_malformed_allowlist_refuses_everyone(self, rf, enqueue):
        register(DemoWebhookIntegration())
        with override_settings(AI_SDK_INTEGRATIONS={"demo": {"ALLOW_FROM": "U1"}}):
            await receive(_post(rf, sender="U1"), "demo")
        enqueue.assert_not_awaited()


class TestWorkspaceAccess:
    async def test_a_listed_workspace_is_answered(self, rf, enqueue):
        register(DemoWebhookIntegration())
        with override_settings(AI_SDK_INTEGRATIONS={"demo": {"ALLOW_WORKSPACES": ["T1"]}}):
            await receive(_post(rf, workspace="T1"), "demo")
        enqueue.assert_awaited_once()

    async def test_another_installation_is_dropped(self, rf, enqueue):
        # A signing secret belongs to the app, not the installation, so a second
        # workspace verifies cleanly and would otherwise run the agent for free.
        register(DemoWebhookIntegration())
        with override_settings(AI_SDK_INTEGRATIONS={"demo": {"ALLOW_WORKSPACES": ["T1"]}}):
            await receive(_post(rf, workspace="T2"), "demo")
        enqueue.assert_not_awaited()

    async def test_a_malformed_workspace_list_refuses_everyone(self, rf, enqueue):
        register(DemoWebhookIntegration())
        with override_settings(AI_SDK_INTEGRATIONS={"demo": {"ALLOW_WORKSPACES": 7}}):
            await receive(_post(rf, workspace="T1"), "demo")
        enqueue.assert_not_awaited()

    async def test_both_lists_must_admit_the_event(self, rf, enqueue):
        register(DemoWebhookIntegration())
        config = {"demo": {"ALLOW_FROM": ["U1"], "ALLOW_WORKSPACES": ["T1"]}}
        with override_settings(AI_SDK_INTEGRATIONS=config):
            await receive(_post(rf, sender="U1", workspace="T2"), "demo")
        enqueue.assert_not_awaited()


class TestTextLimit:
    async def test_a_long_message_is_truncated(self, rf, enqueue):
        register(DemoWebhookIntegration())
        with override_settings(AI_SDK_WEBHOOK_MAX_TEXT=10):
            await receive(_post(rf, text="x" * 500), "demo")
        assert enqueue.await_args.args[0]["text"] == "x" * 10


class TestRateLimit:
    @pytest.mark.django_db
    @override_settings(
        AI_SDK_WEBHOOK_RATE_LIMIT_PER_MINUTE=2,
        CACHES={
            "default": {
                "BACKEND": "django.core.cache.backends.db.DatabaseCache",
                "LOCATION": "webhook_rate_cache",
            }
        },
    )
    def test_a_sender_over_the_limit_is_dropped(self, rf, enqueue):
        # DatabaseCache, as in production: a sync cache call on the event loop would raise.
        from asgiref.sync import async_to_sync
        from django.core.management import call_command

        call_command("createcachetable", verbosity=0)
        register(DemoWebhookIntegration())
        codes = [
            async_to_sync(receive)(_post(rf, event_id=f"Ev{n}"), "demo").status_code
            for n in range(3)
        ]
        assert codes == [200, 200, 200]
        assert enqueue.await_count == 2

    async def test_other_senders_are_not_held_back(self, rf, enqueue):
        register(DemoWebhookIntegration())
        with override_settings(AI_SDK_WEBHOOK_RATE_LIMIT_PER_MINUTE=1):
            await receive(_post(rf, event_id="Ev1", sender="U1"), "demo")
            await receive(_post(rf, event_id="Ev2", sender="U2"), "demo")
        assert enqueue.await_count == 2


class TestEventTriggers:
    @staticmethod
    async def _handle(event_id="Ev7", text="new invoice"):
        integration = DemoWebhookIntegration()
        register(integration)
        agent = Mock(run=AsyncMock(return_value="ahoy"), storage_adapter=None)
        payload = InboundEvent(
            integration="demo", event_id=event_id, text=text, conversation_id="C1"
        ).model_dump()
        with patch.object(DemoWebhookIntegration, "resolve_agent", return_value=agent):
            await run_inbound(payload)
        return integration

    async def test_an_event_starts_the_workflows_waiting_for_it(self):
        start = AsyncMock(return_value=[])
        with patch("django_ai_sdk.workflows.triggers.start_for_event", start):
            await self._handle()
        start.assert_awaited_once_with(
            "demo.message",
            {"text": "new invoice", "conversation_id": "C1"},
            idempotency_key="demo:Ev7",
        )

    async def test_the_request_starts_nothing(self, rf, enqueue):
        register(DemoWebhookIntegration())
        start = AsyncMock(return_value=[])
        with patch("django_ai_sdk.workflows.triggers.start_for_event", start):
            await receive(_post(rf), "demo")
        start.assert_not_awaited()
        assert enqueue.await_count == 1

    async def test_a_repeat_starts_nothing_again(self):
        start = AsyncMock(return_value=[])
        with patch("django_ai_sdk.workflows.triggers.start_for_event", start):
            await self._handle()
            await self._handle()
        assert start.await_count == 1

    async def test_a_failing_trigger_still_answers(self):
        start = AsyncMock(side_effect=RuntimeError("queue down"))
        with patch("django_ai_sdk.workflows.triggers.start_for_event", start):
            integration = await self._handle()
        assert integration.replies == ["ahoy"]

    async def test_a_failing_receiver_still_answers(self, caplog):
        from django_ai_sdk.integrations.webhooks.signals import event_received

        async def broken(sender, event, **kwargs):
            raise RuntimeError("receiver bug")

        event_received.connect(broken, dispatch_uid="test_broken_receiver")
        try:
            integration = await self._handle()
        finally:
            event_received.disconnect(dispatch_uid="test_broken_receiver")
        assert integration.replies == ["ahoy"]
        assert "receiver" in caplog.text


class TestDeduplication:
    @staticmethod
    async def _runs(*event_ids, reply_error=None):
        integration = DemoWebhookIntegration()
        register(integration)
        if reply_error:
            integration.reply = AsyncMock(side_effect=reply_error)
        agent = Mock(run=AsyncMock(return_value="ahoy"), storage_adapter=None)
        with patch.object(DemoWebhookIntegration, "resolve_agent", return_value=agent):
            for event_id in event_ids:
                payload = InboundEvent(
                    integration="demo", event_id=event_id, text="hi", conversation_id="C1"
                ).model_dump()
                try:
                    await run_inbound(payload)
                except OSError:
                    pass
        return agent.run.await_count

    async def test_every_delivery_is_queued(self, rf, enqueue):
        register(DemoWebhookIntegration())
        await receive(_post(rf, event_id="Ev9"), "demo")
        await receive(_post(rf, event_id="Ev9"), "demo")
        assert enqueue.await_count == 2

    async def test_a_repeated_event_id_runs_once(self):
        assert await self._runs("Ev9", "Ev9") == 1

    async def test_a_failed_run_is_not_run_again(self):
        assert await self._runs("Ev3", "Ev3", reply_error=OSError("platform is down")) == 1

    async def test_a_different_event_id_runs_again(self):
        assert await self._runs("Ev1", "Ev2") == 2

    async def test_an_event_without_an_id_is_answered_each_time(self):
        assert await self._runs("", "") == 2


class TestDedupCacheCheck:
    @staticmethod
    def _warnings(backend, configured=True):
        from django_ai_sdk.checks import check_webhook_dedup_cache

        register(DemoWebhookIntegration())
        with override_settings(
            CACHES={"default": {"BACKEND": backend}},
            AI_SDK_INTEGRATIONS={"demo": {}} if configured else {},
        ):
            return check_webhook_dedup_cache()

    def test_a_per_process_cache_is_warned_about(self):
        [warning] = self._warnings("django.core.cache.backends.locmem.LocMemCache")
        assert warning.id == "ai_sdk.W001"
        assert "demo" in warning.msg

    def test_a_shared_cache_passes(self):
        assert self._warnings("django.core.cache.backends.db.DatabaseCache") == []

    def test_an_integration_left_unconfigured_is_not_counted(self):
        backend = "django.core.cache.backends.locmem.LocMemCache"
        assert self._warnings(backend, configured=False) == []


class TestDedupWindow:
    def test_the_framework_default_is_ten_minutes(self):
        assert DemoWebhookIntegration().dedup_ttl == 600

    def test_an_integration_widens_it(self):
        class Unsigned(DemoWebhookIntegration):
            dedup_ttl = 7 * 24 * 60 * 60

        assert Unsigned().dedup_ttl == 7 * 24 * 60 * 60


class TestConfiguration:
    def test_the_config_slice_overrides_the_class_attribute(self):
        integration = DemoWebhookIntegration()
        with override_settings(AI_SDK_INTEGRATIONS={"demo": {"AGENT": "other.Agent"}}):
            assert integration.agent_path() == "other.Agent"

    def test_a_missing_agent_names_the_setting_to_change(self):
        integration = DemoWebhookIntegration()
        integration.agent = ""
        assert "AI_SDK_INTEGRATIONS['demo']['AGENT']" in integration.detail

    async def test_a_configured_integration_reports_active(self):
        integration = DemoWebhookIntegration()
        assert integration.detail is None
        assert (await integration.get_status()).value == "active"

    async def test_a_receive_only_integration_offers_no_tools(self):
        assert await DemoWebhookIntegration().get_tools() == []

    def test_an_unimportable_agent_path_resolves_to_none(self):
        integration = DemoWebhookIntegration()
        with override_settings(AI_SDK_INTEGRATIONS={"demo": {"AGENT": "no.such.module.Agent"}}):
            assert integration.resolve_agent() is None

    def test_a_path_that_is_not_an_agent_resolves_to_none(self, caplog):
        integration = DemoWebhookIntegration()
        with override_settings(AI_SDK_INTEGRATIONS={"demo": {"AGENT": "json.dumps"}}):
            assert integration.resolve_agent() is None
        assert "not a registered Agent subclass" in caplog.text


@pytest.mark.django_db(transaction=True)
class TestPrincipal:
    @staticmethod
    def _configured(value):
        return override_settings(AI_SDK_INTEGRATIONS={"demo": {"RUN_AS": value}})

    async def test_no_run_as_answers_anonymously(self):
        assert await DemoWebhookIntegration().aget_principal() is None

    async def test_a_configured_account_is_resolved(self):
        from tests.factories.db import UserFactory

        user = await UserFactory.acreate(email="bot@example.com")
        with self._configured("bot@example.com"):
            assert await DemoWebhookIntegration().aget_principal() == user

    async def test_an_unknown_account_answers_anonymously(self, caplog):
        with self._configured("nobody@example.com"):
            assert await DemoWebhookIntegration().aget_principal() is None
        assert "RUN_AS" in caplog.text

    async def test_a_deactivated_account_answers_anonymously(self):
        from tests.factories.db import UserFactory

        await UserFactory.acreate(email="retired@example.com", is_active=False)
        with self._configured("retired@example.com"):
            assert await DemoWebhookIntegration().aget_principal() is None

    async def test_an_ambiguous_account_answers_anonymously(self, caplog):
        from django.contrib.auth import get_user_model

        user_model = get_user_model()
        with (
            self._configured("bot@example.com"),
            patch.object(
                user_model._default_manager,
                "aget",
                AsyncMock(side_effect=user_model.MultipleObjectsReturned()),
            ),
        ):
            assert await DemoWebhookIntegration().aget_principal() is None
        assert "no single active user" in caplog.text


class TestInboundTask:
    @staticmethod
    def _payload():
        return InboundEvent(
            integration="demo", event_id="Ev1", text="hi", conversation_id="C1"
        ).model_dump()

    async def test_the_agent_answer_is_replied(self):
        integration = DemoWebhookIntegration()
        register(integration)
        agent = Mock(name="agent", run=AsyncMock(return_value="ahoy"), storage_adapter=None)
        with patch.object(DemoWebhookIntegration, "resolve_agent", return_value=agent):
            await run_inbound(self._payload())
        assert integration.replies == ["ahoy"]
        assert agent.run.await_args.kwargs["tools"] is True
        assert agent.run.await_args.kwargs["user"] is None
        assert agent.run.await_args.kwargs.get("thread_id") is None

    @pytest.mark.django_db(transaction=True)
    async def test_the_run_acts_as_the_service_account(self):
        from tests.factories.db import UserFactory

        user = await UserFactory.acreate(email="bot@example.com")
        integration = DemoWebhookIntegration()
        register(integration)
        agent = Mock(name="agent", run=AsyncMock(return_value="ahoy"), storage_adapter=None)
        with (
            patch.object(DemoWebhookIntegration, "resolve_agent", return_value=agent),
            override_settings(AI_SDK_INTEGRATIONS={"demo": {"RUN_AS": "bot@example.com"}}),
        ):
            await run_inbound(self._payload())
        assert agent.run.await_args.kwargs["user"] == user
        assert agent.run.await_args.kwargs["tools"] is True

    async def test_the_event_is_current_during_the_run(self):
        register(DemoWebhookIntegration())
        seen = []

        async def run(*args, **kwargs):
            seen.append(current_event.get())
            return "ahoy"

        agent = Mock(name="agent", run=run, storage_adapter=None)
        with patch.object(DemoWebhookIntegration, "resolve_agent", return_value=agent):
            await run_inbound(self._payload())
        assert seen[0].event_id == "Ev1"
        assert current_event.get() is None

    async def test_a_failing_run_still_replies(self):
        integration = DemoWebhookIntegration()
        register(integration)
        agent = Mock(
            run=AsyncMock(side_effect=RuntimeError("model exploded")), storage_adapter=None
        )
        with patch.object(DemoWebhookIntegration, "resolve_agent", return_value=agent):
            await run_inbound(self._payload())
        assert integration.replies == [ERROR_MESSAGE]

    async def test_a_timeout_says_so(self):
        integration = DemoWebhookIntegration()
        register(integration)

        async def _forever(*args, **kwargs):
            await asyncio.sleep(5)

        agent = Mock(run=_forever, storage_adapter=None)
        with (
            patch.object(DemoWebhookIntegration, "resolve_agent", return_value=agent),
            override_settings(AI_SDK_WEBHOOK_TIMEOUT=0.01),
        ):
            await run_inbound(self._payload())
        assert integration.replies == [TIMEOUT_MESSAGE]

    async def test_an_uninstalled_integration_is_dropped(self):
        await run_inbound(self._payload())
        assert await cache.aget("ai_sdk:webhook:demo:Ev1") is None

    async def test_an_unresolvable_agent_still_replies(self):
        integration = DemoWebhookIntegration()
        register(integration)
        with patch.object(DemoWebhookIntegration, "resolve_agent", return_value=None):
            await run_inbound(self._payload())
        assert integration.replies == [ERROR_MESSAGE]

    async def test_history_is_loaded_and_the_turn_is_persisted(self):
        from django_ai_sdk.storage.memory import MemoryStorageAdapter, MemoryStore

        MemoryStore.clear()
        integration = DemoWebhookIntegration()
        register(integration)
        agent = Mock(
            name="agent",
            model="test-model",
            max_history=40,
            storage_adapter=MemoryStorageAdapter,
            run=AsyncMock(return_value="still here"),
        )
        agent.__class__._agent_id = "agent-demo"
        with (
            patch.object(DemoWebhookIntegration, "resolve_agent", return_value=agent),
            patch("django_ai_sdk.memories.services.MemoryService.link_memories", AsyncMock()),
        ):
            await run_inbound(self._payload())
            await run_inbound(
                InboundEvent(
                    integration="demo",
                    event_id="Ev2",
                    text="remember?",
                    conversation_id="C1",
                ).model_dump()
            )

        second_call = agent.run.await_args_list[1]
        messages = second_call.args[0]
        assert [m.content for m in messages] == ["hi", "still here", "remember?"]
        thread_id = second_call.kwargs["thread_id"]
        assert thread_id
        stored = await MemoryStorageAdapter(thread_id).get_messages()
        assert [(m.role, m.content) for m in stored] == [
            ("user", "hi"),
            ("assistant", "still here"),
            ("user", "remember?"),
            ("assistant", "still here"),
        ]
        assert agent.run.await_args_list[0].kwargs["thread_id"] == thread_id
        MemoryStore.clear()

    async def test_a_new_thread_gets_the_agents_memories(self):
        from django_ai_sdk.storage.memory import MemoryStorageAdapter, MemoryStore

        MemoryStore.clear()
        register(DemoWebhookIntegration())
        agent = Mock(
            name="agent",
            model="test-model",
            max_history=40,
            storage_adapter=MemoryStorageAdapter,
            run=AsyncMock(return_value="ahoy"),
        )
        agent.__class__._agent_id = "agent-demo"
        link = AsyncMock()
        with (
            patch.object(DemoWebhookIntegration, "resolve_agent", return_value=agent),
            patch("django_ai_sdk.memories.services.MemoryService.link_memories", link),
        ):
            await run_inbound(self._payload())
            await run_inbound(self._payload())

        thread_id = agent.run.await_args.kwargs["thread_id"]
        link.assert_awaited_once_with("agent-demo", thread_id, user=None)
        MemoryStore.clear()

    async def test_a_thread_creation_race_is_lost_gracefully(self):
        from django_ai_sdk.storage.memory import MemoryStorageAdapter, MemoryStore

        class RaceLosingAdapter(MemoryStorageAdapter):
            @classmethod
            async def create_thread(cls, **kwargs):
                await super().create_thread(**kwargs)
                raise RuntimeError("UNIQUE constraint failed: thread id")

        MemoryStore.clear()
        integration = DemoWebhookIntegration()
        register(integration)
        agent = Mock(
            name="agent",
            model="test-model",
            max_history=40,
            storage_adapter=RaceLosingAdapter,
            run=AsyncMock(return_value="still here"),
        )
        agent.__class__._agent_id = "agent-demo"
        with patch.object(DemoWebhookIntegration, "resolve_agent", return_value=agent):
            await run_inbound(self._payload())
        assert integration.replies == ["still here"]
        MemoryStore.clear()


class TestThreadScope:
    def test_telegram_scopes_the_chat(self):
        from django_ai_sdk.integrations.telegram.integration import TelegramIntegration

        event = InboundEvent(
            integration="telegram",
            event_id="1",
            text="hi",
            conversation_id="848151",
            thread_ref="99",
        )
        assert TelegramIntegration().thread_scope(event) == "ai-sdk:webhook:telegram:848151"

    def test_slack_scopes_the_slack_thread(self):
        from django_ai_sdk.integrations.slack.integration import SlackIntegration

        event = InboundEvent(
            integration="slack",
            event_id="1",
            text="hi",
            conversation_id="C024BE91L",
            thread_ref="1512085950.000216",
        )
        assert (
            SlackIntegration().thread_scope(event)
            == "ai-sdk:webhook:slack:C024BE91L:1512085950.000216"
        )

    def test_slack_without_thread_ref_stays_stateless(self):
        from django_ai_sdk.integrations.slack.integration import SlackIntegration

        event = InboundEvent(
            integration="slack",
            event_id="1",
            text="hi",
            conversation_id="C024BE91L",
        )
        assert SlackIntegration().thread_scope(event) is None

    def test_github_scopes_the_issue(self):
        from django_ai_sdk.integrations.github_events.integration import GitHubEventsIntegration

        event = InboundEvent(
            integration="github_events",
            event_id="1",
            text="hi",
            conversation_id="acme/widgets",
            thread_ref="42",
        )
        assert (
            GitHubEventsIntegration().thread_scope(event)
            == "ai-sdk:webhook:github_events:acme/widgets:42"
        )
