from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any

import pytest
from aiohttp import web

from agentle.agents.channels.channel_bot import ChannelBot
from agentle.agents.channels.channel_bot_config import ChannelBotConfig
from agentle.agents.channels.models.channel_session import ChannelSession
from agentle.agents.channels.providers.instagram_direct import (
    InstagramDirectConfig,
    InstagramDirectProvider,
)
from agentle.agents.channels.providers.instagram_direct.instagram_direct_provider import (
    InstagramDirectError,
)
from agentle.agents.channels.providers.microsoft_teams import (
    MicrosoftTeamsConfig,
    MicrosoftTeamsProvider,
)
from agentle.agents.channels.providers.microsoft_teams.microsoft_teams_provider import (
    MicrosoftTeamsError,
)
from agentle.agents.channels.providers.whatsapp_cloud import (
    WhatsAppCloudConfig,
    WhatsAppCloudProvider,
)
from agentle.agents.channels.providers.whatsapp_cloud.whatsapp_cloud_provider import (
    WhatsAppCloudError,
)
from agentle.sessions.in_memory_session_store import InMemorySessionStore
from agentle.sessions.session_manager import SessionManager

from .conftest import (
    PROVIDER_NAMES,
    CallbackRecord,
    FakeStressAgent,
    collect_callbacks,
    install_offline_http,
    make_instagram_message,
    make_teams_message,
    make_whatsapp_message,
)


pytestmark = [
    pytest.mark.stress,
    pytest.mark.skipif(
        os.environ.get("AGENTLE_RUN_CHANNEL_STRESS") != "1",
        reason="Set AGENTLE_RUN_CHANNEL_STRESS=1 to run channel stress tests.",
    ),
]


@dataclass
class CapturedRequest:
    method: str
    path: str
    headers: dict[str, str]
    json_body: dict[str, Any]


@dataclass
class LocalCaptureServer:
    base_url: str
    requests: list[CapturedRequest] = field(default_factory=list)
    runner: web.AppRunner | None = None

    async def close(self) -> None:
        if self.runner is not None:
            await self.runner.cleanup()
            self.runner = None


async def _start_capture_server() -> LocalCaptureServer:
    captured: list[CapturedRequest] = []

    async def handler(request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except Exception:
            body = {}

        captured.append(
            CapturedRequest(
                method=request.method,
                path=request.path,
                headers={key.lower(): value for key, value in request.headers.items()},
                json_body=body,
            )
        )

        if body.get("messaging_product") == "whatsapp":
            return web.json_response({"messages": [{"id": "wamid.local"}]})
        if "recipient" in body and "message" in body:
            return web.json_response({"message_id": "mid.local"})
        return web.json_response({"id": "activity.local"}, status=201)

    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    sockets = site._server.sockets if site._server is not None else None
    assert sockets
    port = sockets[0].getsockname()[1]
    return LocalCaptureServer(
        base_url=f"http://127.0.0.1:{port}",
        requests=captured,
        runner=runner,
    )


def _make_shared_provider(
    provider_name: str,
    session_manager: SessionManager[ChannelSession],
) -> tuple[Any, Any]:
    if provider_name == "whatsapp_cloud":
        provider = WhatsAppCloudProvider(
            WhatsAppCloudConfig(
                access_token="offline-token",
                phone_number_id="phone-number-1",
                max_retries=0,
                retry_delay=0,
            ),
            session_manager=session_manager,
        )
        probe = install_offline_http(provider, provider_name)
        return provider, probe

    if provider_name == "instagram_direct":
        provider = InstagramDirectProvider(
            InstagramDirectConfig(
                access_token="offline-token",
                instagram_user_id="ig-business-1",
                max_retries=0,
                retry_delay=0,
            ),
            session_manager=session_manager,
        )
        probe = install_offline_http(provider, provider_name)
        return provider, probe

    if provider_name == "microsoft_teams":
        provider = MicrosoftTeamsProvider(
            MicrosoftTeamsConfig(
                app_id="teams-app-1",
                access_token="offline-token",
                bot_name="Agentle",
                max_retries=0,
                retry_delay=0,
            ),
            session_manager=session_manager,
        )
        probe = install_offline_http(provider, provider_name)
        return provider, probe

    raise ValueError(f"Unknown provider: {provider_name}")


def _make_message_for_provider(provider_name: str, provider: Any, contact: int, index: int):
    if provider_name == "whatsapp_cloud":
        return make_whatsapp_message(contact, index, text=f"shared {contact}:{index}")
    if provider_name == "instagram_direct":
        return make_instagram_message(contact, index, text=f"shared {contact}:{index}")
    return make_teams_message(provider, contact, index, text=f"shared {contact}:{index}")


@pytest.mark.parametrize("provider_name", PROVIDER_NAMES)
async def test_multiple_bot_instances_share_session_manager_under_load(
    provider_name: str,
) -> None:
    shared_manager = SessionManager(
        session_store=InMemorySessionStore[ChannelSession](),
        default_ttl_seconds=60,
    )
    agent = FakeStressAgent(delay_seconds=0.001)
    callbacks: list[CallbackRecord] = []
    bots: list[ChannelBot] = []
    providers: list[Any] = []
    probes: list[Any] = []
    bot_count = 3
    contact_count = 12
    messages_per_contact = 10

    try:
        for _ in range(bot_count):
            provider, probe = _make_shared_provider(provider_name, shared_manager)
            providers.append(provider)
            probes.append(probe)
            bot = ChannelBot(
                agent=agent,
                provider=provider,
                config=ChannelBotConfig(
                    enable_message_batching=False,
                    spam_protection_enabled=False,
                    auto_read_messages=False,
                    typing_indicator=False,
                ),
            )
            bot.add_response_callback(collect_callbacks(callbacks))
            bots.append(bot)

        await asyncio.gather(*(bot.start_async() for bot in bots))

        async def process(contact_index: int, message_index: int) -> Any:
            bot_index = message_index % bot_count
            message = _make_message_for_provider(
                provider_name,
                providers[bot_index],
                contact_index,
                message_index,
            )
            return await bots[bot_index].handle_channel_message(message)

        results = await asyncio.gather(
            *(
                process(contact_index, message_index)
                for contact_index in range(contact_count)
                for message_index in range(messages_per_contact)
            ),
            return_exceptions=True,
        )

        exceptions = [result for result in results if isinstance(result, Exception)]
        assert exceptions == []
        assert all(result is not None for result in results)
        assert len(agent.calls) == contact_count * messages_per_contact
        assert len(callbacks) == contact_count * messages_per_contact
        assert sum(len(probe.text_sends) for probe in probes) == (
            contact_count * messages_per_contact
        )
        assert await shared_manager.get_session_count() == contact_count

        for contact_index in range(contact_count):
            contact_identifier = _make_message_for_provider(
                provider_name,
                providers[0],
                contact_index,
                0,
            ).contact_identifier
            session = await providers[0].get_session(contact_identifier)
            assert session is not None
            assert session.message_count == messages_per_contact
    finally:
        await asyncio.gather(*(bot.stop_async() for bot in bots), return_exceptions=True)
        await shared_manager.close()


@pytest.mark.parametrize(
    ("provider_name", "error_type"),
    (
        ("whatsapp_cloud", WhatsAppCloudError),
        ("instagram_direct", InstagramDirectError),
        ("microsoft_teams", MicrosoftTeamsError),
    ),
)
async def test_provider_retry_backoff_retries_500_and_429_before_success(
    provider_name: str,
    error_type: type[Exception],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, _, _ = {
        "whatsapp_cloud": lambda: (
            WhatsAppCloudProvider(
                WhatsAppCloudConfig(
                    access_token="offline-token",
                    phone_number_id="phone-number-1",
                    max_retries=2,
                    retry_delay=0.25,
                )
            ),
            None,
            None,
        ),
        "instagram_direct": lambda: (
            InstagramDirectProvider(
                InstagramDirectConfig(
                    access_token="offline-token",
                    instagram_user_id="ig-business-1",
                    max_retries=2,
                    retry_delay=0.25,
                )
            ),
            None,
            None,
        ),
        "microsoft_teams": lambda: (
            MicrosoftTeamsProvider(
                MicrosoftTeamsConfig(
                    app_id="teams-app-1",
                    access_token="offline-token",
                    max_retries=2,
                    retry_delay=0.25,
                )
            ),
            None,
            None,
        ),
    }[provider_name]()

    statuses = [500, 429]
    attempts: list[int] = []
    sleeps: list[float] = []

    async def fake_request(*_: Any, **__: Any) -> dict[str, Any]:
        attempts.append(len(attempts) + 1)
        if statuses:
            status = statuses.pop(0)
            raise error_type("transient", status, {"error": {"message": "transient"}})
        return {"ok": True}

    async def fake_sleep(duration: float) -> None:
        sleeps.append(duration)

    monkeypatch.setattr(provider, "_make_request", fake_request)
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    result = await provider._make_request_with_retry("POST", "https://local.test")

    assert result == {"ok": True}
    assert attempts == [1, 2, 3]
    assert sleeps == [0.25, 0.5]

    await provider.shutdown()


@pytest.mark.parametrize(
    ("provider_name", "error_type", "terminal_status"),
    (
        ("whatsapp_cloud", WhatsAppCloudError, 401),
        ("instagram_direct", InstagramDirectError, 404),
        ("microsoft_teams", MicrosoftTeamsError, 403),
    ),
)
async def test_provider_retry_backoff_does_not_retry_terminal_client_errors(
    provider_name: str,
    error_type: type[Exception],
    terminal_status: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, _, _ = {
        "whatsapp_cloud": lambda: (
            WhatsAppCloudProvider(
                WhatsAppCloudConfig(
                    access_token="offline-token",
                    phone_number_id="phone-number-1",
                    max_retries=3,
                    retry_delay=0.25,
                )
            ),
            None,
            None,
        ),
        "instagram_direct": lambda: (
            InstagramDirectProvider(
                InstagramDirectConfig(
                    access_token="offline-token",
                    instagram_user_id="ig-business-1",
                    max_retries=3,
                    retry_delay=0.25,
                )
            ),
            None,
            None,
        ),
        "microsoft_teams": lambda: (
            MicrosoftTeamsProvider(
                MicrosoftTeamsConfig(
                    app_id="teams-app-1",
                    access_token="offline-token",
                    max_retries=3,
                    retry_delay=0.25,
                )
            ),
            None,
            None,
        ),
    }[provider_name]()

    attempts = 0
    sleeps: list[float] = []

    async def fake_request(*_: Any, **__: Any) -> dict[str, Any]:
        nonlocal attempts
        attempts += 1
        raise error_type("terminal", terminal_status, {"error": {"message": "terminal"}})

    async def fake_sleep(duration: float) -> None:
        sleeps.append(duration)

    monkeypatch.setattr(provider, "_make_request", fake_request)
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    with pytest.raises(error_type):
        await provider._make_request_with_retry("POST", "https://local.test")

    assert attempts == 1
    assert sleeps == []

    await provider.shutdown()


@pytest.mark.parametrize("provider_name", PROVIDER_NAMES)
async def test_provider_http_layer_uses_expected_methods_headers_and_payloads(
    provider_name: str,
) -> None:
    server = await _start_capture_server()

    if provider_name == "whatsapp_cloud":
        provider = WhatsAppCloudProvider(
            WhatsAppCloudConfig(
                access_token="offline-token",
                phone_number_id="phone-number-1",
                base_url=server.base_url,
                max_retries=0,
            )
        )
        message = make_whatsapp_message(1, 0, text="http contract")
    elif provider_name == "instagram_direct":
        provider = InstagramDirectProvider(
            InstagramDirectConfig(
                access_token="offline-token",
                instagram_user_id="ig-business-1",
                base_url=server.base_url,
                max_retries=0,
            )
        )
        message = make_instagram_message(1, 0, text="http contract")
    else:
        provider = MicrosoftTeamsProvider(
            MicrosoftTeamsConfig(
                app_id="teams-app-1",
                access_token="offline-token",
                bot_name="Agentle",
                max_retries=0,
            )
        )
        message = provider.parse_channel_message(
            {
                "type": "message",
                "id": "teams-http-0",
                "timestamp": "2026-04-25T21:35:00.000Z",
                "serviceUrl": f"{server.base_url}/",
                "channelId": "msteams",
                "from": {"id": "teams-user-1", "name": "Teams User 1"},
                "recipient": {"id": "teams-app-1", "name": "Agentle"},
                "conversation": {
                    "id": "teams-conversation-1",
                    "conversationType": "personal",
                },
                "text": "<at>Agentle</at> http contract",
                "entities": [
                    {
                        "type": "mention",
                        "text": "<at>Agentle</at>",
                        "mentioned": {"id": "teams-app-1", "name": "Agentle"},
                    }
                ],
            }
        )

    try:
        await provider.initialize()
        result = await provider.send_text_message(
            message.contact_identifier,
            "local hello",
            quoted_message_id=message.id,
        )

        assert result.provider == provider_name
        assert len(server.requests) == 1
        request = server.requests[0]
        assert request.method == "POST"
        assert request.headers["authorization"] == "Bearer offline-token"
        assert request.headers["content-type"].startswith("application/json")

        if provider_name == "whatsapp_cloud":
            assert request.path == "/v24.0/phone-number-1/messages"
            assert request.json_body["messaging_product"] == "whatsapp"
            assert request.json_body["type"] == "text"
            assert request.json_body["text"]["body"] == "local hello"
            assert request.json_body["context"]["message_id"] == message.id
        elif provider_name == "instagram_direct":
            assert request.path == "/v24.0/ig-business-1/messages"
            assert request.json_body == {
                "recipient": {"id": message.contact_identifier},
                "message": {"text": "local hello"},
            }
        else:
            assert request.path == (
                "/v3/conversations/teams-conversation-1/activities/teams-http-0"
            )
            assert request.json_body["type"] == "message"
            assert request.json_body["text"] == "local hello"
            assert request.json_body["replyToId"] == message.id
            assert request.json_body["conversation"]["id"] == "teams-conversation-1"
    finally:
        await provider.shutdown()
        await server.close()


@pytest.mark.parametrize("provider_name", PROVIDER_NAMES)
async def test_session_ttl_expiration_under_load_starts_fresh_sessions(
    provider_name: str,
) -> None:
    provider, probe, make_message = {
        "whatsapp_cloud": lambda: (
            WhatsAppCloudProvider(
                WhatsAppCloudConfig(
                    access_token="offline-token",
                    phone_number_id="phone-number-1",
                    max_retries=0,
                    retry_delay=0,
                ),
                session_ttl_seconds=1,
            ),
            None,
            lambda contact, index, text=None: make_whatsapp_message(
                contact, index, text=text
            ),
        ),
        "instagram_direct": lambda: (
            InstagramDirectProvider(
                InstagramDirectConfig(
                    access_token="offline-token",
                    instagram_user_id="ig-business-1",
                    max_retries=0,
                    retry_delay=0,
                ),
                session_ttl_seconds=1,
            ),
            None,
            lambda contact, index, text=None: make_instagram_message(
                contact, index, text=text
            ),
        ),
        "microsoft_teams": lambda: (
            MicrosoftTeamsProvider(
                MicrosoftTeamsConfig(
                    app_id="teams-app-1",
                    access_token="offline-token",
                    bot_name="Agentle",
                    max_retries=0,
                    retry_delay=0,
                ),
                session_ttl_seconds=1,
            ),
            None,
            None,
        ),
    }[provider_name]()

    if provider_name == "microsoft_teams":
        make_message = lambda contact, index, text=None: make_teams_message(
            provider, contact, index, text=text
        )
    probe = install_offline_http(provider, provider_name)
    agent = FakeStressAgent(delay_seconds=0.001)
    bot = ChannelBot(
        agent=agent,
        provider=provider,
        config=ChannelBotConfig(
            enable_message_batching=False,
            spam_protection_enabled=False,
            auto_read_messages=False,
            typing_indicator=False,
        ),
    )
    contact_count = 24

    await bot.start_async()
    try:
        first_results = await asyncio.gather(
            *(
                bot.handle_channel_message(
                    make_message(contact_index, 0, f"ttl first {contact_index}")
                )
                for contact_index in range(contact_count)
            )
        )
        assert all(result is not None for result in first_results)
        assert await provider.session_manager.get_session_count() == contact_count

        await asyncio.sleep(1.1)
        cleaned = await provider.session_manager.cleanup_expired_sessions()
        assert cleaned == contact_count
        assert await provider.session_manager.get_session_count() == 0

        second_results = await asyncio.gather(
            *(
                bot.handle_channel_message(
                    make_message(contact_index, 1, f"ttl second {contact_index}")
                )
                for contact_index in range(contact_count)
            )
        )
        assert all(result is not None for result in second_results)
        assert await provider.session_manager.get_session_count() == contact_count
        assert len(probe.text_sends) == contact_count * 2

        for contact_index in range(contact_count):
            contact_identifier = make_message(contact_index, 1, None).contact_identifier
            session = await provider.get_session(contact_identifier)
            assert session is not None
            assert session.message_count == 1
    finally:
        await bot.stop_async()
