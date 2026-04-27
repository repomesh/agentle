from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from types import SimpleNamespace
from typing import Any

from agentle.agents.channels.models.channel_message import ChannelMessage
from agentle.agents.channels.providers.instagram_direct import (
    InstagramDirectConfig,
    InstagramDirectProvider,
)
from agentle.agents.channels.providers.microsoft_teams import (
    MicrosoftTeamsConfig,
    MicrosoftTeamsProvider,
)
from agentle.agents.channels.providers.whatsapp_cloud import (
    WhatsAppCloudConfig,
    WhatsAppCloudProvider,
)
from agentle.generations.models.message_parts.text import TextPart
from agentle.generations.models.messages.generated_assistant_message import (
    GeneratedAssistantMessage,
)


PROVIDER_NAMES = ("whatsapp_cloud", "instagram_direct", "microsoft_teams")


@dataclass(frozen=True)
class AgentCall:
    chat_id: str | None
    text: str
    parts: tuple[Any, ...]


@dataclass
class CallbackRecord:
    contact_identifier: str
    chat_id: str | None
    response: GeneratedAssistantMessage[Any] | None
    context: dict[str, Any]


@dataclass
class TextSendRecord:
    provider: str
    recipient_id: str
    text: str
    quoted_message_id: str | None
    url: str
    payload: Mapping[str, Any]


@dataclass
class HTTPRecord:
    method: str
    url: str
    data: Mapping[str, Any] | None
    expected_status: int | tuple[int, ...]


@dataclass
class HTTPProbe:
    provider_name: str
    records: list[HTTPRecord] = field(default_factory=list)
    text_sends: list[TextSendRecord] = field(default_factory=list)
    fail_text_sends: int = 0
    _counter: int = 0

    async def handle_request(
        self,
        method: str,
        url: str,
        data: Mapping[str, Any] | None = None,
        expected_status: int | tuple[int, ...] = 200,
    ) -> Mapping[str, Any]:
        self.records.append(
            HTTPRecord(
                method=method.upper(),
                url=url,
                data=data,
                expected_status=expected_status,
            )
        )

        text_send = self._extract_text_send(url, data)
        if text_send is not None:
            if self.fail_text_sends > 0:
                self.fail_text_sends -= 1
                raise RuntimeError("offline send failure")
            self.text_sends.append(text_send)

        self._counter += 1
        if self.provider_name == "whatsapp_cloud":
            return {"messages": [{"id": f"wamid.offline.{self._counter}"}]}
        if self.provider_name == "instagram_direct":
            return {"recipient_id": "recipient", "message_id": f"mid.{self._counter}"}
        return {"id": f"activity.{self._counter}"}

    def _extract_text_send(
        self, url: str, data: Mapping[str, Any] | None
    ) -> TextSendRecord | None:
        if not data:
            return None

        if self.provider_name == "whatsapp_cloud" and data.get("type") == "text":
            text_payload = data.get("text") or {}
            context = data.get("context") or {}
            return TextSendRecord(
                provider=self.provider_name,
                recipient_id=str(data.get("to") or ""),
                text=str(text_payload.get("body") or ""),
                quoted_message_id=context.get("message_id"),
                url=url,
                payload=data,
            )

        if self.provider_name == "instagram_direct":
            message = data.get("message") or {}
            if not isinstance(message, Mapping) or "text" not in message:
                return None
            recipient = data.get("recipient") or {}
            return TextSendRecord(
                provider=self.provider_name,
                recipient_id=str(recipient.get("id") or ""),
                text=str(message.get("text") or ""),
                quoted_message_id=None,
                url=url,
                payload=data,
            )

        if self.provider_name == "microsoft_teams":
            if data.get("type") != "message" or "text" not in data:
                return None
            conversation = data.get("conversation") or {}
            return TextSendRecord(
                provider=self.provider_name,
                recipient_id=str(conversation.get("id") or ""),
                text=str(data.get("text") or ""),
                quoted_message_id=data.get("replyToId"),
                url=url,
                payload=data,
            )

        return None


class FakeStressAgent:
    def __init__(
        self,
        *,
        delay_seconds: float = 0.001,
        response_factory: Callable[[AgentCall], str] | None = None,
        fail_for_chat_ids: set[str] | None = None,
    ) -> None:
        self.conversation_store = object()
        self.delay_seconds = delay_seconds
        self.response_factory = response_factory or (
            lambda call: f"reply:{call.chat_id}:{len(call.text)}"
        )
        self.fail_for_chat_ids = fail_for_chat_ids or set()
        self.calls: list[AgentCall] = []
        self.active_calls = 0
        self.max_active_calls = 0
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def start_mcp_servers_async(self):
        yield

    async def run_async(
        self,
        agent_input: Any,
        *,
        chat_id: str | None = None,
        **_: Any,
    ) -> Any:
        async with self._lock:
            self.active_calls += 1
            self.max_active_calls = max(self.max_active_calls, self.active_calls)

        try:
            if self.delay_seconds:
                await asyncio.sleep(self.delay_seconds)

            parts = tuple(getattr(agent_input, "parts", ()))
            text = "".join(
                str(part.text) for part in parts if isinstance(part, TextPart)
            )
            call = AgentCall(chat_id=chat_id, text=text, parts=parts)
            self.calls.append(call)

            if chat_id in self.fail_for_chat_ids:
                raise RuntimeError(f"forced failure for {chat_id}")

            response_text = self.response_factory(call)
            message = GeneratedAssistantMessage(
                parts=[TextPart(text=response_text)],
                parsed=None,
            )
            return SimpleNamespace(
                generation=SimpleNamespace(message=message),
                input_tokens=max(1, len(text) // 4),
                output_tokens=max(1, len(response_text) // 4),
            )
        finally:
            async with self._lock:
                self.active_calls -= 1


def install_offline_http(provider: Any, provider_name: str) -> HTTPProbe:
    probe = HTTPProbe(provider_name=provider_name)

    async def fake_request(
        method: str,
        url: str,
        data: Mapping[str, Any] | None = None,
        expected_status: int | tuple[int, ...] = 200,
    ) -> Mapping[str, Any]:
        return await probe.handle_request(method, url, data, expected_status)

    provider._make_request_with_retry = fake_request
    provider._make_request = fake_request
    return probe


def make_offline_provider(
    provider_name: str,
) -> tuple[Any, HTTPProbe, Callable[[int, int, str | None], ChannelMessage]]:
    if provider_name == "whatsapp_cloud":
        provider = WhatsAppCloudProvider(
            WhatsAppCloudConfig(
                access_token="offline-token",
                phone_number_id="phone-number-1",
                max_retries=0,
                retry_delay=0,
            )
        )
        probe = install_offline_http(provider, provider_name)
        return provider, probe, lambda contact, index, text=None: make_whatsapp_message(
            contact, index, text=text
        )

    if provider_name == "instagram_direct":
        provider = InstagramDirectProvider(
            InstagramDirectConfig(
                access_token="offline-token",
                instagram_user_id="ig-business-1",
                max_retries=0,
                retry_delay=0,
            )
        )
        probe = install_offline_http(provider, provider_name)
        return provider, probe, lambda contact, index, text=None: make_instagram_message(
            contact, index, text=text
        )

    if provider_name == "microsoft_teams":
        provider = MicrosoftTeamsProvider(
            MicrosoftTeamsConfig(
                app_id="teams-app-1",
                access_token="offline-token",
                bot_name="Agentle",
                max_retries=0,
                retry_delay=0,
            )
        )
        probe = install_offline_http(provider, provider_name)
        return provider, probe, lambda contact, index, text=None: make_teams_message(
            provider, contact, index, text=text
        )

    raise ValueError(f"Unknown provider: {provider_name}")


def make_whatsapp_message(
    contact_index: int,
    message_index: int,
    *,
    text: str | None = None,
) -> ChannelMessage:
    contact_id = f"5511999{contact_index:06d}"
    message_text = text or f"whatsapp message {contact_index}:{message_index}"
    return WhatsAppCloudProvider.parse_channel_message(
        value={
            "contacts": [
                {"wa_id": contact_id, "profile": {"name": f"WhatsApp {contact_index}"}}
            ],
            "metadata": {"phone_number_id": "phone-number-1"},
        },
        message_data={
            "id": f"wa-{contact_index}-{message_index}",
            "from": contact_id,
            "timestamp": str(1_777_152_000 + message_index),
            "type": "text",
            "text": {"body": message_text},
        },
        phone_number_id="phone-number-1",
    )


def make_instagram_message(
    contact_index: int,
    message_index: int,
    *,
    text: str | None = None,
) -> ChannelMessage:
    contact_id = f"igsid-user-{contact_index}"
    message_text = text or f"instagram message {contact_index}:{message_index}"
    return InstagramDirectProvider.messaging_event_to_channel_message(
        {
            "sender": {"id": contact_id, "username": f"ig-user-{contact_index}"},
            "recipient": {"id": "ig-business-1"},
            "timestamp": 1_777_152_000_000 + message_index,
            "message": {
                "mid": f"ig-{contact_index}-{message_index}",
                "text": message_text,
            },
        },
        resource_id="ig-business-1",
    )


def make_teams_message(
    provider: MicrosoftTeamsProvider,
    contact_index: int,
    message_index: int,
    *,
    text: str | None = None,
) -> ChannelMessage:
    conversation_id = f"teams-conversation-{contact_index}"
    user_id = f"teams-user-{contact_index}"
    message_text = text or f"teams message {contact_index}:{message_index}"
    return provider.parse_channel_message(
        {
            "type": "message",
            "id": f"teams-{contact_index}-{message_index}",
            "timestamp": "2026-04-25T21:35:00.000Z",
            "serviceUrl": "https://smba.trafficmanager.net/teams/",
            "channelId": "msteams",
            "from": {"id": user_id, "name": f"Teams User {contact_index}"},
            "recipient": {"id": "teams-app-1", "name": "Agentle"},
            "conversation": {"id": conversation_id, "conversationType": "personal"},
            "text": f"<at>Agentle</at> {message_text}",
            "entities": [
                {
                    "type": "mention",
                    "text": "<at>Agentle</at>",
                    "mentioned": {"id": "teams-app-1", "name": "Agentle"},
                }
            ],
            "channelData": {"tenant": {"id": "tenant-1"}},
        }
    )


def collect_callbacks(records: list[CallbackRecord]):
    async def callback(
        contact_identifier: str,
        chat_id: str | None,
        response: GeneratedAssistantMessage[Any] | None,
        context: dict[str, Any],
    ) -> None:
        records.append(
            CallbackRecord(
                contact_identifier=contact_identifier,
                chat_id=chat_id,
                response=response,
                context=dict(context),
            )
        )

    return callback


async def wait_for(
    predicate: Callable[[], bool],
    *,
    timeout_seconds: float = 5.0,
    interval_seconds: float = 0.02,
) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(interval_seconds)
    if not predicate():
        raise AssertionError("Timed out waiting for stress test condition")


async def assert_no_exceptions(results: list[Any]) -> None:
    exceptions = [result for result in results if isinstance(result, Exception)]
    if exceptions:
        raise AssertionError(f"Unexpected exceptions: {exceptions!r}")
