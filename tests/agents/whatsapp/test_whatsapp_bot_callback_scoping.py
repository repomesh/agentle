from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from agentle.agents.agent import Agent
from agentle.agents.conversations.conversation_store import ConversationStore
from agentle.agents.whatsapp.models.downloaded_media import DownloadedMedia
from agentle.agents.whatsapp.models.whatsapp_contact import WhatsAppContact
from agentle.agents.whatsapp.models.whatsapp_bot_config import WhatsAppBotConfig
from agentle.agents.whatsapp.models.whatsapp_session import WhatsAppSession
from agentle.agents.whatsapp.models.whatsapp_text_message import WhatsAppTextMessage
from agentle.agents.whatsapp.models.whatsapp_webhook_payload import (
    WhatsAppWebhookPayload,
)
from agentle.agents.whatsapp.providers.base.whatsapp_provider import WhatsAppProvider
from agentle.agents.whatsapp.whatsapp_bot import (
    QueuedMessageResult,
    WhatsAppBot,
)
from agentle.generations.models.message_parts.text import TextPart
from agentle.generations.models.messages.assistant_message import AssistantMessage
from agentle.generations.models.messages.developer_message import DeveloperMessage
from agentle.generations.models.messages.generated_assistant_message import (
    GeneratedAssistantMessage,
)
from agentle.generations.models.messages.user_message import UserMessage
from agentle.generations.providers.base.generation_provider import GenerationProvider
from agentle.generations.providers.types.model_kind import ModelKind


class InMemoryConversationStore(ConversationStore):
    def __init__(self) -> None:
        super().__init__()
        self.messages: dict[
            str, list[DeveloperMessage | UserMessage | AssistantMessage]
        ] = {}

    async def get_conversation_history_async(
        self, chat_id: str
    ) -> list[DeveloperMessage | UserMessage | AssistantMessage]:
        return list(self.messages.get(chat_id, []))

    async def add_message_async[T = Any](
        self,
        chat_id: str,
        message: DeveloperMessage
        | UserMessage
        | AssistantMessage
        | GeneratedAssistantMessage[T],
    ) -> None:
        stored_message = (
            message.to_assistant_message()
            if isinstance(message, GeneratedAssistantMessage)
            else message
        )
        self.messages.setdefault(chat_id, []).append(stored_message)

    async def clear_conversation_async(self, chat_id: str) -> None:
        self.messages.pop(chat_id, None)


class NullGenerationProvider(GenerationProvider):
    @property
    def default_model(self) -> str:
        return "test-model"

    @property
    def organization(self) -> str:
        return "tests"

    async def generate_async[T = None](
        self,
        *,
        model: str | ModelKind | None = None,
        messages=None,
        response_schema: type[T] | None = None,
        generation_config=None,
        tools=None,
        fallback_models=None,
    ):
        raise AssertionError("Generation should not be called in this WhatsApp test")

    async def price_per_million_tokens_input(
        self, model: str, estimate_tokens: int | None = None
    ) -> float:
        return 0.0

    async def price_per_million_tokens_output(
        self, model: str, estimate_tokens: int | None = None
    ) -> float:
        return 0.0

    def map_model_kind_to_provider_model(self, model_kind: ModelKind) -> str:
        return str(model_kind)


class DummyWhatsAppProvider(WhatsAppProvider):
    def get_instance_identifier(self) -> str:
        return "tests"

    async def initialize(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None

    async def send_text_message(
        self, to: str, text: str, quoted_message_id: str | None = None
    ) -> WhatsAppTextMessage:
        return WhatsAppTextMessage(id="sent", from_number="bot", to_number=to, text=text)

    async def send_media_message(
        self,
        to: str,
        media_url: str,
        media_type: str,
        caption: str | None = None,
        filename: str | None = None,
        quoted_message_id: str | None = None,
    ):
        raise NotImplementedError

    async def send_audio_message(
        self,
        to: str,
        audio_base64: str,
        quoted_message_id: str | None = None,
    ):
        raise NotImplementedError

    async def send_audio_message_by_url(
        self,
        to: str,
        audio_url: str,
        quoted_message_id: str | None = None,
    ):
        raise NotImplementedError

    async def send_typing_indicator(self, to: str, duration: int = 3) -> None:
        return None

    async def send_recording_indicator(self, to: str, duration: int = 3) -> None:
        return None

    async def mark_message_as_read(self, message_id: str) -> None:
        return None

    async def get_contact_info(self, phone: str) -> WhatsAppContact | None:
        return None

    async def get_session(self, phone: str) -> WhatsAppSession | None:
        return None

    async def update_session(self, session: WhatsAppSession) -> None:
        return None

    async def validate_webhook(self, payload: WhatsAppWebhookPayload) -> None:
        return None

    async def download_media(self, media_id: str) -> DownloadedMedia:
        raise NotImplementedError

    def get_webhook_url(self) -> str:
        return "https://example.com/webhook"

    async def set_webhook_url(self, url: str) -> None:
        return None


def make_bot() -> WhatsAppBot:
    agent = Agent(
        generation_provider=NullGenerationProvider(),
        model="test-model",
        instructions="Be helpful.",
        conversation_store=InMemoryConversationStore(),
    )

    return WhatsAppBot(
        agent=agent,
        provider=DummyWhatsAppProvider(),
        config=WhatsAppBotConfig(
            enable_human_delays=False,
            spam_protection_enabled=False,
            enable_message_batching=True,
            auto_read_messages=False,
        ),
    )


@pytest.mark.asyncio
async def test_handle_webhook_returns_queued_result_and_keeps_scoped_callback_until_completion() -> None:
    bot = make_bot()
    payload = WhatsAppWebhookPayload(
        event="messages.upsert",
        phone_number_id="5511999999999",
    )

    callback_calls: list[dict[str, object]] = []

    async def scoped_callback(phone_number, chat_id, response, context):
        callback_calls.append(
            {
                "phone_number": phone_number,
                "chat_id": chat_id,
                "response": response,
                "context": context,
            }
        )

    queued_result = QueuedMessageResult(
        phone_number="5511999999999",
        chat_id="chat-1",
        pending_messages=2,
        processing_token="batch-token",
    )
    with patch.object(
        bot,
        "_handle_message_upsert",
        AsyncMock(return_value=queued_result),
    ):
        response = await bot.handle_webhook(
            payload,
            callback=scoped_callback,
            callback_context={"source": "webhook"},
            chat_id="chat-1",
        )

    assert response == queued_result
    assert callback_calls == []
    assert len(bot._response_callbacks) == 1
    assert bot._response_callbacks[0].persistent is False

    await bot._call_response_callbacks(
        phone_number="5511999999999",
        chat_id="chat-1",
        response=None,
        input_tokens=0,
        output_tokens=0,
        processing_status="failed",
    )

    assert len(callback_calls) == 1
    assert callback_calls[0]["context"]["source"] == "webhook"
    assert callback_calls[0]["context"]["processing_status"] == "failed"
    assert len(bot._response_callbacks) == 0


@pytest.mark.asyncio
async def test_scoped_callbacks_only_fire_for_matching_conversation() -> None:
    bot = make_bot()

    callback_hits: list[tuple[str, str]] = []

    async def global_callback(phone_number, chat_id, response, context):
        callback_hits.append(("global", context["processing_status"]))

    async def chat_one_callback(phone_number, chat_id, response, context):
        callback_hits.append(("chat-1", context["processing_status"]))

    async def chat_two_callback(phone_number, chat_id, response, context):
        callback_hits.append(("chat-2", context["processing_status"]))

    bot.add_response_callback(global_callback, context={"kind": "global"})
    bot.add_response_callback(
        chat_one_callback,
        context={"kind": "chat-1"},
        chat_id="chat-1",
        persistent=False,
    )
    bot.add_response_callback(
        chat_two_callback,
        context={"kind": "chat-2"},
        chat_id="chat-2",
        persistent=False,
    )

    response = GeneratedAssistantMessage(parts=[TextPart(text="Done")], parsed=None)

    await bot._call_response_callbacks(
        phone_number="551100000001",
        chat_id="chat-1",
        response=response,
        input_tokens=3,
        output_tokens=5,
        processing_status="completed",
    )

    assert callback_hits == [("global", "completed"), ("chat-1", "completed")]
    assert len(bot._response_callbacks) == 2

    await bot._call_response_callbacks(
        phone_number="551100000002",
        chat_id="chat-2",
        response=None,
        input_tokens=0,
        output_tokens=0,
        processing_status="failed",
    )

    assert callback_hits == [
        ("global", "completed"),
        ("chat-1", "completed"),
        ("global", "failed"),
        ("chat-2", "failed"),
    ]
    assert len(bot._response_callbacks) == 1
