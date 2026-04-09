from __future__ import annotations

import base64
import logging
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock

import pytest

from agentle.agents.agent import Agent
from agentle.agents.conversations.conversation_store import ConversationStore
from agentle.agents.whatsapp.models.audio_message import AudioMessage
from agentle.agents.whatsapp.models.data import Data
from agentle.agents.whatsapp.models.document_message import DocumentMessage
from agentle.agents.whatsapp.models.downloaded_media import DownloadedMedia
from agentle.agents.whatsapp.models.image_message import ImageMessage
from agentle.agents.whatsapp.models.key import Key
from agentle.agents.whatsapp.models.message import Message
from agentle.agents.whatsapp.models.video_message import VideoMessage
from agentle.agents.whatsapp.models.whatsapp_audio_message import WhatsAppAudioMessage
from agentle.agents.whatsapp.models.whatsapp_bot_config import WhatsAppBotConfig
from agentle.agents.whatsapp.models.whatsapp_contact import WhatsAppContact
from agentle.agents.whatsapp.models.whatsapp_document_message import (
    WhatsAppDocumentMessage,
)
from agentle.agents.whatsapp.models.whatsapp_image_message import WhatsAppImageMessage
from agentle.agents.whatsapp.models.whatsapp_message import WhatsAppMessage
from agentle.agents.whatsapp.models.whatsapp_session import WhatsAppSession
from agentle.agents.whatsapp.models.whatsapp_video_message import WhatsAppVideoMessage
from agentle.agents.whatsapp.providers.base.whatsapp_provider import WhatsAppProvider
from agentle.agents.whatsapp.providers.evolution.evolution_api_config import (
    EvolutionAPIConfig,
)
from agentle.agents.whatsapp.providers.evolution.evolution_api_provider import (
    EvolutionAPIError,
    EvolutionAPIProvider,
)
from agentle.agents.whatsapp.whatsapp_bot import WhatsAppBot
from agentle.generations.models.message_parts.file import FilePart
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
    ):
        raise NotImplementedError

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

    async def validate_webhook(self, payload) -> None:
        return None

    async def download_media(self, media_id: str) -> DownloadedMedia:
        raise NotImplementedError

    def get_webhook_url(self) -> str:
        return "https://example.com/webhook"

    async def set_webhook_url(self, url: str) -> None:
        return None


def make_bot(provider: DummyWhatsAppProvider | None = None) -> WhatsAppBot:
    agent = Agent(
        generation_provider=NullGenerationProvider(),
        model="test-model",
        instructions="Be helpful.",
        conversation_store=InMemoryConversationStore(),
    )

    return WhatsAppBot(
        agent=agent,
        provider=provider or DummyWhatsAppProvider(),
        config=WhatsAppBotConfig(
            enable_human_delays=False,
            spam_protection_enabled=False,
            enable_message_batching=True,
            auto_read_messages=False,
        ),
    )


def make_session() -> WhatsAppSession:
    return WhatsAppSession(
        session_id="test-session",
        phone_number="5511999999999",
        contact=WhatsAppContact(phone="5511999999999", name="Test User"),
        last_activity=datetime.now(),
        message_count=0,
        is_processing=False,
        pending_messages=[],
        context_data={},
    )


def make_evolution_data(message_type: str, message: Message) -> Data:
    return Data(
        key=Key(
            remoteJid="5511999999999@s.whatsapp.net",
            fromMe=False,
            id=f"{message_type}-id",
        ),
        pushName="Karla Marina",
        messageType=message_type,
        messageTimestamp=1_710_000_000_000,
        message=message,
    )


INLINE_MEDIA_BYTES = b"inline-media-bytes"
INLINE_MEDIA_BASE64 = base64.b64encode(INLINE_MEDIA_BYTES).decode("ascii")


@pytest.mark.parametrize(
    ("message_type", "message", "expected_class", "expected_mime_type", "expected_filename"),
    [
        (
            "imageMessage",
            Message(
                imageMessage=ImageMessage(
                    url="",
                    mimetype="image/png",
                    caption="radiografia",
                ),
                base64=INLINE_MEDIA_BASE64,
            ),
            WhatsAppImageMessage,
            "image/png",
            None,
        ),
        (
            "documentMessage",
            Message(
                documentMessage=DocumentMessage(
                    url="",
                    mimetype="application/pdf",
                    fileName="laudo.pdf",
                    caption="anexo",
                ),
                base64=INLINE_MEDIA_BASE64,
            ),
            WhatsAppDocumentMessage,
            "application/pdf",
            "laudo.pdf",
        ),
        (
            "audioMessage",
            Message(
                audioMessage=AudioMessage(
                    url="",
                    mimetype="audio/ogg; codecs=opus",
                ),
                base64=INLINE_MEDIA_BASE64,
            ),
            WhatsAppAudioMessage,
            "audio/ogg; codecs=opus",
            None,
        ),
        (
            "videoMessage",
            Message(
                videoMessage=VideoMessage(
                    url="",
                    mimetype="video/mp4",
                    caption="video do exame",
                ),
                base64=INLINE_MEDIA_BASE64,
            ),
            WhatsAppVideoMessage,
            "video/mp4",
            None,
        ),
    ],
)
def test_parse_evolution_media_messages_preserve_inline_base64(
    message_type: str,
    message: Message,
    expected_class: type[WhatsAppMessage],
    expected_mime_type: str,
    expected_filename: str | None,
) -> None:
    bot = make_bot()

    parsed_message = bot._parse_evolution_message_from_data(
        make_evolution_data(message_type, message),
        "5511999999999",
    )

    assert isinstance(parsed_message, expected_class)
    assert parsed_message is not None
    assert parsed_message.base64_data == INLINE_MEDIA_BASE64
    assert parsed_message.media_mime_type == expected_mime_type
    if expected_filename is not None:
        assert getattr(parsed_message, "filename", None) == expected_filename


@pytest.mark.asyncio
async def test_convert_message_to_input_prefers_inline_base64_over_download() -> None:
    provider = DummyWhatsAppProvider()
    provider.download_media = AsyncMock(
        return_value=DownloadedMedia(data=b"downloaded", mime_type="image/png")
    )
    bot = make_bot(provider)

    message = WhatsAppImageMessage(
        id="image-inline",
        from_number="5511999999999",
        to_number="tests",
        push_name="Karla Marina",
        timestamp=datetime.now(),
        media_url="",
        media_mime_type="image/png",
        caption="segue a imagem",
        base64_data=INLINE_MEDIA_BASE64,
    )

    converted = await bot._convert_message_to_input(message, make_session())

    provider.download_media.assert_not_awaited()
    assert isinstance(converted, UserMessage)

    file_parts = [part for part in converted.parts if isinstance(part, FilePart)]
    assert len(file_parts) == 1
    assert file_parts[0].data == INLINE_MEDIA_BYTES
    assert file_parts[0].mime_type == "image/png"

    text_parts = [part for part in converted.parts if isinstance(part, TextPart)]
    assert any(str(part.text) == "Caption: segue a imagem" for part in text_parts)


@pytest.mark.asyncio
async def test_convert_message_batch_to_input_prefers_inline_base64_over_download() -> None:
    provider = DummyWhatsAppProvider()
    provider.download_media = AsyncMock(
        return_value=DownloadedMedia(data=b"downloaded", mime_type="image/jpeg")
    )
    bot = make_bot(provider)

    message = WhatsAppImageMessage(
        id="image-inline",
        from_number="5511999999999",
        to_number="tests",
        push_name="Karla Marina",
        timestamp=datetime.now(),
        media_url="",
        media_mime_type="image/png",
        caption="foto",
        base64_data=INLINE_MEDIA_BASE64,
    )
    message_batch = [await bot._message_to_dict(message)]

    converted = await bot._convert_message_batch_to_input(message_batch, make_session())

    provider.download_media.assert_not_awaited()
    assert isinstance(converted, UserMessage)

    file_parts = [part for part in converted.parts if isinstance(part, FilePart)]
    assert len(file_parts) == 1
    assert file_parts[0].data == INLINE_MEDIA_BYTES
    assert file_parts[0].mime_type == "image/png"


@pytest.mark.asyncio
async def test_convert_message_to_input_downloads_media_when_inline_base64_missing() -> None:
    provider = DummyWhatsAppProvider()
    provider.download_media = AsyncMock(
        return_value=DownloadedMedia(
            data=b"downloaded-by-provider",
            mime_type="image/png",
        )
    )
    bot = make_bot(provider)

    message = WhatsAppImageMessage(
        id="image-download",
        from_number="5511999999999",
        to_number="tests",
        push_name="Karla Marina",
        timestamp=datetime.now(),
        media_url="https://example.com/imagem.png",
        media_mime_type="image/png",
        caption=None,
    )

    converted = await bot._convert_message_to_input(message, make_session())

    provider.download_media.assert_awaited_once_with("image-download")
    assert isinstance(converted, UserMessage)
    file_parts = [part for part in converted.parts if isinstance(part, FilePart)]
    assert len(file_parts) == 1
    assert file_parts[0].data == b"downloaded-by-provider"
    assert file_parts[0].mime_type == "image/png"


@pytest.mark.asyncio
async def test_convert_message_to_input_keeps_placeholder_when_download_fails() -> None:
    provider = DummyWhatsAppProvider()
    provider.download_media = AsyncMock(
        side_effect=EvolutionAPIError("No base64 data in media response")
    )
    bot = make_bot(provider)

    message = WhatsAppVideoMessage(
        id="video-failure",
        from_number="5511999999999",
        to_number="tests",
        push_name="Karla Marina",
        timestamp=datetime.now(),
        media_url="https://example.com/video.mp4",
        media_mime_type="video/mp4",
        caption=None,
    )

    converted = await bot._convert_message_to_input(message, make_session())

    text_parts = [part for part in converted.parts if isinstance(part, TextPart)]
    assert any(
        str(part.text) == "[Media file - failed to download]" for part in text_parts
    )


def make_evolution_provider() -> EvolutionAPIProvider:
    return EvolutionAPIProvider(
        config=EvolutionAPIConfig(
            base_url="https://evolution.example.com",
            instance_name="test-instance",
            api_key="secret",
        ),
        enable_circuit_breaker=False,
        enable_rate_limiting=False,
    )


@pytest.mark.asyncio
async def test_download_media_accepts_top_level_response_shape() -> None:
    provider = make_evolution_provider()
    provider._make_request_with_resilience = AsyncMock(
        return_value={
            "base64": base64.b64encode(b"top-level").decode("ascii"),
            "mimetype": "image/png; charset=utf-8",
        }
    )

    media = await provider.download_media("media-top-level")

    provider._make_request_with_resilience.assert_awaited_once()
    call_args = provider._make_request_with_resilience.await_args
    assert call_args.args[0] == "POST"
    assert call_args.args[2] == {
        "message": {"key": {"id": "media-top-level"}},
        "convertToMp4": True,
    }
    assert media.data == b"top-level"
    assert media.mime_type == "image/png"


@pytest.mark.asyncio
async def test_download_media_accepts_nested_data_response_shape() -> None:
    provider = make_evolution_provider()
    provider._make_request_with_resilience = AsyncMock(
        return_value={
            "status": "ok",
            "data": {
                "base64": base64.b64encode(b"nested").decode("ascii"),
                "mimetype": "application/pdf",
            },
        }
    )

    media = await provider.download_media("media-nested")

    assert media.data == b"nested"
    assert media.mime_type == "application/pdf"


@pytest.mark.asyncio
async def test_download_media_falls_back_to_octet_stream_when_mimetype_missing() -> None:
    provider = make_evolution_provider()
    provider._make_request_with_resilience = AsyncMock(
        return_value={
            "base64": base64.b64encode(b"unknown-type").decode("ascii"),
        }
    )

    media = await provider.download_media("media-no-mimetype")

    assert media.data == b"unknown-type"
    assert media.mime_type == "application/octet-stream"


@pytest.mark.asyncio
async def test_download_media_logs_response_shape_when_base64_missing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider = make_evolution_provider()
    provider._make_request_with_resilience = AsyncMock(
        return_value={
            "status": "ok",
            "data": {"mimetype": "application/pdf"},
        }
    )

    with caplog.at_level(logging.ERROR):
        with pytest.raises(EvolutionAPIError, match="No base64 data in media response"):
            await provider.download_media("media-missing-base64")

    assert "media-missing-base64" in caplog.text
    assert "response_keys=['data', 'status']" in caplog.text
    assert "nested_data_keys=['mimetype']" in caplog.text
