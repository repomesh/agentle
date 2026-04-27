from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, MutableMapping
from datetime import datetime
from typing import Any
from urllib.parse import urljoin

import aiohttp
from rsb.functions.bytes2mime import bytes2mime

from agentle.agents.channels.models.channel_capabilities import ChannelCapabilities
from agentle.agents.channels.models.channel_media import ChannelMedia
from agentle.agents.channels.models.channel_message import ChannelMessage
from agentle.agents.channels.models.channel_send_result import ChannelSendResult
from agentle.agents.channels.models.channel_session import ChannelSession
from agentle.agents.channels.models.downloaded_channel_media import (
    DownloadedChannelMedia,
)
from agentle.agents.channels.providers.instagram_direct.instagram_direct_config import (
    InstagramDirectConfig,
)
from agentle.sessions.in_memory_session_store import InMemorySessionStore
from agentle.sessions.session_manager import SessionManager

logger = logging.getLogger(__name__)
ExpectedStatus = int | tuple[int, ...]


class InstagramDirectError(Exception):
    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_data: Mapping[str, Any] | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data


class InstagramDirectProvider:
    """Instagram Direct provider for the generic channel runtime."""

    config: InstagramDirectConfig
    session_manager: SessionManager[ChannelSession]
    session_ttl_seconds: int
    _session: aiohttp.ClientSession | None
    _owns_session_manager: bool

    def __init__(
        self,
        config: InstagramDirectConfig,
        session_manager: SessionManager[ChannelSession] | None = None,
        session_ttl_seconds: int = 3600,
    ):
        self.config = config
        self.session_ttl_seconds = session_ttl_seconds
        self._session = None
        self._owns_session_manager = session_manager is None
        self.session_manager = session_manager or SessionManager(
            session_store=InMemorySessionStore[ChannelSession](),
            default_ttl_seconds=session_ttl_seconds,
        )

    @property
    def capabilities(self) -> ChannelCapabilities:
        return ChannelCapabilities(
            supports_typing_indicator=False,
            supports_recording_indicator=False,
            supports_read_receipt=False,
            supports_quoting=False,
            supports_media=True,
            supports_audio=False,
            supports_markdown=False,
        )

    @property
    def session(self) -> aiohttp.ClientSession:
        if self._session is None:
            headers = {
                "Authorization": f"Bearer {self.config.access_token}",
                "Content-Type": "application/json",
            }
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self._session = aiohttp.ClientSession(headers=headers, timeout=timeout)
        return self._session

    def get_resource_identifier(self) -> str:
        return self.config.instagram_user_id

    async def initialize(self) -> None:
        if not self.config.access_token:
            raise InstagramDirectError("Access token is required.")
        if not self.config.instagram_user_id:
            raise InstagramDirectError("Instagram user ID is required.")
        logger.info(
            "Instagram Direct provider initialized for instagram_user_id=%s",
            self.config.instagram_user_id,
        )

    async def shutdown(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None
        if self._owns_session_manager:
            await self.session_manager.close()

    def _build_url(self, endpoint: str) -> str:
        return urljoin(self.config.base_url, f"/{self.config.api_version}/{endpoint}")

    @staticmethod
    async def _read_response_payload(
        response: aiohttp.ClientResponse,
    ) -> Mapping[str, Any]:
        try:
            payload = await response.json()
            return payload if isinstance(payload, Mapping) else {"value": payload}
        except Exception:
            return {"message": await response.text()}

    async def _handle_response(
        self, response: aiohttp.ClientResponse, expected_status: ExpectedStatus
    ) -> Mapping[str, Any]:
        payload = await self._read_response_payload(response)
        expected_statuses = (
            (expected_status,) if isinstance(expected_status, int) else expected_status
        )
        if response.status in expected_statuses:
            return payload

        error = payload.get("error") if isinstance(payload, Mapping) else None
        if isinstance(error, Mapping):
            message = str(error.get("message") or error.get("code") or response.status)
        else:
            message = str(payload.get("message") or response.status)
        raise InstagramDirectError(message, response.status, payload)

    async def _make_request(
        self,
        method: str,
        url: str,
        data: Mapping[str, Any] | None = None,
        expected_status: ExpectedStatus = 200,
    ) -> Mapping[str, Any]:
        method = method.upper()
        if method == "GET":
            async with self.session.get(url) as response:
                return await self._handle_response(response, expected_status)
        if method == "POST":
            async with self.session.post(url, json=data) as response:
                return await self._handle_response(response, expected_status)
        raise ValueError(f"Unsupported HTTP method: {method}")

    async def _make_request_with_retry(
        self,
        method: str,
        url: str,
        data: Mapping[str, Any] | None = None,
        expected_status: ExpectedStatus = 200,
    ) -> Mapping[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                return await self._make_request(method, url, data, expected_status)
            except InstagramDirectError as exc:
                last_error = exc
                if exc.status_code in {400, 401, 403, 404}:
                    raise
                if attempt < self.config.max_retries:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
        if last_error:
            raise last_error
        raise InstagramDirectError("Request failed.")

    async def send_text_message(
        self, to: str, text: str, quoted_message_id: str | None = None
    ) -> ChannelSendResult:
        del quoted_message_id
        payload = {
            "recipient": {"id": self._normalize_contact_identifier(to)},
            "message": {"text": text},
        }
        url = self._build_url(f"{self.config.instagram_user_id}/messages")
        response = await self._make_request_with_retry(
            "POST", url, payload, expected_status=(200, 201)
        )
        message_id = str(response.get("message_id") or response.get("id") or "")
        return ChannelSendResult(
            id=message_id,
            provider="instagram_direct",
            resource_id=self.config.instagram_user_id,
            recipient_id=to,
            raw=dict(response),
        )

    async def send_media_message(
        self,
        to: str,
        media_url: str,
        media_type: str,
        caption: str | None = None,
        filename: str | None = None,
        quoted_message_id: str | None = None,
    ) -> ChannelSendResult:
        del caption, filename, quoted_message_id
        attachment_type = self._media_type_to_attachment_type(media_type)
        payload: MutableMapping[str, Any] = {
            "recipient": {"id": self._normalize_contact_identifier(to)},
            "message": {
                "attachment": {
                    "type": attachment_type,
                    "payload": {"url": media_url},
                }
            },
        }
        url = self._build_url(f"{self.config.instagram_user_id}/messages")
        response = await self._make_request_with_retry(
            "POST", url, payload, expected_status=(200, 201)
        )
        message_id = str(response.get("message_id") or response.get("id") or "")
        return ChannelSendResult(
            id=message_id,
            provider="instagram_direct",
            resource_id=self.config.instagram_user_id,
            recipient_id=to,
            raw=dict(response),
        )

    async def send_typing_indicator(self, to: str, duration: int = 3) -> None:
        del to, duration

    async def send_recording_indicator(self, to: str, duration: int = 3) -> None:
        del to, duration

    async def mark_message_as_read(self, message_id: str) -> None:
        del message_id

    async def get_session(self, contact_identifier: str) -> ChannelSession | None:
        normalized = self._normalize_contact_identifier(contact_identifier)
        session_id = self._session_id(normalized)
        session = await self.session_manager.get_session(session_id, refresh_ttl=True)
        if session:
            session.last_activity = datetime.now()
            await self.session_manager.update_session(session_id, session)
            return session

        session = ChannelSession(
            session_id=session_id,
            contact_identifier=normalized,
        )
        created = await self.session_manager.create_session(
            session_id,
            session,
            ttl_seconds=self.session_ttl_seconds,
        )
        if not created:
            existing = await self.session_manager.get_session(
                session_id, refresh_ttl=True
            )
            if existing is not None:
                existing.last_activity = datetime.now()
                await self.session_manager.update_session(session_id, existing)
                return existing
        return session

    async def update_session(self, session: ChannelSession) -> None:
        session.last_activity = datetime.now()
        await self.session_manager.update_session(
            session.session_id,
            session,
            ttl_seconds=self.session_ttl_seconds,
        )

    async def download_media(self, media_id: str) -> DownloadedChannelMedia:
        if not str(media_id).lower().startswith(("http://", "https://")):
            raise InstagramDirectError("Instagram media download requires a URL.")
        async with self.session.get(media_id) as response:
            if response.status != 200:
                payload = await self._read_response_payload(response)
                raise InstagramDirectError(
                    "Failed to download Instagram media.", response.status, payload
                )
            data = await response.read()
        return DownloadedChannelMedia(data=data, mime_type=bytes2mime(data))

    def parse_channel_messages(self, payload: Mapping[str, Any]) -> list[ChannelMessage]:
        return self.webhook_to_channel_messages(
            payload,
            instagram_user_id=self.config.instagram_user_id,
        )

    @classmethod
    def webhook_to_channel_messages(
        cls,
        payload: Mapping[str, Any],
        *,
        instagram_user_id: str | None = None,
    ) -> list[ChannelMessage]:
        messages: list[ChannelMessage] = []
        for entry in payload.get("entry") or []:
            if not isinstance(entry, Mapping):
                continue
            resource_id = str(instagram_user_id or entry.get("id") or "")
            for event in entry.get("messaging") or []:
                if not isinstance(event, Mapping):
                    continue
                message = event.get("message")
                if isinstance(message, Mapping) and message.get("is_echo"):
                    continue
                try:
                    messages.append(
                        cls.messaging_event_to_channel_message(
                            event,
                            resource_id=resource_id,
                        )
                    )
                except ValueError:
                    continue
        return messages

    @classmethod
    def messaging_event_to_channel_message(
        cls,
        event: Mapping[str, Any],
        *,
        resource_id: str | None = None,
    ) -> ChannelMessage:
        message = event.get("message")
        if not isinstance(message, Mapping):
            raise ValueError("Evento Instagram Direct sem mensagem de entrada.")

        sender = event.get("sender") or {}
        recipient = event.get("recipient") or {}
        sender_id = str(sender.get("id") or "").strip()
        recipient_id = str(recipient.get("id") or "").strip()
        if not sender_id:
            raise ValueError("Evento Instagram Direct sem remetente.")

        attachments = message.get("attachments") or []
        media = cls._first_attachment_to_media(attachments)
        text = str(message.get("text") or "").strip()
        timestamp = cls._parse_timestamp(event.get("timestamp"))
        message_id = str(message.get("mid") or message.get("id") or "").strip()
        if not message_id:
            message_id = f"{sender_id}:{int(timestamp.timestamp() * 1000)}"

        return ChannelMessage(
            id=message_id,
            provider="instagram_direct",
            resource_id=str(resource_id or recipient_id or ""),
            conversation_id=sender_id,
            sender_id=sender_id,
            sender_display_name=str(sender.get("username") or sender_id),
            message_type="media" if media and not text else "text",
            text=text or (media.caption if media else ""),
            media=media,
            timestamp=timestamp,
            metadata={
                "instagram_sender_id": sender_id,
                "instagram_recipient_id": recipient_id,
                "is_echo": bool(message.get("is_echo")),
                "raw_event": dict(event),
            },
        )

    def _session_id(self, contact_identifier: str) -> str:
        return f"instagram_direct:{self.config.instagram_user_id}:{contact_identifier}"

    @staticmethod
    def _normalize_contact_identifier(value: str) -> str:
        return str(value or "").strip()

    @staticmethod
    def _parse_timestamp(value: Any) -> datetime:
        if not value:
            return datetime.now()
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            text = str(value)
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            try:
                return datetime.fromisoformat(text)
            except ValueError:
                return datetime.now()

        if numeric > 10_000_000_000:
            numeric = numeric / 1000
        return datetime.fromtimestamp(numeric)

    @staticmethod
    def _first_attachment_to_media(
        attachments: list[dict[str, Any]],
    ) -> ChannelMedia | None:
        if not attachments:
            return None
        attachment = attachments[0]
        payload = attachment.get("payload") or {}
        url = str(payload.get("url") or "") or None
        media_id = str(payload.get("id") or url or "") or None
        media_type = str(attachment.get("type") or "attachment")
        return ChannelMedia(
            media_type=media_type,
            media_id=media_id,
            url=url,
            mime_type=None,
            filename=None,
            caption=None,
            metadata=dict(attachment),
        )

    @staticmethod
    def _media_type_to_attachment_type(media_type: str) -> str:
        value = str(media_type or "").lower()
        if value.startswith("image"):
            return "image"
        if value.startswith("video"):
            return "video"
        if value.startswith("audio"):
            return "audio"
        if value in {"image", "video", "audio", "file"}:
            return value
        return "file"
