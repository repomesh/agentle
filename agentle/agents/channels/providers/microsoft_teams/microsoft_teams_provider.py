from __future__ import annotations

import asyncio
import html
import logging
import time
from collections.abc import Mapping
from datetime import datetime
from typing import Any
from urllib.parse import quote, urljoin

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
from agentle.agents.channels.providers.microsoft_teams.microsoft_teams_config import (
    MicrosoftTeamsConfig,
)
from agentle.sessions.in_memory_session_store import InMemorySessionStore
from agentle.sessions.session_manager import SessionManager

logger = logging.getLogger(__name__)
ExpectedStatus = int | tuple[int, ...]


class MicrosoftTeamsError(Exception):
    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_data: Mapping[str, Any] | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data


class MicrosoftTeamsProvider:
    """Microsoft Teams provider backed by the Bot Framework Connector REST API."""

    config: MicrosoftTeamsConfig
    session_manager: SessionManager[ChannelSession]
    session_ttl_seconds: int
    _session: aiohttp.ClientSession | None
    _owns_session_manager: bool
    _access_token: str | None
    _access_token_expires_at: float
    _conversation_references: dict[str, dict[str, Any]]

    def __init__(
        self,
        config: MicrosoftTeamsConfig,
        session_manager: SessionManager[ChannelSession] | None = None,
        session_ttl_seconds: int = 3600,
    ):
        self.config = config
        self.session_ttl_seconds = session_ttl_seconds
        self._session = None
        self._owns_session_manager = session_manager is None
        self._access_token = config.access_token
        self._access_token_expires_at = float("inf") if config.access_token else 0.0
        self._conversation_references = {
            str(key): dict(value)
            for key, value in (config.conversation_references or {}).items()
        }
        self.session_manager = session_manager or SessionManager(
            session_store=InMemorySessionStore[ChannelSession](),
            default_ttl_seconds=session_ttl_seconds,
        )

    @property
    def capabilities(self) -> ChannelCapabilities:
        return ChannelCapabilities(
            supports_typing_indicator=True,
            supports_recording_indicator=False,
            supports_read_receipt=False,
            supports_quoting=True,
            supports_media=True,
            supports_audio=False,
            supports_markdown=True,
        )

    @property
    def session(self) -> aiohttp.ClientSession:
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    def get_resource_identifier(self) -> str:
        return self.config.app_id

    async def initialize(self) -> None:
        if not self.config.app_id:
            raise MicrosoftTeamsError("Microsoft Teams app_id is required.")
        if not self.config.app_password and not self.config.access_token:
            raise MicrosoftTeamsError(
                "Microsoft Teams app_password or access_token is required."
            )
        await self._get_access_token()
        logger.info(
            "Microsoft Teams provider initialized for app_id=%s", self.config.app_id
        )

    async def shutdown(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None
        if self._owns_session_manager:
            await self.session_manager.close()

    async def _get_access_token(self) -> str:
        now = time.time()
        if (
            self._access_token
            and now
            < self._access_token_expires_at - self.config.token_refresh_margin_seconds
        ):
            return self._access_token

        if self.config.access_token and not self.config.app_password:
            self._access_token = self.config.access_token
            self._access_token_expires_at = float("inf")
            return self._access_token

        data = {
            "grant_type": "client_credentials",
            "client_id": self.config.app_id,
            "client_secret": self.config.app_password,
            "scope": self.config.oauth_scope,
        }
        async with self.session.post(self.config.oauth_token_url, data=data) as response:
            payload = await self._read_response_payload(response)
            if response.status != 200:
                raise MicrosoftTeamsError(
                    "Failed to acquire Microsoft Teams Bot Framework access token.",
                    response.status,
                    payload,
                )

        token = str(payload.get("access_token") or "")
        if not token:
            raise MicrosoftTeamsError("Bot Framework token response has no access_token.")

        expires_in = int(payload.get("expires_in") or 3600)
        self._access_token = token
        self._access_token_expires_at = now + expires_in
        return token

    async def _headers(self) -> dict[str, str]:
        token = await self._get_access_token()
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

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
        raise MicrosoftTeamsError(message, response.status, payload)

    async def _make_request(
        self,
        method: str,
        url: str,
        data: Mapping[str, Any] | None = None,
        expected_status: ExpectedStatus = 200,
    ) -> Mapping[str, Any]:
        headers = await self._headers()
        method = method.upper()
        if method == "GET":
            async with self.session.get(url, headers=headers) as response:
                return await self._handle_response(response, expected_status)
        if method == "POST":
            async with self.session.post(url, json=data, headers=headers) as response:
                return await self._handle_response(response, expected_status)
        if method == "PUT":
            async with self.session.put(url, json=data, headers=headers) as response:
                return await self._handle_response(response, expected_status)
        if method == "DELETE":
            async with self.session.delete(url, headers=headers) as response:
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
            except MicrosoftTeamsError as exc:
                last_error = exc
                if exc.status_code in {400, 401, 403, 404}:
                    raise
                if attempt < self.config.max_retries:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
        if last_error:
            raise last_error
        raise MicrosoftTeamsError("Request failed.")

    async def send_text_message(
        self, to: str, text: str, quoted_message_id: str | None = None
    ) -> ChannelSendResult:
        reference = await self._resolve_conversation_reference(to)
        conversation_id = self._conversation_id_from_reference(reference)
        payload = self._activity_payload(
            reference,
            text=text,
            quoted_message_id=quoted_message_id,
        )
        url = self._activity_url(reference, conversation_id, quoted_message_id)
        response = await self._make_request_with_retry(
            "POST", url, payload, expected_status=(200, 201, 202)
        )
        message_id = str(response.get("id") or response.get("activityId") or "")
        return ChannelSendResult(
            id=message_id,
            provider="microsoft_teams",
            resource_id=self.config.app_id,
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
        reference = await self._resolve_conversation_reference(to)
        conversation_id = self._conversation_id_from_reference(reference)
        content_type = self._media_type_to_content_type(media_type)
        payload = self._activity_payload(
            reference,
            text=caption or "",
            quoted_message_id=quoted_message_id,
            attachments=[
                {
                    "contentType": content_type,
                    "contentUrl": media_url,
                    "name": filename or media_url.rsplit("/", 1)[-1] or "attachment",
                }
            ],
        )
        url = self._activity_url(reference, conversation_id, quoted_message_id)
        response = await self._make_request_with_retry(
            "POST", url, payload, expected_status=(200, 201, 202)
        )
        message_id = str(response.get("id") or response.get("activityId") or "")
        return ChannelSendResult(
            id=message_id,
            provider="microsoft_teams",
            resource_id=self.config.app_id,
            recipient_id=to,
            raw=dict(response),
        )

    async def send_typing_indicator(self, to: str, duration: int = 3) -> None:
        del duration
        reference = await self._resolve_conversation_reference(to)
        conversation_id = self._conversation_id_from_reference(reference)
        payload = self._activity_payload(reference, activity_type="typing")
        url = self._activity_url(reference, conversation_id, None)
        await self._make_request_with_retry(
            "POST", url, payload, expected_status=(200, 201, 202)
        )

    async def send_recording_indicator(self, to: str, duration: int = 3) -> None:
        del to, duration

    async def mark_message_as_read(self, message_id: str) -> None:
        del message_id

    async def get_session(self, contact_identifier: str) -> ChannelSession | None:
        normalized = self._normalize_identifier(contact_identifier)
        session_id = self._session_id(normalized)
        session = await self.session_manager.get_session(session_id, refresh_ttl=True)
        reference = self._conversation_references.get(normalized)

        if session:
            session.last_activity = datetime.now()
            if reference:
                session.context_data["teams_reference"] = reference
            await self.session_manager.update_session(session_id, session)
            return session

        session = ChannelSession(
            session_id=session_id,
            contact_identifier=normalized,
            context_data={"teams_reference": reference} if reference else {},
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
                if reference:
                    existing.context_data["teams_reference"] = reference
                await self.session_manager.update_session(session_id, existing)
                return existing
        return session

    async def update_session(self, session: ChannelSession) -> None:
        session.last_activity = datetime.now()
        reference = self._conversation_references.get(session.contact_identifier)
        if reference:
            session.context_data["teams_reference"] = reference
        await self.session_manager.update_session(
            session.session_id,
            session,
            ttl_seconds=self.session_ttl_seconds,
        )

    async def download_media(self, media_id: str) -> DownloadedChannelMedia:
        if not str(media_id).lower().startswith(("http://", "https://")):
            raise MicrosoftTeamsError("Teams media download requires a content URL.")
        headers = await self._headers()
        async with self.session.get(media_id, headers=headers) as response:
            if response.status != 200:
                payload = await self._read_response_payload(response)
                raise MicrosoftTeamsError(
                    "Failed to download Teams media.", response.status, payload
                )
            data = await response.read()
        return DownloadedChannelMedia(data=data, mime_type=bytes2mime(data))

    def parse_channel_message(self, activity: dict[str, Any]) -> ChannelMessage:
        message = self.activity_to_channel_message(
            activity,
            app_id=self.config.app_id,
            contact_identifier_strategy=self.config.contact_identifier_strategy,
        )
        self.remember_conversation_reference(message)
        return message

    def remember_conversation_reference(self, message: ChannelMessage) -> None:
        reference = dict(message.metadata.get("teams_reference") or {})
        if not reference:
            return
        self._conversation_references[message.contact_identifier] = reference

    @classmethod
    def activity_to_channel_message(
        cls,
        activity: dict[str, Any],
        *,
        app_id: str | None = None,
        contact_identifier_strategy: str = "conversation_user",
    ) -> ChannelMessage:
        activity_type = str(activity.get("type") or "")
        if activity_type != "message":
            raise ValueError("Atividade Microsoft Teams sem mensagem de entrada.")

        activity_id = str(activity.get("id") or "").strip()
        conversation = activity.get("conversation") or {}
        conversation_id = str(conversation.get("id") or "").strip()
        sender = activity.get("from") or {}
        sender_account_id = str(sender.get("id") or "").strip()
        if not activity_id or not conversation_id or not sender_account_id:
            raise ValueError("Atividade Microsoft Teams sem id, conversa ou remetente.")

        text = cls._strip_bot_mention(
            str(activity.get("text") or ""),
            activity.get("entities") or [],
            activity.get("recipient") or {},
        )
        attachments = activity.get("attachments") or []
        media = cls._first_attachment_to_media(attachments)
        message_type = "media" if media and not text else "text"

        timestamp = cls._parse_timestamp(activity.get("timestamp"))
        reference = cls._conversation_reference(activity)
        contact_identifier = cls._contact_identifier(
            conversation_id,
            sender_account_id,
            contact_identifier_strategy,
        )
        tenant_id = (
            ((activity.get("channelData") or {}).get("tenant") or {}).get("id") or ""
        )

        return ChannelMessage(
            id=activity_id,
            provider="microsoft_teams",
            resource_id=str(
                app_id or (activity.get("recipient") or {}).get("id") or "msteams"
            ),
            conversation_id=conversation_id,
            sender_id=contact_identifier,
            sender_display_name=str(sender.get("name") or sender_account_id),
            message_type=message_type,
            text=text or (media.caption if media else ""),
            media=media,
            timestamp=timestamp,
            quoted_message_id=activity.get("replyToId"),
            metadata={
                "teams_user_id": sender_account_id,
                "teams_contact_identifier_strategy": contact_identifier_strategy,
                "teams_tenant_id": str(tenant_id),
                "teams_reference": reference,
                "raw_activity": activity,
            },
        )

    @staticmethod
    def _conversation_reference(activity: dict[str, Any]) -> dict[str, Any]:
        return {
            "service_url": activity.get("serviceUrl"),
            "channel_id": activity.get("channelId") or "msteams",
            "conversation": activity.get("conversation") or {},
            "bot": activity.get("recipient") or {},
            "user": activity.get("from") or {},
            "channel_data": activity.get("channelData") or {},
        }

    @staticmethod
    def _parse_timestamp(value: Any) -> datetime:
        if not value:
            return datetime.now()
        text = str(value)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            return datetime.now()

    @staticmethod
    def _strip_bot_mention(
        text: str,
        entities: list[dict[str, Any]],
        recipient: dict[str, Any],
    ) -> str:
        cleaned = text
        recipient_id = str(recipient.get("id") or "")
        for entity in entities:
            if entity.get("type") != "mention":
                continue
            mentioned = entity.get("mentioned") or {}
            if recipient_id and str(mentioned.get("id") or "") != recipient_id:
                continue
            mention_text = str(entity.get("text") or "")
            if mention_text:
                cleaned = cleaned.replace(mention_text, "")
        return html.unescape(cleaned).strip()

    @staticmethod
    def _first_attachment_to_media(
        attachments: list[dict[str, Any]],
    ) -> ChannelMedia | None:
        if not attachments:
            return None
        attachment = attachments[0]
        content_url = str(attachment.get("contentUrl") or "") or None
        content_type = str(attachment.get("contentType") or "") or None
        name = str(attachment.get("name") or "") or None
        return ChannelMedia(
            media_type="attachment",
            media_id=content_url,
            url=content_url,
            mime_type=content_type,
            filename=name,
            caption=None,
            metadata=dict(attachment),
        )

    @staticmethod
    def _contact_identifier(
        conversation_id: str,
        sender_account_id: str,
        strategy: str,
    ) -> str:
        normalized_strategy = str(strategy or "conversation_user").strip().lower()
        if normalized_strategy == "user":
            return sender_account_id
        if normalized_strategy == "conversation":
            return conversation_id
        return f"{conversation_id}:{sender_account_id}"

    async def _resolve_conversation_reference(
        self, contact_identifier: str
    ) -> dict[str, Any]:
        normalized = self._normalize_identifier(contact_identifier)
        reference = self._conversation_references.get(normalized)
        if reference:
            return reference

        session = await self.session_manager.get_session(self._session_id(normalized))
        if session:
            reference = session.context_data.get("teams_reference")
            if isinstance(reference, Mapping):
                return dict(reference)

        raise MicrosoftTeamsError(
            "No Microsoft Teams conversation reference found for contact identifier."
        )

    def _session_id(self, contact_identifier: str) -> str:
        return f"microsoft_teams:{self.config.app_id}:{contact_identifier}"

    @staticmethod
    def _normalize_identifier(value: str) -> str:
        return str(value or "").strip()

    @staticmethod
    def _conversation_id_from_reference(reference: Mapping[str, Any]) -> str:
        conversation = reference.get("conversation") or {}
        conversation_id = str(conversation.get("id") or "").strip()
        if not conversation_id:
            raise MicrosoftTeamsError(
                "Teams conversation reference has no conversation.id."
            )
        return conversation_id

    def _activity_url(
        self,
        reference: Mapping[str, Any],
        conversation_id: str,
        reply_to_id: str | None,
    ) -> str:
        service_url = str(
            reference.get("service_url") or self.config.default_service_url
        )
        endpoint = f"v3/conversations/{quote(conversation_id, safe='')}/activities"
        if reply_to_id:
            endpoint += f"/{quote(reply_to_id, safe='')}"
        return urljoin(service_url.rstrip("/") + "/", endpoint)

    def _activity_payload(
        self,
        reference: Mapping[str, Any],
        *,
        text: str = "",
        activity_type: str = "message",
        quoted_message_id: str | None = None,
        attachments: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        conversation = dict(reference.get("conversation") or {})
        user = dict(reference.get("user") or {})
        bot = dict(reference.get("bot") or {})
        if not bot.get("id"):
            bot["id"] = self.config.app_id
        if self.config.bot_name and not bot.get("name"):
            bot["name"] = self.config.bot_name

        payload: dict[str, Any] = {
            "type": activity_type,
            "channelId": reference.get("channel_id") or "msteams",
            "serviceUrl": reference.get("service_url") or self.config.default_service_url,
            "from": bot,
            "recipient": user,
            "conversation": conversation,
        }
        if activity_type == "message":
            payload["text"] = text
            if quoted_message_id:
                payload["replyToId"] = quoted_message_id
            if attachments:
                payload["attachments"] = attachments
        return payload

    @staticmethod
    def _media_type_to_content_type(media_type: str) -> str:
        value = str(media_type or "").lower()
        if "/" in value:
            return value
        return {
            "image": "image/png",
            "audio": "audio/mpeg",
            "video": "video/mp4",
            "document": "application/octet-stream",
            "file": "application/octet-stream",
        }.get(value, "application/octet-stream")
