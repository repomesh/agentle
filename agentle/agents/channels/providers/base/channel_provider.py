from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from agentle.agents.channels.models.channel_capabilities import ChannelCapabilities
from agentle.agents.channels.models.channel_message import ChannelMessage
from agentle.agents.channels.models.channel_session import ChannelSession


class ChannelProvider(Protocol):
    """Provider contract for messaging channels."""

    @property
    def capabilities(self) -> ChannelCapabilities:
        ...

    def get_resource_identifier(self) -> str:
        ...

    async def initialize(self) -> None:
        ...

    async def shutdown(self) -> None:
        ...

    async def send_text_message(
        self, to: str, text: str, quoted_message_id: str | None = None
    ):
        ...

    async def send_media_message(
        self,
        to: str,
        media_url: str,
        media_type: str,
        caption: str | None = None,
        filename: str | None = None,
        quoted_message_id: str | None = None,
    ):
        ...

    async def send_typing_indicator(self, to: str, duration: int = 3) -> None:
        ...

    async def send_recording_indicator(self, to: str, duration: int = 3) -> None:
        ...

    async def mark_message_as_read(self, message_id: str) -> None:
        ...

    async def get_session(self, contact_identifier: str) -> ChannelSession | None:
        ...

    async def update_session(self, session: ChannelSession) -> None:
        ...

    async def download_media(self, media_id: str):
        ...

    def parse_channel_messages(
        self,
        payload: Mapping[str, Any],
        headers: Mapping[str, str] | None = None,
    ) -> Sequence[ChannelMessage]:
        ...
