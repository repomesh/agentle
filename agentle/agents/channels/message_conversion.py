from __future__ import annotations

import logging
from collections.abc import Sequence

from agentle.agents.channels.models.channel_message import ChannelMessage
from agentle.agents.channels.providers.base import ChannelProvider
from agentle.generations.models.message_parts.file import FilePart
from agentle.generations.models.message_parts.text import TextPart
from agentle.generations.models.messages.user_message import UserMessage

logger = logging.getLogger(__name__)


async def channel_messages_to_parts(
    messages: Sequence[ChannelMessage],
    provider: ChannelProvider,
) -> list[TextPart | FilePart]:
    parts: list[TextPart | FilePart] = []

    for index, message in enumerate(messages):
        if index > 0:
            parts.append(TextPart(text="\n\n"))

        if message.text:
            parts.append(TextPart(text=message.text))

        if not message.media:
            continue

        try:
            if message.media.base64_data:
                parts.append(
                    FilePart(
                        data=message.media.base64_data,
                        mime_type=message.media.mime_type or "application/octet-stream",
                    )
                )
            elif message.media.media_id:
                media = await provider.download_media(message.media.media_id)
                parts.append(FilePart(data=media.data, mime_type=media.mime_type))
            elif message.media.url:
                parts.append(TextPart(text=message.media.url))

            if message.media.caption and message.media.caption != message.text:
                parts.append(TextPart(text=f"Caption: {message.media.caption}"))
        except Exception:
            logger.exception("Failed to download channel media")
            parts.append(TextPart(text="[Media file - failed to download]"))

    return parts or [TextPart(text="")]


async def channel_messages_to_user_message(
    messages: Sequence[ChannelMessage],
    provider: ChannelProvider,
) -> UserMessage:
    parts = await channel_messages_to_parts(messages, provider)
    display_name = messages[0].sender_display_name if messages else None
    return UserMessage.create_named(parts=parts, name=display_name)
