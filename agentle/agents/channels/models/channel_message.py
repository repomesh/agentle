from __future__ import annotations

from datetime import datetime
from typing import Any

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.agents.channels.models.channel_media import ChannelMedia


class ChannelMessage(BaseModel):
    """Provider-neutral inbound message."""

    id: str
    provider: str
    resource_id: str
    conversation_id: str
    sender_id: str
    sender_display_name: str | None = None
    message_type: str = "text"
    text: str | None = None
    media: ChannelMedia | None = None
    timestamp: datetime = Field(default_factory=datetime.now)
    quoted_message_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def contact_identifier(self) -> str:
        return self.sender_id
