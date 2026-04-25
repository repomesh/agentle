from typing import Any

from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class ChannelSendResult(BaseModel):
    """Normalized result for outbound channel sends."""

    id: str
    provider: str
    resource_id: str
    recipient_id: str
    raw: dict[str, Any] = Field(default_factory=dict)
