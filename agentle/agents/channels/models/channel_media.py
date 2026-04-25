from typing import Any

from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class ChannelMedia(BaseModel):
    """Normalized media attachment for any messaging channel."""

    media_type: str
    media_id: str | None = None
    url: str | None = None
    mime_type: str | None = None
    caption: str | None = None
    filename: str | None = None
    base64_data: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
