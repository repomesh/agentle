from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class ChannelCapabilities(BaseModel):
    """Feature flags for a messaging channel provider."""

    supports_typing_indicator: bool = Field(default=False)
    supports_recording_indicator: bool = Field(default=False)
    supports_read_receipt: bool = Field(default=False)
    supports_quoting: bool = Field(default=False)
    supports_media: bool = Field(default=False)
    supports_audio: bool = Field(default=False)
    supports_markdown: bool = Field(default=False)
