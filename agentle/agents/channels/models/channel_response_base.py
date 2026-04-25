from rsb.models.base_model import BaseModel


class ChannelResponseBase(BaseModel):
    """Base structured response for channel bots."""

    response: str
