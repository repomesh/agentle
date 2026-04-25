from rsb.models.base_model import BaseModel


class DownloadedChannelMedia(BaseModel):
    data: bytes
    mime_type: str
