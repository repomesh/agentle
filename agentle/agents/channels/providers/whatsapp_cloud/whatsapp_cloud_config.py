from __future__ import annotations

from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class WhatsAppCloudConfig(BaseModel):
    """Configuration for Meta's official WhatsApp Cloud API."""

    access_token: str
    phone_number_id: str
    business_account_id: str = ""
    app_id: str = ""
    app_secret: str = ""
    webhook_verify_token: str = ""
    api_version: str = "v24.0"
    base_url: str = "https://graph.facebook.com"
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    webhook_url: str | None = Field(default=None)
