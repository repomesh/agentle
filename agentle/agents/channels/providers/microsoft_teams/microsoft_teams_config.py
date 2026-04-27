from __future__ import annotations

from typing import Any

from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class MicrosoftTeamsConfig(BaseModel):
    """Configuration for Microsoft Teams through the Bot Framework Connector."""

    app_id: str
    app_password: str = ""
    bot_name: str = ""
    default_service_url: str = "https://smba.trafficmanager.net/teams/"
    oauth_token_url: str = (
        "https://login.microsoftonline.com/botframework.com/oauth2/v2.0/token"
    )
    oauth_scope: str = "https://api.botframework.com/.default"
    access_token: str | None = Field(default=None)
    token_refresh_margin_seconds: int = Field(default=300)
    timeout: int = Field(default=30)
    max_retries: int = Field(default=3)
    retry_delay: float = Field(default=1.0)
    contact_identifier_strategy: str = Field(default="conversation_user")
    conversation_references: dict[str, dict[str, Any]] = Field(default_factory=dict)
