from __future__ import annotations

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.tts.speech_config import SpeechConfig


class ChannelBotConfig(BaseModel):
    """Configuration for provider-neutral messaging channel bots."""

    typing_indicator: bool = Field(default=True)
    typing_duration: int = Field(default=3)
    auto_read_messages: bool = Field(default=True)
    quote_messages: bool = Field(default=False)
    session_timeout_minutes: int = Field(default=30)
    max_message_length: int = Field(default=4096)
    max_split_messages: int = Field(default=5)
    error_message: str | None = Field(default=None)
    welcome_message: str | None = Field(default=None)
    welcome_image_url: str | None = Field(default=None)
    welcome_image_base64: str | None = Field(default=None)

    enable_message_batching: bool = Field(default=True)
    batch_delay_seconds: float = Field(default=3.0)
    max_batch_size: int = Field(default=10)
    max_batch_timeout_seconds: float = Field(default=15.0)

    spam_protection_enabled: bool = Field(default=True)
    min_message_interval_seconds: float = Field(default=0.5)
    max_messages_per_minute: int = Field(default=20)
    rate_limit_cooldown_seconds: int = Field(default=60)

    debug_mode: bool = Field(default=False)
    track_response_times: bool = Field(default=True)
    slow_response_threshold_seconds: float = Field(default=10.0)

    speech_play_chance: float = Field(default=0.0, ge=0.0, le=1.0)
    speech_config: SpeechConfig | None = Field(default=None)

    retry_failed_messages: bool = Field(default=True)
    max_retry_attempts: int = Field(default=3)
    retry_delay_seconds: float = Field(default=1.0)

    enable_human_delays: bool = Field(default=False)
    min_read_delay_seconds: float = Field(default=2.0, ge=0.0)
    max_read_delay_seconds: float = Field(default=15.0, ge=0.0)
    min_typing_delay_seconds: float = Field(default=3.0, ge=0.0)
    max_typing_delay_seconds: float = Field(default=45.0, ge=0.0)
