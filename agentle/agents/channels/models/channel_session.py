from __future__ import annotations

import logging
import uuid
from collections.abc import MutableMapping, MutableSequence
from datetime import datetime, timedelta
from typing import Any

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

logger = logging.getLogger(__name__)


class ChannelSession(BaseModel):
    """Provider-neutral conversation session with batching and rate limiting."""

    session_id: str
    contact_identifier: str
    contact_display_name: str | None = None
    started_at: datetime = Field(default_factory=datetime.now)
    last_activity: datetime = Field(default_factory=datetime.now)
    message_count: int = 0
    is_active: bool = True
    context_data: MutableMapping[str, Any] = Field(default_factory=dict)

    is_processing: bool = Field(default=False)
    pending_messages: MutableSequence[dict[str, Any]] = Field(default_factory=list)
    batch_started_at: datetime | None = None
    batch_timeout_at: datetime | None = None
    last_message_added_at: datetime | None = None

    last_message_at: datetime | None = None
    messages_in_current_minute: int = 0
    current_minute_start: datetime | None = None
    is_rate_limited: bool = False
    rate_limit_until: datetime | None = None
    processing_token: str | None = None
    last_state_change: datetime = Field(default_factory=datetime.now)

    def add_pending_message(self, message_data: dict[str, Any]) -> None:
        now = datetime.now()
        self.pending_messages.append(message_data)
        self.last_message_added_at = now

        if self.is_processing and self.batch_started_at:
            self.batch_started_at = now

        if not self.is_processing:
            self.last_activity = now
            self.last_state_change = now

    def clear_pending_messages(self) -> MutableSequence[dict[str, Any]]:
        messages = list(self.pending_messages)
        self.pending_messages.clear()
        self.last_state_change = datetime.now()
        return messages

    def update_rate_limiting(
        self, max_messages_per_minute: int, cooldown_seconds: int
    ) -> bool:
        now = datetime.now()

        if (
            self.is_rate_limited
            and self.rate_limit_until
            and now >= self.rate_limit_until
        ):
            self.is_rate_limited = False
            self.rate_limit_until = None
            self.messages_in_current_minute = 0
            self.current_minute_start = None
            self.last_state_change = now

        if self.is_rate_limited:
            return False

        if (
            self.current_minute_start is None
            or (now - self.current_minute_start).total_seconds() >= 60
        ):
            self.current_minute_start = now
            self.messages_in_current_minute = 0

        self.messages_in_current_minute += 1
        self.last_message_at = now

        if self.messages_in_current_minute > max_messages_per_minute:
            self.is_rate_limited = True
            self.rate_limit_until = now + timedelta(seconds=cooldown_seconds)
            self.last_state_change = now
            return False

        return True

    def should_process_batch(
        self, batch_delay_seconds: float, max_wait_seconds: float
    ) -> bool:
        del max_wait_seconds
        if not self.pending_messages:
            return False

        now = datetime.now()
        if self.batch_timeout_at and now >= self.batch_timeout_at:
            return True

        reference_time = self.last_message_added_at or self.batch_started_at
        if not reference_time:
            return False

        return (now - reference_time).total_seconds() >= batch_delay_seconds

    def start_batch_processing(self, max_wait_seconds: float) -> str:
        now = datetime.now()
        token = str(uuid.uuid4())
        self.is_processing = True
        self.batch_started_at = now
        self.batch_timeout_at = now + timedelta(seconds=max_wait_seconds)
        self.processing_token = token
        self.last_state_change = now
        return token

    def finish_batch_processing(self, processing_token: str | None = None) -> bool:
        if processing_token is not None and self.processing_token != processing_token:
            logger.warning(
                "Channel batch token mismatch for %s: expected=%s got=%s",
                self.contact_identifier,
                self.processing_token,
                processing_token,
            )
            return False

        now = datetime.now()
        self.is_processing = False
        self.batch_started_at = None
        self.batch_timeout_at = None
        self.last_message_added_at = None
        self.processing_token = None
        self.last_activity = now
        self.last_state_change = now
        return True

    def reset_session(self) -> None:
        self.is_processing = False
        self.pending_messages.clear()
        self.batch_started_at = None
        self.batch_timeout_at = None
        self.processing_token = None
        self.is_rate_limited = False
        self.rate_limit_until = None
        self.messages_in_current_minute = 0
        self.current_minute_start = None
        self.last_activity = datetime.now()
        self.last_state_change = datetime.now()
