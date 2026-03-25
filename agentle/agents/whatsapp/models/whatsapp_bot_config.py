from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.tts.speech_config import SpeechConfig


class WhatsAppBotConfig(BaseModel):
    """Configuration for WhatsApp bot behavior with simplified constructors and better organization.

    This configuration class provides comprehensive control over WhatsApp bot behavior including:
    - Core bot behavior (typing indicators, message reading, quoting)
    - Message batching for handling rapid message sequences
    - Spam protection and rate limiting
    - Human-like delays to simulate realistic human behavior patterns
    - Text-to-speech integration
    - Error handling and retry logic
    - Debug and monitoring settings

    Human-Like Delays Feature:
        The human-like delays feature simulates realistic human behavior patterns by introducing
        configurable delays at three critical points in message processing:

        1. Read Delay: Time between receiving a message and marking it as read
           - Simulates the time a human takes to read and comprehend a message
           - Calculated based on message length using realistic reading speeds

        2. Typing Delay: Time between generating a response and sending it
           - Simulates the time a human takes to compose and type a response
           - Calculated based on response length using realistic typing speeds

        3. Send Delay: Brief final delay before message transmission
           - Simulates the final review time before a human sends a message
           - Random delay within configured bounds

        These delays help prevent platform detection and account restrictions while
        maintaining natural interaction timing. All delays support jitter (random variation)
        to prevent detectable patterns.

    Configuration Presets:
        Use the class methods to create pre-configured instances optimized for specific use cases:
        - development(): Fast iteration with delays disabled
        - production(): Balanced configuration with delays enabled
        - high_volume(): Optimized for throughput with balanced delays
        - customer_service(): Professional timing with thoughtful delays
        - minimal(): Bare minimum configuration with delays disabled

    Examples:
        >>> # Create a production configuration with default delay settings
        >>> config = WhatsAppBotConfig.production()

        >>> # Create a custom configuration with specific delay bounds
        >>> config = WhatsAppBotConfig(
        ...     enable_human_delays=True,
        ...     min_read_delay_seconds=3.0,
        ...     max_read_delay_seconds=20.0,
        ...     min_typing_delay_seconds=5.0,
        ...     max_typing_delay_seconds=60.0
        ... )

        >>> # Override delay settings on an existing configuration
        >>> prod_config = WhatsAppBotConfig.production()
        >>> custom_config = prod_config.with_overrides(
        ...     min_read_delay_seconds=5.0,
        ...     max_typing_delay_seconds=90.0
        ... )
    """

    # === Core Bot Behavior ===
    typing_indicator: bool = Field(
        default=True, description="Show typing indicator while processing"
    )
    typing_duration: int = Field(
        default=3, description="Duration to show typing indicator in seconds"
    )
    auto_read_messages: bool = Field(
        default=True, description="Automatically mark messages as read"
    )
    quote_messages: bool = Field(
        default=False, description="Whether to quote user messages in replies"
    )
    session_timeout_minutes: int = Field(
        default=30, description="Minutes of inactivity before session reset"
    )
    max_message_length: int = Field(
        default=4096, description="Maximum message length (WhatsApp limit)"
    )
    max_split_messages: int = Field(
        default=5,
        description="Maximum number of split messages to send (remaining will be grouped)",
    )
    error_message: str | None = Field(
        default="Sorry, I encountered an error processing your message. Please try again.",
        description="Default error message",
    )
    welcome_message: str | None = Field(
        default=None, description="Message to send on first interaction (or caption if welcome image is set)"
    )
    welcome_image_url: str | None = Field(
        default=None, description="URL of welcome image to send on first interaction"
    )
    welcome_image_base64: str | None = Field(
        default=None, description="Base64-encoded welcome image to send on first interaction (alternative to welcome_image_url)"
    )

    # === Message Batching (Simplified) ===
    enable_message_batching: bool = Field(
        default=True, description="Enable message batching to prevent spam"
    )
    batch_delay_seconds: float = Field(
        default=3.0,
        description="Time to wait for additional messages before processing batch",
    )
    max_batch_size: int = Field(
        default=10, description="Maximum number of messages to batch together"
    )
    max_batch_timeout_seconds: float = Field(
        default=15.0,
        description="Maximum time to wait before forcing batch processing",
    )

    # === Spam Protection ===
    spam_protection_enabled: bool = Field(
        default=True, description="Enable spam protection mechanisms"
    )
    min_message_interval_seconds: float = Field(
        default=0.5,
        description="Minimum interval between processing messages from same user",
    )
    max_messages_per_minute: int = Field(
        default=20,
        description="Maximum messages per minute per user before rate limiting",
    )
    rate_limit_cooldown_seconds: int = Field(
        default=60, description="Cooldown period after rate limit is triggered"
    )

    # === Debug and Monitoring ===
    debug_mode: bool = Field(
        default=False, description="Enable comprehensive debug logging"
    )
    track_response_times: bool = Field(
        default=True,
        description="Track and log response times for performance monitoring",
    )
    slow_response_threshold_seconds: float = Field(
        default=10.0, description="Threshold for logging slow responses"
    )

    # === Text-to-Speech (TTS) ===
    speech_play_chance: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Probability (0.0-1.0) of sending audio response instead of text",
    )
    speech_config: SpeechConfig | None = Field(
        default=None,
        description="Optional SpeechConfig for TTS provider customization",
    )

    # === Error Handling ===
    retry_failed_messages: bool = Field(
        default=True, description="Retry processing failed messages"
    )
    max_retry_attempts: int = Field(
        default=3, description="Maximum number of retry attempts for failed messages"
    )
    retry_delay_seconds: float = Field(
        default=1.0, description="Delay between retry attempts"
    )

    # === Human-Like Delays ===
    enable_human_delays: bool = Field(
        default=False,
        description="Enable human-like delays for message processing to simulate realistic human behavior patterns",
    )
    min_read_delay_seconds: float = Field(
        default=2.0,
        ge=0.0,
        description="Minimum delay before marking message as read (seconds). Simulates time to read incoming messages.",
    )
    max_read_delay_seconds: float = Field(
        default=15.0,
        ge=0.0,
        description="Maximum delay before marking message as read (seconds). Prevents excessively long read delays.",
    )
    min_typing_delay_seconds: float = Field(
        default=3.0,
        ge=0.0,
        description="Minimum delay before sending response (seconds). Simulates time to compose a response.",
    )
    max_typing_delay_seconds: float = Field(
        default=45.0,
        ge=0.0,
        description="Maximum delay before sending response (seconds). Prevents excessively long typing delays.",
    )
    min_send_delay_seconds: float = Field(
        default=0.5,
        ge=0.0,
        description="Minimum delay before message transmission (seconds). Simulates final message review time.",
    )
    max_send_delay_seconds: float = Field(
        default=4.0,
        ge=0.0,
        description="Maximum delay before message transmission (seconds). Prevents excessively long send delays.",
    )
    enable_delay_jitter: bool = Field(
        default=True,
        description="Enable random variation (±20%) in delay calculations to prevent detectable patterns and simulate natural human behavior variability",
    )
    show_typing_during_delay: bool = Field(
        default=True,
        description="Show typing indicator during typing delays to provide visual feedback to users while the bot is 'composing' a response",
    )
    batch_read_compression_factor: float = Field(
        default=0.7,
        ge=0.1,
        le=1.0,
        description="Compression factor (0.1-1.0) applied to batch read delays. Lower values simulate faster batch reading (e.g., 0.7 = 30% faster than reading individually)",
    )

    # === Backward Compatibility (Deprecated) ===
    # These are kept for backward compatibility but map to the simplified parameters
    @property
    def message_batch_delay_seconds(self) -> float:
        """Deprecated: Use batch_delay_seconds instead."""
        return self.batch_delay_seconds

    @message_batch_delay_seconds.setter
    def message_batch_delay_seconds(self, value: float) -> None:
        """Deprecated: Use batch_delay_seconds instead."""
        self.batch_delay_seconds = value

    @property
    def max_batch_wait_seconds(self) -> float:
        """Deprecated: Use max_batch_timeout_seconds instead."""
        return self.max_batch_timeout_seconds

    @max_batch_wait_seconds.setter
    def max_batch_wait_seconds(self, value: float) -> None:
        """Deprecated: Use max_batch_timeout_seconds instead."""
        self.max_batch_timeout_seconds = value

    # === Override Method ===

    def with_overrides(
        self,
        *,
        # Base configuration to start from
        base_config: "WhatsAppBotConfig | None" = None,
        # Core Bot Behavior
        typing_indicator: bool | None = None,
        typing_duration: int | None = None,
        auto_read_messages: bool | None = None,
        quote_messages: bool | None = None,
        session_timeout_minutes: int | None = None,
        max_message_length: int | None = None,
        max_split_messages: int | None = None,
        error_message: str | None = None,
        welcome_message: str | None = None,
        welcome_image_url: str | None = None,
        welcome_image_base64: str | None = None,
        # Message Batching
        enable_message_batching: bool | None = None,
        batch_delay_seconds: float | None = None,
        max_batch_size: int | None = None,
        max_batch_timeout_seconds: float | None = None,
        # Spam Protection
        spam_protection_enabled: bool | None = None,
        min_message_interval_seconds: float | None = None,
        max_messages_per_minute: int | None = None,
        rate_limit_cooldown_seconds: int | None = None,
        # Debug and Monitoring
        debug_mode: bool | None = None,
        track_response_times: bool | None = None,
        slow_response_threshold_seconds: float | None = None,
        # Error Handling
        retry_failed_messages: bool | None = None,
        max_retry_attempts: int | None = None,
        retry_delay_seconds: float | None = None,
        # Text-to-Speech
        speech_play_chance: float | None = None,
        speech_config: SpeechConfig | None = None,
        # Human-Like Delays
        enable_human_delays: bool | None = None,
        min_read_delay_seconds: float | None = None,
        max_read_delay_seconds: float | None = None,
        min_typing_delay_seconds: float | None = None,
        max_typing_delay_seconds: float | None = None,
        min_send_delay_seconds: float | None = None,
        max_send_delay_seconds: float | None = None,
        enable_delay_jitter: bool | None = None,
        show_typing_during_delay: bool | None = None,
        batch_read_compression_factor: float | None = None,
    ) -> "WhatsAppBotConfig":
        """
        Create a new configuration instance with specified parameters overridden.

        Args:
            base_config: Optional base configuration to start from. If provided,
                        values from this config will be used instead of the current instance.
            All other parameters are optional and correspond to the configuration fields.
            Only non-None parameters will override the base configuration.

        Returns:
            New WhatsAppBotConfig instance with overridden parameters.

        Examples:
            >>> # Override individual parameters from current config
            >>> base_config = WhatsAppBotConfig.production()
            >>> debug_config = base_config.with_overrides(
            ...     debug_mode=True,
            ...     typing_duration=1,
            ...     welcome_message="Debug mode enabled!"
            ... )

            >>> # Start from a different config and override parameters
            >>> prod_config = WhatsAppBotConfig.production()
            >>> dev_config = WhatsAppBotConfig.development()
            >>> hybrid_config = prod_config.with_overrides(
            ...     base_config=dev_config,
            ...     spam_protection_enabled=True,
            ...     max_messages_per_minute=50
            ... )

            >>> # Combine two configs
            >>> cs_config = WhatsAppBotConfig.customer_service()
            >>> hv_config = WhatsAppBotConfig.high_volume()
            >>> combined = cs_config.with_overrides(
            ...     base_config=hv_config,
            ...     welcome_message="Welcome to our high-volume support!"
            ... )

            >>> # Enable human-like delays with custom timing
            >>> prod_config = WhatsAppBotConfig.production()
            >>> natural_config = prod_config.with_overrides(
            ...     enable_human_delays=True,
            ...     min_read_delay_seconds=3.0,
            ...     max_read_delay_seconds=20.0,
            ...     min_typing_delay_seconds=5.0,
            ...     max_typing_delay_seconds=60.0
            ... )
        """
        # Determine starting configuration
        if base_config is not None:
            current_config = base_config.model_dump()
        else:
            current_config = self.model_dump()

        # Build overrides dict, only including non-None values
        overrides: Mapping[str, Any] = {}

        # Core Bot Behavior
        if typing_indicator is not None:
            overrides["typing_indicator"] = typing_indicator
        if typing_duration is not None:
            overrides["typing_duration"] = typing_duration
        if auto_read_messages is not None:
            overrides["auto_read_messages"] = auto_read_messages
        if quote_messages is not None:
            overrides["quote_messages"] = quote_messages
        if session_timeout_minutes is not None:
            overrides["session_timeout_minutes"] = session_timeout_minutes
        if max_message_length is not None:
            overrides["max_message_length"] = max_message_length
        if max_split_messages is not None:
            overrides["max_split_messages"] = max_split_messages
        if error_message is not None:
            overrides["error_message"] = error_message
        if welcome_message is not None:
            overrides["welcome_message"] = welcome_message
        if welcome_image_url is not None:
            overrides["welcome_image_url"] = welcome_image_url
        if welcome_image_base64 is not None:
            overrides["welcome_image_base64"] = welcome_image_base64

        # Message Batching
        if enable_message_batching is not None:
            overrides["enable_message_batching"] = enable_message_batching
        if batch_delay_seconds is not None:
            overrides["batch_delay_seconds"] = batch_delay_seconds
        if max_batch_size is not None:
            overrides["max_batch_size"] = max_batch_size
        if max_batch_timeout_seconds is not None:
            overrides["max_batch_timeout_seconds"] = max_batch_timeout_seconds

        # Spam Protection
        if spam_protection_enabled is not None:
            overrides["spam_protection_enabled"] = spam_protection_enabled
        if min_message_interval_seconds is not None:
            overrides["min_message_interval_seconds"] = min_message_interval_seconds
        if max_messages_per_minute is not None:
            overrides["max_messages_per_minute"] = max_messages_per_minute
        if rate_limit_cooldown_seconds is not None:
            overrides["rate_limit_cooldown_seconds"] = rate_limit_cooldown_seconds

        # Debug and Monitoring
        if debug_mode is not None:
            overrides["debug_mode"] = debug_mode
        if track_response_times is not None:
            overrides["track_response_times"] = track_response_times
        if slow_response_threshold_seconds is not None:
            overrides["slow_response_threshold_seconds"] = (
                slow_response_threshold_seconds
            )

        # Error Handling
        if retry_failed_messages is not None:
            overrides["retry_failed_messages"] = retry_failed_messages
        if max_retry_attempts is not None:
            overrides["max_retry_attempts"] = max_retry_attempts
        if retry_delay_seconds is not None:
            overrides["retry_delay_seconds"] = retry_delay_seconds

        # Text-to-Speech
        if speech_play_chance is not None:
            overrides["speech_play_chance"] = speech_play_chance
        if speech_config is not None:
            overrides["speech_config"] = speech_config

        # Human-Like Delays
        if enable_human_delays is not None:
            overrides["enable_human_delays"] = enable_human_delays
        if min_read_delay_seconds is not None:
            overrides["min_read_delay_seconds"] = min_read_delay_seconds
        if max_read_delay_seconds is not None:
            overrides["max_read_delay_seconds"] = max_read_delay_seconds
        if min_typing_delay_seconds is not None:
            overrides["min_typing_delay_seconds"] = min_typing_delay_seconds
        if max_typing_delay_seconds is not None:
            overrides["max_typing_delay_seconds"] = max_typing_delay_seconds
        if min_send_delay_seconds is not None:
            overrides["min_send_delay_seconds"] = min_send_delay_seconds
        if max_send_delay_seconds is not None:
            overrides["max_send_delay_seconds"] = max_send_delay_seconds
        if enable_delay_jitter is not None:
            overrides["enable_delay_jitter"] = enable_delay_jitter
        if show_typing_during_delay is not None:
            overrides["show_typing_during_delay"] = show_typing_during_delay
        if batch_read_compression_factor is not None:
            overrides["batch_read_compression_factor"] = batch_read_compression_factor

        # Update configuration with overrides
        current_config.update(overrides)

        # Create and return new instance
        return self.__class__(**current_config)

    # === Simplified Constructors ===

    @classmethod
    def development(
        cls,
        *,
        welcome_message: str | None = "Hello! I'm your development bot assistant.",
        quote_messages: bool = False,
        debug_mode: bool = True,
    ) -> "WhatsAppBotConfig":
        """
        Create a configuration optimized for development.

        Features:
        - Debug mode enabled for comprehensive logging
        - Faster response times for quick iteration
        - Lenient rate limiting (100 messages/minute)
        - Detailed logging and performance tracking
        - Human-like delays DISABLED for fast iteration and testing
        - Fast batching (1s delay, 5s timeout)

        Delay Settings:
        - enable_human_delays: False (disabled for development speed)

        Use this preset during development and testing when you need fast feedback
        and don't want to wait for realistic human-like delays.

        Example:
            >>> config = WhatsAppBotConfig.development()
            >>> bot = WhatsAppBot(agent=agent, provider=provider, config=config)
        """
        return cls(
            # Core behavior
            typing_indicator=True,
            typing_duration=1,
            auto_read_messages=True,
            quote_messages=quote_messages,
            welcome_message=welcome_message,
            # Fast batching for development
            enable_message_batching=True,
            batch_delay_seconds=1.0,
            max_batch_size=5,
            max_batch_timeout_seconds=5.0,
            # Lenient spam protection
            spam_protection_enabled=False,
            max_messages_per_minute=100,
            # Debug settings
            debug_mode=debug_mode,
            track_response_times=True,
            slow_response_threshold_seconds=5.0,
            # Error handling
            retry_failed_messages=True,
            max_retry_attempts=2,
            # Human-like delays (disabled for development)
            enable_human_delays=False,
        )

    @classmethod
    def production(
        cls,
        *,
        welcome_message: str | None = None,
        quote_messages: bool = False,
        enable_spam_protection: bool = True,
    ) -> WhatsAppBotConfig:
        """
        Create a configuration optimized for production.

        Features:
        - Robust spam protection (20 messages/minute limit)
        - Efficient batching (10s delay, 60s timeout)
        - Conservative rate limiting with 60s cooldown
        - Minimal debug output for performance
        - Human-like delays ENABLED with recommended baseline values
        - Typing indicators shown during delays for user feedback

        Delay Settings:
        - enable_human_delays: True
        - Read delays: 2.0s - 15.0s (simulates reading incoming messages)
        - Typing delays: 3.0s - 45.0s (simulates composing responses)
        - Send delays: 0.5s - 4.0s (simulates final review)
        - Jitter enabled: ±20% random variation
        - Typing indicator: Shown during typing delays
        - Batch compression: 0.7x (30% faster batch reading)

        These delay settings provide a good balance between natural behavior and
        reasonable response times. They help prevent platform detection while
        maintaining acceptable user experience.

        Example:
            >>> config = WhatsAppBotConfig.production()
            >>> bot = WhatsAppBot(agent=agent, provider=provider, config=config)
        """
        return cls(
            # Core behavior
            typing_indicator=True,
            typing_duration=2,
            auto_read_messages=True,
            quote_messages=quote_messages,
            welcome_message=welcome_message,
            # Efficient batching
            enable_message_batching=True,
            batch_delay_seconds=10.0,
            max_batch_size=10,
            max_batch_timeout_seconds=60,
            # Strong spam protection
            spam_protection_enabled=enable_spam_protection,
            max_messages_per_minute=20,
            rate_limit_cooldown_seconds=60,
            # Production settings
            debug_mode=False,
            track_response_times=True,
            slow_response_threshold_seconds=10.0,
            # Robust error handling
            retry_failed_messages=True,
            max_retry_attempts=3,
            retry_delay_seconds=1.0,
            # Human-like delays (enabled with baseline values)
            enable_human_delays=True,
            min_read_delay_seconds=2.0,
            max_read_delay_seconds=15.0,
            min_typing_delay_seconds=3.0,
            max_typing_delay_seconds=45.0,
            min_send_delay_seconds=0.5,
            max_send_delay_seconds=4.0,
            enable_delay_jitter=True,
            show_typing_during_delay=True,
            batch_read_compression_factor=0.7,
        )

    @classmethod
    def high_volume(
        cls,
        *,
        welcome_message: str | None = None,
        quote_messages: bool = False,
    ) -> "WhatsAppBotConfig":
        """
        Create a configuration optimized for high-volume scenarios.

        Features:
        - Aggressive batching (1s delay, 10s timeout, up to 20 messages)
        - Strong rate limiting (15 messages/minute, 120s cooldown)
        - Fast processing with minimal overhead
        - Typing indicators disabled for performance
        - Human-like delays ENABLED with balanced timing optimized for throughput

        Delay Settings:
        - enable_human_delays: True
        - Read delays: 1.5s - 12.0s (shorter for faster processing)
        - Typing delays: 2.5s - 35.0s (shorter for faster responses)
        - Send delays: 0.3s - 3.0s (shorter for faster transmission)
        - Jitter enabled: ±20% random variation
        - Typing indicator: DISABLED for performance
        - Batch compression: 0.7x (30% faster batch reading)

        These delay settings are optimized for high-volume scenarios where throughput
        is important but you still want to maintain natural behavior patterns to
        prevent platform detection. Delays are shorter than production but still
        provide realistic timing.

        Example:
            >>> config = WhatsAppBotConfig.high_volume()
            >>> bot = WhatsAppBot(agent=agent, provider=provider, config=config)
        """
        return cls(
            # Fast core behavior
            typing_indicator=False,  # Disabled for performance
            typing_duration=0,
            auto_read_messages=True,
            quote_messages=quote_messages,
            welcome_message=welcome_message,
            # Aggressive batching
            enable_message_batching=True,
            batch_delay_seconds=1.0,  # Fast batching
            max_batch_size=20,  # Larger batches
            max_batch_timeout_seconds=10.0,
            # Strong spam protection
            spam_protection_enabled=True,
            max_messages_per_minute=15,  # More restrictive
            rate_limit_cooldown_seconds=120,  # Longer cooldown
            # Performance settings
            debug_mode=False,
            track_response_times=False,  # Disabled for performance
            # Quick error handling
            retry_failed_messages=True,
            max_retry_attempts=2,  # Fewer retries
            retry_delay_seconds=0.5,  # Faster retries
            # Human-like delays (enabled with balanced timing)
            enable_human_delays=True,
            min_read_delay_seconds=1.5,
            max_read_delay_seconds=12.0,
            min_typing_delay_seconds=2.5,
            max_typing_delay_seconds=35.0,
            min_send_delay_seconds=0.3,
            max_send_delay_seconds=3.0,
            enable_delay_jitter=True,
            show_typing_during_delay=False,  # Disabled for performance
            batch_read_compression_factor=0.7,
        )

    @classmethod
    def customer_service(
        cls,
        *,
        welcome_message: str = "Hello! How can I help you today?",
        quote_messages: bool = True,  # Enabled for context
        support_hours_message: str | None = None,
    ) -> "WhatsAppBotConfig":
        """
        Create a configuration optimized for customer service.

        Features:
        - Message quoting enabled for conversation context
        - Moderate batching (5s delay, 20s timeout, up to 8 messages)
        - Professional response times with thoughtful delays
        - Welcome message for first-time users
        - Human-like delays ENABLED with thoughtful, professional timing
        - Typing indicators shown for premium user experience

        Delay Settings:
        - enable_human_delays: True
        - Read delays: 5.0s - 30.0s (longer for thoughtful reading)
        - Typing delays: 10.0s - 90.0s (longer for careful composition)
        - Send delays: 1.0s - 5.0s (longer for final review)
        - Jitter enabled: ±20% random variation
        - Typing indicator: Shown during typing delays
        - Batch compression: 0.7x (30% faster batch reading)

        These delay settings are optimized for customer service scenarios where
        users expect thoughtful, professional responses. Longer delays give the
        impression of careful consideration and attention to detail, which can
        improve perceived service quality.

        Example:
            >>> config = WhatsAppBotConfig.customer_service()
            >>> bot = WhatsAppBot(agent=agent, provider=provider, config=config)
        """
        return cls(
            # Professional behavior
            typing_indicator=True,
            typing_duration=3,  # Gives impression of thoughtful response
            auto_read_messages=True,
            quote_messages=quote_messages,
            welcome_message=welcome_message,
            # Moderate batching
            enable_message_batching=True,
            batch_delay_seconds=5.0,  # Allow time for complete thoughts
            max_batch_size=8,
            max_batch_timeout_seconds=20.0,
            # Moderate spam protection
            spam_protection_enabled=True,
            max_messages_per_minute=30,  # Allow for conversations
            rate_limit_cooldown_seconds=45,
            # Customer service settings
            debug_mode=False,
            track_response_times=True,
            slow_response_threshold_seconds=15.0,  # Higher tolerance
            # Reliable error handling
            retry_failed_messages=True,
            max_retry_attempts=3,
            retry_delay_seconds=2.0,
            # Custom error message for customer service
            error_message=support_hours_message
            or "I apologize for the inconvenience. Please try again, or contact our support team if the issue persists.",
            # Human-like delays (enabled with thoughtful timing)
            enable_human_delays=True,
            min_read_delay_seconds=5.0,
            max_read_delay_seconds=30.0,
            min_typing_delay_seconds=10.0,
            max_typing_delay_seconds=90.0,
            min_send_delay_seconds=1.0,
            max_send_delay_seconds=5.0,
            enable_delay_jitter=True,
            show_typing_during_delay=True,
            batch_read_compression_factor=0.7,
        )

    @classmethod
    def minimal(
        cls,
        *,
        quote_messages: bool = False,
    ) -> "WhatsAppBotConfig":
        """
        Create a minimal configuration with basic functionality.

        Features:
        - No batching for immediate processing
        - No spam protection
        - Immediate responses with no delays
        - Minimal overhead and resource usage
        - Human-like delays DISABLED for minimal configuration
        - No typing indicators
        - No retry logic

        Delay Settings:
        - enable_human_delays: False (disabled for minimal overhead)

        This preset provides the absolute minimum configuration with no delays,
        batching, or spam protection. Use this only when you need the fastest
        possible responses and don't care about natural behavior patterns or
        platform detection.

        Example:
            >>> config = WhatsAppBotConfig.minimal()
            >>> bot = WhatsAppBot(agent=agent, provider=provider, config=config)
        """
        return cls(
            # Basic behavior
            typing_indicator=False,
            auto_read_messages=True,
            quote_messages=quote_messages,
            welcome_message=None,
            # No batching
            enable_message_batching=False,
            # No spam protection
            spam_protection_enabled=False,
            # Minimal settings
            debug_mode=False,
            track_response_times=False,
            # Basic error handling
            retry_failed_messages=False,
            max_retry_attempts=1,
            # Human-like delays (disabled for minimal overhead)
            enable_human_delays=False,
        )

    def validate_config(self) -> list[str]:
        """
        Validate configuration and return list of warnings/issues.

        This method performs comprehensive validation of all configuration parameters,
        including timing conflicts, rate limiting settings, retry configuration, and
        human-like delay parameters.

        Delay Validation:
            When human-like delays are enabled, this method validates:
            - All minimum delay values are non-negative (>= 0.0)
            - All maximum delay values are >= their corresponding minimum values
            - Batch read compression factor is between 0.1 and 1.0

        Returns:
            List of validation messages (empty if configuration is valid).
            Each message describes a specific configuration issue or warning.

        Example:
            >>> config = WhatsAppBotConfig.production()
            >>> issues = config.validate_config()
            >>> if issues:
            ...     for issue in issues:
            ...         print(f"Warning: {issue}")
            ... else:
            ...     print("Configuration is valid")
        """
        issues = []

        # Check timing conflicts
        if self.enable_message_batching and self.typing_indicator:
            if self.typing_duration >= self.batch_delay_seconds:
                issues.append(
                    f"Typing duration ({self.typing_duration}s) >= batch delay ({self.batch_delay_seconds}s). "
                    + "This may cause confusing UX where typing indicator outlasts batch processing."
                )

        # Check batch timeout vs delay
        if self.max_batch_timeout_seconds <= self.batch_delay_seconds:
            issues.append(
                f"Max batch timeout ({self.max_batch_timeout_seconds}s) <= batch delay ({self.batch_delay_seconds}s). "
                + "Batch timeout should be significantly larger than delay."
            )

        # Check rate limiting
        if self.spam_protection_enabled:
            if self.max_messages_per_minute <= 0:
                issues.append(
                    "Max messages per minute must be positive when spam protection is enabled."
                )

            if self.max_messages_per_minute > 60:
                issues.append(
                    f"Max messages per minute ({self.max_messages_per_minute}) is very high. "
                    + "Consider if this provides effective spam protection."
                )

        # Check retry configuration
        if self.retry_failed_messages and self.max_retry_attempts <= 0:
            issues.append("Max retry attempts must be positive when retry is enabled.")

        # Check message length
        if self.max_message_length > 4096:
            issues.append(
                f"Max message length ({self.max_message_length}) exceeds WhatsApp limit (4096). "
                + "Messages will be truncated."
            )

        # Check human-like delay configuration
        if self.enable_human_delays:
            # Validate read delays
            if self.min_read_delay_seconds < 0.0:
                issues.append(
                    f"min_read_delay_seconds ({self.min_read_delay_seconds}) must be non-negative (>= 0.0)."
                )
            if self.max_read_delay_seconds < self.min_read_delay_seconds:
                issues.append(
                    f"max_read_delay_seconds ({self.max_read_delay_seconds}) must be >= min_read_delay_seconds ({self.min_read_delay_seconds})."
                )

            # Validate typing delays
            if self.min_typing_delay_seconds < 0.0:
                issues.append(
                    f"min_typing_delay_seconds ({self.min_typing_delay_seconds}) must be non-negative (>= 0.0)."
                )
            if self.max_typing_delay_seconds < self.min_typing_delay_seconds:
                issues.append(
                    f"max_typing_delay_seconds ({self.max_typing_delay_seconds}) must be >= min_typing_delay_seconds ({self.min_typing_delay_seconds})."
                )

            # Validate send delays
            if self.min_send_delay_seconds < 0.0:
                issues.append(
                    f"min_send_delay_seconds ({self.min_send_delay_seconds}) must be non-negative (>= 0.0)."
                )
            if self.max_send_delay_seconds < self.min_send_delay_seconds:
                issues.append(
                    f"max_send_delay_seconds ({self.max_send_delay_seconds}) must be >= min_send_delay_seconds ({self.min_send_delay_seconds})."
                )

            # Validate batch read compression factor
            if not (0.1 <= self.batch_read_compression_factor <= 1.0):
                issues.append(
                    f"batch_read_compression_factor ({self.batch_read_compression_factor}) must be between 0.1 and 1.0."
                )

        return issues

    def __str__(self) -> str:
        """Human-readable configuration summary."""
        batching_status = "enabled" if self.enable_message_batching else "disabled"
        spam_protection_status = (
            "enabled" if self.spam_protection_enabled else "disabled"
        )

        return (
            f"WhatsAppBotConfig("
            f"batching={batching_status}, "
            f"spam_protection={spam_protection_status}, "
            f"quote_messages={self.quote_messages}, "
            f"debug={self.debug_mode})"
        )
