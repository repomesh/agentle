from __future__ import annotations

import asyncio
import inspect
import logging
import re
from collections.abc import (
    Awaitable,
    Callable,
    MutableMapping,
    MutableSequence,
    Sequence,
)
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional, cast
from dataclasses import dataclass, field

import mistune

from rsb.coroutines.run_sync import run_sync
from rsb.models.base_model import BaseModel
from rsb.models.config_dict import ConfigDict
from rsb.models.field import Field
from rsb.models.private_attr import PrivateAttr

from agentle.agents.agent import Agent
from agentle.agents.agent_input import AgentInput
from agentle.agents.conversations.conversation_store import ConversationStore
from agentle.agents.whatsapp.models.data import Data
from agentle.agents.whatsapp.models.whatsapp_audio_message import WhatsAppAudioMessage
from agentle.agents.whatsapp.models.whatsapp_bot_config import WhatsAppBotConfig
from agentle.agents.whatsapp.models.whatsapp_document_message import (
    WhatsAppDocumentMessage,
)
from agentle.agents.whatsapp.models.whatsapp_image_message import WhatsAppImageMessage
from agentle.agents.whatsapp.models.whatsapp_media_message import WhatsAppMediaMessage
from agentle.agents.whatsapp.models.whatsapp_message import WhatsAppMessage
from agentle.agents.whatsapp.models.whatsapp_response_base import WhatsAppResponseBase
from agentle.agents.whatsapp.models.whatsapp_session import WhatsAppSession
from agentle.agents.whatsapp.models.whatsapp_text_message import WhatsAppTextMessage
from agentle.agents.whatsapp.models.whatsapp_video_message import WhatsAppVideoMessage
from agentle.agents.whatsapp.models.whatsapp_webhook_payload import (
    WhatsAppWebhookPayload,
)
from agentle.agents.whatsapp.providers.base.whatsapp_provider import WhatsAppProvider
from agentle.agents.whatsapp.providers.evolution.evolution_api_provider import (
    EvolutionAPIProvider,
)
from agentle.agents.whatsapp.human_delay_calculator import HumanDelayCalculator
from agentle.generations.models.message_parts.file import FilePart
from agentle.generations.models.message_parts.text import TextPart
from agentle.generations.models.message_parts.tool_execution_suggestion import (
    ToolExecutionSuggestion,
)
from agentle.generations.models.messages.generated_assistant_message import (
    GeneratedAssistantMessage,
)
from agentle.generations.models.messages.user_message import UserMessage
from agentle.generations.tools.tool import Tool
from agentle.generations.tools.tool_execution_result import ToolExecutionResult
from agentle.storage.file_storage_manager import FileStorageManager
from agentle.tts.tts_provider import TtsProvider

if TYPE_CHECKING:
    from blacksheep import Application
    from blacksheep.server.openapi.v3 import OpenAPIHandler
    from blacksheep.server.routing import MountRegistry, Router
    from rodi import ContainerProtocol

try:
    import blacksheep
except ImportError:
    pass

# Type aliases for cleaner type hints
PhoneNumber = str  # WhatsApp phone number (e.g., "5511999999999")
ChatId = str  # Chat/conversation identifier

CallbackFunction = (
    Callable[
        [
            PhoneNumber,
            ChatId | None,
            GeneratedAssistantMessage[Any] | None,
            dict[str, Any] | None,
        ],
        None,
    ]
    | Callable[
        [
            PhoneNumber,
            ChatId | None,
            GeneratedAssistantMessage[Any] | None,
            dict[str, Any] | None,
        ],
        Awaitable[None],
    ]
)

CallbackInput = CallbackFunction | list[CallbackFunction] | None

logger = logging.getLogger(__name__)


@dataclass
class CallbackWithContext:
    """Container for callback function with optional context."""

    # Callbacks must accept (phone_number, chat_id, response, context) now
    callback: (
        Callable[
            [
                PhoneNumber,
                ChatId | None,
                GeneratedAssistantMessage[Any] | None,
                dict[str, Any],
            ],
            Awaitable[None],
        ]
        | Callable[
            [
                PhoneNumber,
                ChatId | None,
                GeneratedAssistantMessage[Any] | None,
                dict[str, Any],
            ],
            None,
        ]
    )

    context: dict[str, Any] = field(default_factory=dict)
    scope_key: str | None = None
    persistent: bool = True


@dataclass(frozen=True)
class QueuedMessageResult:
    """Represents a webhook/message accepted for asynchronous batch processing."""

    phone_number: PhoneNumber
    chat_id: ChatId | None = None
    pending_messages: int = 0
    processing_token: str | None = None
    status: str = "queued"
    reason: str = "message_batched"


type MessageHandlingResult = GeneratedAssistantMessage[Any] | QueuedMessageResult | None


class WhatsAppBot[T_Schema: WhatsAppResponseBase = WhatsAppResponseBase](BaseModel):
    """
    WhatsApp bot that wraps an Agentle agent with enhanced message batching and spam protection.

    Now supports structured outputs through generic type parameter T_Schema.
    The schema must extend WhatsAppResponseBase to ensure a 'response' field is always present.

    Examples:
    ```python
        # Basic usage (no structured output)
        agent = Agent(...)
        bot = WhatsAppBot(agent=agent, provider=provider)

        # With structured output
        class MyResponse(WhatsAppResponseBase):
            sentiment: Literal["happy", "sad", "neutral"]
            urgency_level: int

        agent = Agent[MyResponse](
            response_schema=MyResponse,
            instructions="Extract sentiment and urgency from the conversation..."
        )
        bot = WhatsAppBot[MyResponse](agent=agent, provider=provider)

        # Access structured data in callbacks
        async def my_callback(phone, chat_id, response, context):
            if response and response.parsed:
                print(f"Sentiment: {response.parsed.sentiment}")
                print(f"Urgency: {response.parsed.urgency_level}")
                # response.parsed.response is automatically sent to WhatsApp

        bot.add_response_callback(my_callback)
    ```
    """

    agent: Agent[T_Schema]
    provider: WhatsAppProvider
    tts_provider: TtsProvider | None = Field(default=None)
    file_storage_manager: FileStorageManager | None = Field(default=None)
    config: WhatsAppBotConfig = Field(default_factory=WhatsAppBotConfig)

    _running: bool = PrivateAttr(default=False)
    _webhook_handlers: MutableSequence[Callable[..., Any]] = PrivateAttr(
        default_factory=list
    )
    _batch_processors: MutableMapping[str, asyncio.Task[Any]] = PrivateAttr(
        default_factory=dict
    )
    _processing_locks: MutableMapping[str, asyncio.Lock] = PrivateAttr(
        default_factory=dict
    )
    _cleanup_task: Optional[asyncio.Task[Any]] = PrivateAttr(default=None)
    _response_callbacks: MutableSequence[CallbackWithContext] = PrivateAttr(
        default_factory=list
    )
    _delay_calculator: HumanDelayCalculator | None = PrivateAttr(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __post_init__(self):
        """Validate that agent has conversation store configured."""
        if self.agent.conversation_store is None:
            raise ValueError(
                "Agent must have a conversation_store configured for WhatsApp integration. "
                + "Please set agent.conversation_store before creating WhatsAppBot."
            )

        # Log configuration validation
        validation_issues = self.config.validate_config()
        if validation_issues:
            logger.warning(
                f"[CONFIG_VALIDATION] Configuration has {len(validation_issues)} validation issue(s):"
            )
            for issue in validation_issues:
                logger.warning(f"[CONFIG_VALIDATION]   - {issue}")
        else:
            logger.info("[CONFIG_VALIDATION] Configuration validation passed")

        # Initialize delay calculator if human delays are enabled
        if self.config.enable_human_delays:
            logger.info(
                "[DELAY_CONFIG] ═══════════ HUMAN-LIKE DELAYS ENABLED ═══════════"
            )
            logger.info(
                "[DELAY_CONFIG] Read delay bounds: "
                + f"[{self.config.min_read_delay_seconds:.2f}s - {self.config.max_read_delay_seconds:.2f}s]"
            )
            logger.info(
                "[DELAY_CONFIG] Typing delay bounds: "
                + f"[{self.config.min_typing_delay_seconds:.2f}s - {self.config.max_typing_delay_seconds:.2f}s]"
            )
            logger.info(
                "[DELAY_CONFIG] Send delay bounds: "
                + f"[{self.config.min_send_delay_seconds:.2f}s - {self.config.max_send_delay_seconds:.2f}s]"
            )
            logger.info(
                "[DELAY_CONFIG] Delay behavior settings: "
                + f"jitter_enabled={self.config.enable_delay_jitter}, "
                + f"show_typing={self.config.show_typing_during_delay}, "
                + f"batch_compression={self.config.batch_read_compression_factor:.2f}"
            )

            # Initialize delay calculator
            self._delay_calculator = HumanDelayCalculator(self.config)
            logger.info("[DELAY_CONFIG] Delay calculator initialized successfully")
            logger.info(
                "[DELAY_CONFIG] ═══════════════════════════════════════════════"
            )
        else:
            logger.info(
                "[DELAY_CONFIG] Human-like delays disabled (enable_human_delays=False)"
            )
            logger.debug(
                "[DELAY_CONFIG] To enable delays, set enable_human_delays=True in WhatsAppBotConfig"
            )

    @staticmethod
    def _build_callback_scope_key(
        *,
        chat_id: ChatId | None = None,
        phone_number: PhoneNumber | None = None,
    ) -> str | None:
        if chat_id:
            return f"chat:{chat_id}"
        if phone_number:
            return f"phone:{phone_number}"
        return None

    @staticmethod
    def _describe_message_handling_result(response: MessageHandlingResult) -> str:
        if isinstance(response, QueuedMessageResult):
            return response.status
        if response is not None:
            return "completed"
        return "no_response"

    def _register_response_callback(
        self,
        *,
        callback: CallbackFunction,
        context: dict[str, Any] | None,
        allow_duplicates: bool,
        scope_key: str | None,
        persistent: bool,
    ) -> bool:
        normalized_context = dict(context or {})
        callback_with_context = CallbackWithContext(
            callback=callback,
            context=normalized_context,
            scope_key=scope_key,
            persistent=persistent,
        )

        if not allow_duplicates:
            callback_exists = any(
                existing.callback == callback
                and existing.context == normalized_context
                and existing.scope_key == scope_key
                and existing.persistent == persistent
                for existing in self._response_callbacks
            )
            if callback_exists:
                logger.warning(
                    "[CALLBACKS] ⚠️ Duplicate callback registration prevented for %s (scope=%s)",
                    callback.__name__ if hasattr(callback, "__name__") else "unnamed",
                    scope_key or "<global>",
                )
                return False

        self._response_callbacks.append(callback_with_context)
        return True

    def start(self) -> None:
        """Start the WhatsApp bot."""
        run_sync(self.start_async)

    def stop(self) -> None:
        """Stop the WhatsApp bot."""
        run_sync(self.stop_async)

    def change_instance(self, instance_name: str) -> None:
        """Change the instance of the WhatsApp bot."""
        provider = self.provider
        if isinstance(provider, EvolutionAPIProvider):
            provider.change_instance(instance_name)

    async def start_async(self) -> None:
        """Start the WhatsApp bot with proper initialization."""
        await self.provider.initialize()
        self._running = True

        # Start cleanup task for abandoned batch processors
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        logger.info("WhatsApp bot started with message batching enabled")

    async def stop_async(self) -> None:
        """Stop the WhatsApp bot with proper cleanup."""
        self._running = False

        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

        # Cancel all batch processors
        for phone_number, task in self._batch_processors.items():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                logger.debug(f"Cancelled batch processor for {phone_number}")

        self._batch_processors.clear()
        self._processing_locks.clear()

        await self.provider.shutdown()
        # REMOVED: context_manager.close() - no longer needed
        logger.info("WhatsApp bot stopped")

    async def handle_message(
        self, message: WhatsAppMessage, chat_id: ChatId | None = None
    ) -> MessageHandlingResult:
        """
        Handle incoming WhatsApp message with enhanced error handling and batching.

        This is the main entry point for processing incoming WhatsApp messages. It handles
        rate limiting, spam protection, message batching, and applies human-like delays
        to simulate realistic behavior patterns.

        Message Processing Flow:
            1. Retrieve or create user session
            2. Check rate limiting (if spam protection enabled)
            3. Apply read delay (if human delays enabled) - simulates reading time
            4. Mark message as read (if auto_read_messages enabled)
            5. Send welcome message (if first interaction)
            6. Process message (with batching if enabled) or immediately
            7. Return generated response

        Human-Like Delays:
            When enable_human_delays is True, this method applies a read delay before
            marking the message as read. The delay simulates the time a human would take
            to read and comprehend the incoming message, creating a realistic gap between
            message receipt and read receipt.

            For batched messages, a batch read delay is applied instead, which accounts
            for reading multiple messages in sequence with compression for faster batch
            reading.

        Args:
            message: The incoming WhatsApp message to process.
            chat_id: Optional custom chat identifier for conversation tracking.
                    If not provided, uses the sender's phone number.

        Returns:
            Generated assistant response message, or None if processing failed or
            was rate limited.

        Raises:
            Exceptions are caught and logged. User-facing errors trigger error messages.

        Example:
            >>> message = WhatsAppTextMessage(
            ...     from_number="1234567890",
            ...     text="Hello!",
            ...     id="msg_123"
            ... )
            >>> response = await bot.handle_message(message)
            >>> if response:
            ...     print(f"Response: {response.text}")
        """

        logger.info("[MESSAGE_HANDLER] ═══════════ MESSAGE HANDLER ENTRY ═══════════")
        logger.info(
            f"[MESSAGE_HANDLER] Received message from {message.from_number}: ID={message.id}, Type={type(message).__name__}"
        )
        # ADICIONE ESTE LOG:
        logger.info(f"[MESSAGE_HANDLER] Chat ID recebido: {chat_id}")
        logger.info(
            f"[MESSAGE_HANDLER] Current response callbacks count: {len(self._response_callbacks)}"
        )

        # Log callback details
        for i, cb in enumerate(self._response_callbacks):
            logger.info(
                f"[MESSAGE_HANDLER] Callback {i + 1}: {cb.callback.__name__ if hasattr(cb.callback, '__name__') else 'unnamed'}, Context: {cb.context}"
            )

        try:
            # Get or create session FIRST to check rate limiting
            logger.debug(f"[MESSAGE_HANDLER] Getting session for {message.from_number}")
            session = await self.provider.get_session(message.from_number)
            if not session:
                logger.error(
                    f"[MESSAGE_HANDLER] ❌ Failed to get session for {message.from_number}"
                )
                return

            # CRITICAL FIX: Check rate limiting BEFORE any message processing
            if self.config.spam_protection_enabled:
                logger.debug(
                    f"[SPAM_PROTECTION] Checking rate limits for {message.from_number}"
                )
                can_process = session.update_rate_limiting(
                    self.config.max_messages_per_minute,
                    self.config.rate_limit_cooldown_seconds,
                )

                if not can_process:
                    logger.warning(
                        f"[SPAM_PROTECTION] ❌ Rate limited user {message.from_number} - BLOCKING message processing"
                    )
                    if session.is_rate_limited:
                        await self._send_rate_limit_message(message.from_number)
                    # CRITICAL: Update session to persist rate limiting state and return immediately
                    await self.provider.update_session(session)
                    return None

            # Apply read delay before marking message as read (simulates human reading time)
            await self._apply_read_delay(message)

            # Mark as read if configured (only after rate limiting check passes)
            if self.config.auto_read_messages:
                logger.debug(f"[MESSAGE_HANDLER] Marking message {message.id} as read")
                await self.provider.mark_message_as_read(message.id)

            # Store custom chat_id in session
            if chat_id:
                session.context_data["custom_chat_id"] = chat_id
                logger.info(
                    f"[MESSAGE_HANDLER] Stored custom chat_id in session: {chat_id}"
                )

            # Store remoteJid for later use when sending messages
            if message.remote_jid:
                session.context_data["remote_jid"] = message.remote_jid
                logger.info(
                    f"[MESSAGE_HANDLER] 🔑 Stored remote_jid in session: {message.remote_jid} for phone: {message.from_number}"
                )

            logger.info(
                f"[SESSION_STATE] Session for {message.from_number}: is_processing={session.is_processing}, pending_messages={len(session.pending_messages)}, message_count={session.message_count}"
            )

            effective_chat_id = chat_id or message.from_number
            logger.info(
                f"[MESSAGE_HANDLER] Effective chat_id para conversação: {effective_chat_id}"
            )

            # Check welcome message/image for first interaction
            if (
                await cast(
                    ConversationStore, self.agent.conversation_store
                ).get_conversation_history_length(effective_chat_id)
                == 0
                and (
                    self.config.welcome_message
                    or self.config.welcome_image_url
                    or self.config.welcome_image_base64
                )
            ):
                logger.info(
                    f"[WELCOME] Sending welcome message/image to {message.from_number}"
                )

                # Determine if we're sending an image or text
                has_welcome_image = (
                    self.config.welcome_image_url or self.config.welcome_image_base64
                )

                if has_welcome_image:
                    # Prepare caption from welcome_message if available
                    caption = None
                    if self.config.welcome_message:
                        caption = self._format_whatsapp_markdown(
                            self.config.welcome_message
                        )

                    # Handle welcome image via URL
                    if self.config.welcome_image_url:
                        logger.info(
                            f"[WELCOME] Sending welcome image from URL to {message.from_number}"
                        )
                        await self.provider.send_media_message(
                            to=message.from_number,
                            media_url=self.config.welcome_image_url,
                            media_type="image",
                            caption=caption,
                        )
                    # Handle welcome image via base64
                    elif self.config.welcome_image_base64:
                        # Try to upload base64 to file storage if available
                        image_url = None
                        if self.file_storage_manager:
                            try:
                                import base64
                                import time

                                logger.info(
                                    "[WELCOME] Uploading base64 welcome image to file storage"
                                )

                                # Decode base64 to bytes
                                image_bytes = base64.b64decode(
                                    self.config.welcome_image_base64
                                )

                                # Generate unique filename
                                timestamp = int(time.time())
                                filename = f"welcome_image_{timestamp}.jpg"

                                # Upload to storage
                                image_url = await self.file_storage_manager.upload_file(
                                    file_data=image_bytes,
                                    filename=filename,
                                    mime_type="image/jpeg",
                                )

                                logger.info(
                                    f"[WELCOME] Welcome image uploaded to storage: {image_url}"
                                )
                            except Exception as e:
                                logger.warning(
                                    f"[WELCOME] Failed to upload welcome image to storage: {e}"
                                )
                                image_url = None

                        # Send image via URL if upload succeeded, otherwise send text fallback
                        if image_url:
                            logger.info(
                                f"[WELCOME] Sending uploaded welcome image to {message.from_number}"
                            )
                            await self.provider.send_media_message(
                                to=message.from_number,
                                media_url=image_url,
                                media_type="image",
                                caption=caption,
                            )
                        else:
                            # Fallback to text if base64 upload failed and no URL available
                            logger.warning(
                                "[WELCOME] Could not send welcome image (base64 upload failed and no URL), falling back to text"
                            )
                            if caption:
                                await self.provider.send_text_message(
                                    message.from_number, caption
                                )
                            else:
                                logger.warning(
                                    "[WELCOME] No caption available, skipping welcome message"
                                )
                else:
                    # Send text-only welcome message (backward compatible)
                    formatted_welcome = self._format_whatsapp_markdown(
                        self.config.welcome_message
                    )
                    await self.provider.send_text_message(
                        message.from_number, formatted_welcome
                    )

                session.message_count += 1
                await self.provider.update_session(session)

                # Get updated session after welcome message
                updated_session = await self.provider.get_session(message.from_number)
                if updated_session:
                    session = updated_session
                else:
                    logger.warning(
                        f"[WELCOME] Could not retrieve updated session for {message.from_number}"
                    )

            # Handle message based on batching configuration
            response = None
            if self.config.enable_message_batching:
                logger.info(
                    f"[BATCHING] 📦 Processing message with batching for {message.from_number}"
                )
                logger.info(
                    f"[BATCHING] Batching config: delay={self.config.batch_delay_seconds}s, max_size={self.config.max_batch_size}, max_timeout={self.config.max_batch_timeout_seconds}s"
                )
                response = await self._handle_message_with_batching(
                    message, session, chat_id=effective_chat_id
                )
            else:
                logger.info(
                    f"[IMMEDIATE] ⚡ Processing message immediately for {message.from_number}"
                )
                response = await self._process_single_message(
                    message, session, chat_id=effective_chat_id
                )

            logger.info(
                "[MESSAGE_HANDLER] ✅ Message handling completed. Result status: %s",
                self._describe_message_handling_result(response),
            )
            return response

        except Exception as e:
            logger.error(
                f"[MESSAGE_HANDLER_ERROR] ❌ Error handling message from {message.from_number}: {e}",
                exc_info=True,
            )
            if self._is_user_facing_error(e):
                await self._send_error_message(message.from_number, message.id)
            return None
        finally:
            logger.info(
                "[MESSAGE_HANDLER] ═══════════ MESSAGE HANDLER EXIT ═══════════"
            )

    async def _cleanup_loop(self) -> None:
        """Background task to clean up abandoned batch processors."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._cleanup_abandoned_processors()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")

    async def handle_webhook(
        self,
        payload: WhatsAppWebhookPayload,
        callback: CallbackInput = None,
        callback_context: dict[str, Any] | None = None,
        chat_id: ChatId | None = None,
    ) -> MessageHandlingResult:
        """
        Handle incoming webhook from WhatsApp.
        """
        logger.info("[WEBHOOK] ═══════════ WEBHOOK HANDLER ENTRY ═══════════")
        logger.info(f"[WEBHOOK] Received webhook event: {payload.event}")
        logger.info(f"[WEBHOOK] Callback provided: {callback is not None}")
        logger.info(
            f"[WEBHOOK] Callback context provided: {callback_context is not None}"
        )
        logger.info(
            f"[WEBHOOK] Current response callbacks count: {len(self._response_callbacks)}"
        )

        if callback:
            logger.info("[WEBHOOK] Processing scoped callback registration...")

            # Handle both single callback and list of callbacks
            callbacks_to_register: list[CallbackFunction]
            if isinstance(callback, list):
                callbacks_to_register = callback
            else:
                callbacks_to_register = [callback]

            callback_scope_key = self._build_callback_scope_key(
                chat_id=chat_id,
                phone_number=getattr(payload, "phone_number_id", None),
            )

            logger.info(
                "[WEBHOOK] Registering %s callback(s) for scope=%s",
                len(callbacks_to_register),
                callback_scope_key or "<global>",
            )

            for i, cb in enumerate(callbacks_to_register):
                added = self._register_response_callback(
                    callback=cb,
                    context=callback_context,
                    allow_duplicates=False,
                    scope_key=callback_scope_key,
                    persistent=False,
                )

                if added:
                    logger.info(
                        f"[WEBHOOK] ✅ Added callback {i + 1} for deferred processing. Total callbacks: {len(self._response_callbacks)}"
                    )
                else:
                    logger.warning(f"[WEBHOOK] ⚠️ Duplicate callback {i + 1} not added")

            logger.info(
                "[WEBHOOK] Callback(s) scoped to this webhook will be removed after completion or failure"
            )

        try:
            logger.info("[WEBHOOK] Starting webhook validation...")
            await self.provider.validate_webhook(payload)
            logger.info("[WEBHOOK] ✅ Webhook validation passed")

            response = None

            # Handle Evolution API events
            if payload.event == "messages.upsert":
                logger.info("[WEBHOOK] 🔄 Handling messages.upsert event")
                response = await self._handle_message_upsert(payload, chat_id=chat_id)
                logger.info(
                    "[WEBHOOK] Message upsert result status: %s",
                    self._describe_message_handling_result(response),
                )
            elif payload.event == "messages.update":
                logger.info("[WEBHOOK] 🔄 Handling messages.update event")
                await self._handle_message_update(payload)
            elif payload.event == "connection.update":
                logger.info("[WEBHOOK] 🔄 Handling connection.update event")
                await self._handle_connection_update(payload)
            # Handle Meta API events
            elif payload.entry:
                logger.info("[WEBHOOK] 🔄 Handling Meta API webhook")
                response = await self._handle_meta_webhook(payload)
                logger.info(
                    "[WEBHOOK] Meta webhook result status: %s",
                    self._describe_message_handling_result(response),
                )

            # Call custom handlers
            logger.info(
                f"[WEBHOOK] Calling {len(self._webhook_handlers)} custom webhook handlers"
            )
            for i, handler in enumerate(self._webhook_handlers):
                logger.debug(
                    f"[WEBHOOK] Calling custom webhook handler {i + 1}/{len(self._webhook_handlers)}"
                )
                await handler(payload)

            logger.info(
                "[WEBHOOK] ✅ Webhook processing completed. Result status: %s",
                self._describe_message_handling_result(response),
            )
            return response

        except Exception as e:
            logger.error(
                f"[WEBHOOK_ERROR] ❌ Error handling webhook: {e}", exc_info=True
            )
            return None
        finally:
            logger.info(
                "[WEBHOOK] ℹ️  Scoped callbacks remain registered only until their correlated processing finishes"
            )
            logger.info(
                f"[WEBHOOK] Current callbacks after webhook: {len(self._response_callbacks)}"
            )
            logger.info("[WEBHOOK] ═══════════ WEBHOOK HANDLER EXIT ═══════════")

    def to_blacksheep_app(
        self,
        *,
        router: "Router | None" = None,
        services: "ContainerProtocol | None" = None,
        show_error_details: bool = False,
        mount: "MountRegistry | None" = None,
        docs: "OpenAPIHandler | None" = None,
        webhook_path: str = "/webhook/whatsapp",
    ) -> "Application":
        """
        Convert the WhatsApp bot to a BlackSheep ASGI application.

        Args:
            router: Optional router to use
            services: Optional services container
            show_error_details: Whether to show error details in responses
            mount: Optional mount registry
            docs: Optional OpenAPI handler
            webhook_path: Path for the webhook endpoint

        Returns:
            BlackSheep application with webhook endpoint
        """
        import blacksheep
        from blacksheep.server.openapi.ui import ScalarUIProvider
        from blacksheep.server.openapi.v3 import OpenAPIHandler
        from openapidocs.v3 import Info

        app = blacksheep.Application(
            router=router,
            services=services,
            show_error_details=show_error_details,
            mount=mount,
        )

        if docs is None:
            docs = OpenAPIHandler(
                ui_path="/openapi",
                info=Info(title="Agentle WhatsApp Bot API", version="1.0.0"),
            )
            docs.ui_providers.append(ScalarUIProvider(ui_path="/docs"))

        docs.bind_app(app)

        @blacksheep.post(webhook_path)
        async def _(
            webhook_payload: blacksheep.FromJSON[WhatsAppWebhookPayload],
        ) -> blacksheep.Response:
            """
            Handle incoming WhatsApp webhooks.

            Args:
                webhook_payload: The webhook payload from WhatsApp

            Returns:
                Success response
            """
            try:
                # Process the webhook payload
                payload_data: WhatsAppWebhookPayload = webhook_payload.value
                logger.info(
                    f"[WEBHOOK_ENDPOINT] Received webhook payload: {payload_data.event}"
                )
                await self.handle_webhook(payload_data)

                # Return success response
                return blacksheep.json(
                    {"status": "success", "message": "Webhook processed"}
                )

            except Exception as e:
                logger.error(
                    f"[WEBHOOK_ENDPOINT_ERROR] Webhook processing error: {e}",
                    exc_info=True,
                )
                return blacksheep.json(
                    {"status": "error", "message": "Failed to process webhook"},
                    status=500,
                )

        @app.on_start
        async def _() -> None:
            await self.start_async()

        return app

    def add_webhook_handler(self, handler: Callable[..., Any]) -> None:
        """Add custom webhook handler."""
        self._webhook_handlers.append(handler)

    def add_response_callback(
        self,
        callback: (
            Callable[
                [
                    PhoneNumber,
                    ChatId | None,
                    GeneratedAssistantMessage[Any] | None,
                    dict[str, Any],
                ],
                Awaitable[None],
            ]
            | Callable[
                [
                    PhoneNumber,
                    ChatId | None,
                    GeneratedAssistantMessage[Any] | None,
                    dict[str, Any],
                ],
                None,
            ]
        ),
        context: dict[str, Any] | None = None,
        allow_duplicates: bool = False,
        *,
        chat_id: ChatId | None = None,
        phone_number: PhoneNumber | None = None,
        persistent: bool = True,
    ) -> None:
        """Add callback to be called when a response is generated."""
        logger.info("[ADD_CALLBACK] ═══════════ ADDING RESPONSE CALLBACK ═══════════")
        logger.info(
            f"[ADD_CALLBACK] Callback function: {callback.__name__ if hasattr(callback, '__name__') else 'unnamed'}"
        )
        logger.info(f"[ADD_CALLBACK] Context provided: {context is not None}")
        logger.info(f"[ADD_CALLBACK] Allow duplicates: {allow_duplicates}")
        logger.info(f"[ADD_CALLBACK] Persistent: {persistent}")
        logger.info(
            f"[ADD_CALLBACK] Current callbacks count: {len(self._response_callbacks)}"
        )

        scope_key = self._build_callback_scope_key(
            chat_id=chat_id,
            phone_number=phone_number,
        )

        added = self._register_response_callback(
            callback=callback,
            context=context,
            allow_duplicates=allow_duplicates,
            scope_key=scope_key,
            persistent=persistent,
        )
        if added:
            logger.info(
                f"[ADD_CALLBACK] ✅ Callback added successfully. Total callbacks: {len(self._response_callbacks)}"
            )
        logger.info("[ADD_CALLBACK] ═══════════ CALLBACK ADDED ═══════════")

    def remove_response_callback(
        self,
        callback: (
            Callable[
                [
                    PhoneNumber,
                    ChatId | None,
                    GeneratedAssistantMessage[Any] | None,
                    dict[str, Any],
                ],
                Awaitable[None],
            ]
            | Callable[
                [
                    PhoneNumber,
                    ChatId | None,
                    GeneratedAssistantMessage[Any] | None,
                    dict[str, Any],
                ],
                None,
            ]
        ),
        context: dict[str, Any] | None = None,
        *,
        chat_id: ChatId | None = None,
        phone_number: PhoneNumber | None = None,
    ) -> bool:
        """Remove a specific callback from the registered callbacks."""
        logger.info(
            "[REMOVE_CALLBACK] ═══════════ REMOVING RESPONSE CALLBACK ═══════════"
        )
        logger.info(
            f"[REMOVE_CALLBACK] Callback function: {callback.__name__ if hasattr(callback, '__name__') else 'unnamed'}"
        )
        logger.info(f"[REMOVE_CALLBACK] Context provided: {context is not None}")
        logger.info(
            f"[REMOVE_CALLBACK] Current callbacks count: {len(self._response_callbacks)}"
        )

        scope_key = self._build_callback_scope_key(
            chat_id=chat_id,
            phone_number=phone_number,
        )

        normalized_context = dict(context or {})

        for i, existing in enumerate(self._response_callbacks):
            if (
                existing.callback == callback
                and existing.context == normalized_context
                and (scope_key is None or existing.scope_key == scope_key)
            ):
                self._response_callbacks.pop(i)
                logger.info(
                    f"[REMOVE_CALLBACK] ✅ Callback removed successfully. Remaining callbacks: {len(self._response_callbacks)}"
                )
                logger.info(
                    "[REMOVE_CALLBACK] ═══════════ CALLBACK REMOVED ═══════════"
                )
                return True

        logger.warning("[REMOVE_CALLBACK] ⚠️ Callback not found for removal")
        logger.info("[REMOVE_CALLBACK] ═══════════ CALLBACK NOT FOUND ═══════════")
        return False

    def clear_response_callbacks(self) -> int:
        """Remove all registered response callbacks."""
        logger.info(
            "[CLEAR_CALLBACKS] ═══════════ CLEARING ALL RESPONSE CALLBACKS ═══════════"
        )
        count = len(self._response_callbacks)
        logger.info(f"[CLEAR_CALLBACKS] Clearing {count} callbacks")
        self._response_callbacks.clear()
        logger.info("[CLEAR_CALLBACKS] ✅ All callbacks cleared")
        logger.info("[CLEAR_CALLBACKS] ═══════════ CALLBACKS CLEARED ═══════════")
        return count

    async def _cleanup_abandoned_processors(self) -> None:
        """Clean up batch processors that have been running too long, but protect active message sending."""
        abandoned_processors: MutableSequence[PhoneNumber] = []

        for phone_number, task in self._batch_processors.items():
            if task.done():
                abandoned_processors.append(phone_number)
                continue

            # Check if session is still processing
            session = await self.provider.get_session(phone_number)
            if not session:
                # No session found, abandon this processor
                abandoned_processors.append(phone_number)
                task.cancel()
                continue

            # CRITICAL: Don't abandon if currently sending messages
            is_sending_messages = session.context_data.get("is_sending_messages", False)

            if is_sending_messages:
                logger.info(
                    f"[CLEANUP] Protecting batch processor for {phone_number} - currently sending messages"
                )
                continue

            # Check if batch has been running too long
            if session.is_batch_expired(
                self.config.max_batch_timeout_seconds * 3
            ):  # Give more time
                logger.warning(
                    f"[CLEANUP] Found abandoned batch processor for {phone_number} (not sending messages)"
                )
                abandoned_processors.append(phone_number)
                task.cancel()

                # Reset session state
                session.reset_session()
                await self.provider.update_session(session)

        # Clean up abandoned processors
        for phone_number in abandoned_processors:
            if phone_number in self._batch_processors:
                del self._batch_processors[phone_number]
            if phone_number in self._processing_locks:
                del self._processing_locks[phone_number]

        if abandoned_processors:
            logger.info(
                f"Cleaned up {len(abandoned_processors)} abandoned batch processors"
            )

    async def _handle_message_with_batching(
        self,
        message: WhatsAppMessage,
        session: WhatsAppSession,
        chat_id: ChatId | None = None,
    ) -> GeneratedAssistantMessage[Any] | QueuedMessageResult | None:
        """Handle message with improved batching logic and atomic state management."""
        phone_number = message.from_number

        logger.info("[BATCHING] ═══════════ BATCH HANDLING START ═══════════")
        logger.info(f"[BATCHING] Phone: {phone_number}")
        logger.info(
            f"[BATCHING] Current session state: processing={session.is_processing}, pending={len(session.pending_messages)}"
        )
        logger.info(
            f"[BATCHING] Current response callbacks count: {len(self._response_callbacks)}"
        )

        if chat_id:
            session.context_data["custom_chat_id"] = chat_id
            logger.info(f"[BATCHING] ✅ Stored custom_chat_id in session: {chat_id}")
        else:
            logger.warning("[BATCHING] ⚠️ No chat_id provided to store in session")

        try:
            if phone_number not in self._processing_locks:
                logger.info(
                    f"[BATCHING] Creating new processing lock for {phone_number}"
                )
                self._processing_locks[phone_number] = asyncio.Lock()

            async with self._processing_locks[phone_number]:
                logger.info(f"[BATCHING] Acquired processing lock for {phone_number}")

                # Re-fetch session to ensure we have latest state
                current_session = await self.provider.get_session(phone_number)
                if not current_session:
                    logger.error(f"[BATCHING] ❌ Lost session for {phone_number}")
                    return None

                # CRITICAL FIX: Preserve custom_chat_id from original session
                original_chat_id = session.context_data.get("custom_chat_id")
                if original_chat_id and not current_session.context_data.get(
                    "custom_chat_id"
                ):
                    logger.warning(
                        f"[BATCHING] ⚠️ custom_chat_id lost during session re-fetch, restoring: {original_chat_id}"
                    )
                    current_session.context_data["custom_chat_id"] = original_chat_id
                    logger.info(
                        f"[BATCHING] ✅ Restored custom_chat_id: {original_chat_id}"
                    )
                elif original_chat_id:
                    logger.info(
                        f"[BATCHING] ✅ custom_chat_id preserved during re-fetch: {original_chat_id}"
                    )
                else:
                    logger.info("[BATCHING] No custom_chat_id to preserve")

                # Convert message to storable format
                message_data = await self._message_to_dict(message)
                logger.info(
                    f"[BATCHING] Converted message to dict: {message_data.get('id')}"
                )

                # Atomic session update with validation
                success = await self._atomic_session_update(
                    phone_number, current_session, message_data
                )

                if not success:
                    logger.error(
                        f"[BATCHING] ❌ Failed to update session for {phone_number}"
                    )
                    logger.info("[BATCHING] 🔄 Falling back to immediate processing")
                    return await self._process_single_message(message, current_session)

                # Re-fetch session after update to get latest processing state
                updated_session = await self.provider.get_session(phone_number)
                if not updated_session:
                    logger.error(
                        f"[BATCHING] ❌ Lost session after update for {phone_number}"
                    )
                    return None

                # CRITICAL FIX: Ensure custom_chat_id is preserved after update
                expected_chat_id = current_session.context_data.get("custom_chat_id")
                actual_chat_id = updated_session.context_data.get("custom_chat_id")
                if expected_chat_id and not actual_chat_id:
                    logger.error(
                        f"[BATCHING] ❌ custom_chat_id lost after session update! Expected: {expected_chat_id}, Got: {actual_chat_id}"
                    )
                    updated_session.context_data["custom_chat_id"] = expected_chat_id
                    await self.provider.update_session(updated_session)
                    logger.info(
                        f"[BATCHING] ✅ Restored custom_chat_id after update: {expected_chat_id}"
                    )
                elif expected_chat_id:
                    logger.info(
                        f"[BATCHING] ✅ custom_chat_id preserved after update: {expected_chat_id}"
                    )

                logger.info(
                    f"[BATCHING] Updated session state: processing={updated_session.is_processing}, token={updated_session.processing_token}, pending={len(updated_session.pending_messages)}"
                )

                # CRITICAL FIX: Enhanced race condition protection for batch processor creation
                should_create_processor = (
                    updated_session.is_processing
                    and updated_session.processing_token
                    and phone_number not in self._batch_processors
                )

                # Double-check to prevent race conditions
                if should_create_processor:
                    # Check again inside the lock to prevent duplicate processors
                    if phone_number in self._batch_processors:
                        existing_task = self._batch_processors[phone_number]
                        if not existing_task.done():
                            logger.info(
                                f"[BATCHING] ⚠️ Processor already exists for {phone_number}, skipping creation"
                            )
                            should_create_processor = False
                        else:
                            logger.info(
                                f"[BATCHING] 🧹 Cleaning up completed processor for {phone_number}"
                            )
                            del self._batch_processors[phone_number]

                if should_create_processor:
                    logger.info(
                        f"[BATCHING] 🚀 Starting new batch processor for {phone_number}"
                    )
                    logger.info(
                        f"[BATCHING] Processing token: {updated_session.processing_token}"
                    )
                    # Ensure processing_token is not None before passing to _batch_processor
                    if updated_session.processing_token:
                        self._batch_processors[phone_number] = asyncio.create_task(
                            self._batch_processor(
                                phone_number, updated_session.processing_token
                            )
                        )
                        logger.info(
                            f"[BATCHING] ✅ Batch processor task created for {phone_number}"
                        )
                    else:
                        logger.error(
                            f"[BATCHING] ❌ Cannot create processor: processing_token is None for {phone_number}"
                        )
                else:
                    logger.info(
                        f"[BATCHING] Message added to existing batch for {phone_number}"
                    )
                    logger.info(
                        f"[BATCHING] Existing processor active: {phone_number in self._batch_processors}"
                    )

                queued_result = QueuedMessageResult(
                    phone_number=phone_number,
                    chat_id=chat_id,
                    pending_messages=len(updated_session.pending_messages),
                    processing_token=updated_session.processing_token,
                )
                logger.info(
                    "[BATCHING] Returning queued result (pending_messages=%s, token=%s)",
                    queued_result.pending_messages,
                    queued_result.processing_token,
                )
                return queued_result

        except Exception as e:
            logger.error(
                f"[BATCHING_ERROR] ❌ Error in message batching for {phone_number}: {e}",
                exc_info=True,
            )
            # Always fall back to immediate processing on error
            try:
                logger.info("[BATCHING] 🔄 Attempting fallback to immediate processing")
                return await self._process_single_message(message, session)
            except Exception as fallback_error:
                logger.error(
                    f"[FALLBACK_ERROR] ❌ Fallback processing failed: {fallback_error}",
                    exc_info=True,
                )
                await self._send_error_message(message.from_number, message.id)
                return None
        finally:
            logger.info("[BATCHING] ═══════════ BATCH HANDLING END ═══════════")

    async def _atomic_session_update(
        self,
        phone_number: PhoneNumber,
        session: WhatsAppSession,
        message_data: dict[str, Any],
    ) -> bool:
        """Atomically update session with proper state transitions."""
        try:
            # Add message to pending queue
            session.add_pending_message(message_data)

            # If not currently processing, transition to processing state
            if not session.is_processing:
                processing_token = session.start_batch_processing(
                    self.config.max_batch_timeout_seconds
                )

                # Validate the state transition worked
                if not session.is_processing or not session.processing_token:
                    logger.error(
                        f"[ATOMIC_UPDATE] Failed to start processing for {phone_number}"
                    )
                    return False

                logger.info(
                    f"[ATOMIC_UPDATE] Started processing for {phone_number} with token {processing_token}"
                )

            # Log context_data before persisting
            logger.info(
                f"[ATOMIC_UPDATE] Context data before update: {session.context_data}"
            )

            # Persist the updated session
            await self.provider.update_session(session)

            # Verify the session was persisted correctly by re-reading
            verification_session = await self.provider.get_session(phone_number)
            if not verification_session:
                logger.error(
                    f"[ATOMIC_UPDATE] Session disappeared after update for {phone_number}"
                )
                return False

            # Log context_data after persisting
            logger.info(
                f"[ATOMIC_UPDATE] Context data after update: {verification_session.context_data}"
            )

            # Verify context_data is preserved
            if verification_session.context_data.get(
                "custom_chat_id"
            ) != session.context_data.get("custom_chat_id"):
                logger.error(
                    f"[ATOMIC_UPDATE] ❌ custom_chat_id not preserved! Before: {session.context_data.get('custom_chat_id')}, After: {verification_session.context_data.get('custom_chat_id')}"
                )
                return False

            # Verify critical state is preserved
            if verification_session.is_processing != session.is_processing:
                logger.error(
                    f"[ATOMIC_UPDATE] Processing state not persisted for {phone_number}"
                )
                return False

            if len(verification_session.pending_messages) != len(
                session.pending_messages
            ):
                logger.error(
                    f"[ATOMIC_UPDATE] Pending messages not persisted for {phone_number}"
                )
                return False

            return True

        except Exception as e:
            logger.error(
                f"[ATOMIC_UPDATE] Failed atomic session update for {phone_number}: {e}"
            )
            return False

    async def _batch_processor(
        self, phone_number: PhoneNumber, processing_token: str
    ) -> None:
        """
        Background task to process batched messages for a user with improved reliability.
        """
        logger.info("[BATCH_PROCESSOR] ═══════════ BATCH PROCESSOR START ═══════════")
        logger.info(
            f"[BATCH_PROCESSOR] Phone: {phone_number}, Token: {processing_token}"
        )
        logger.info(
            f"[BATCH_PROCESSOR] Current response callbacks count: {len(self._response_callbacks)}"
        )

        iteration_count = 0
        max_iterations = 1000  # Safety limit to prevent infinite loops
        batch_processed = False

        try:
            while (
                self._running
                and not batch_processed
                and iteration_count < max_iterations
            ):
                iteration_count += 1

                # Log early iterations for debugging
                if iteration_count <= 10:
                    logger.info(
                        f"[BATCH_PROCESSOR] 🔄 ENTERING iteration {iteration_count} for {phone_number}"
                    )

                try:
                    # Get current session
                    session = await self.provider.get_session(phone_number)
                    if not session:
                        logger.error(
                            f"[BATCH_PROCESSOR] ❌ No session found for {phone_number}, exiting at iteration {iteration_count}"
                        )
                        break

                    # Validate processing token
                    if session.processing_token != processing_token:
                        logger.warning(
                            f"[BATCH_PROCESSOR] ⚠️ Token mismatch for {phone_number}, exiting. Expected: {processing_token}, Got: {session.processing_token}"
                        )
                        break

                    if not session.is_processing:
                        logger.info(
                            f"[BATCH_PROCESSOR] ℹ️ Session no longer processing for {phone_number}, exiting at iteration {iteration_count}"
                        )
                        break

                    # Check for pending messages
                    if not session.pending_messages:
                        logger.warning(
                            f"[BATCH_PROCESSOR] ⚠️ No pending messages for {phone_number}, exiting at iteration {iteration_count}"
                        )
                        break

                    # Log session state for debugging
                    if iteration_count <= 20:
                        logger.info(
                            f"[BATCH_PROCESSOR] Session state for {phone_number}: pending_messages={len(session.pending_messages)}, batch_timeout_at={session.batch_timeout_at}, batch_started_at={session.batch_started_at}, iteration={iteration_count}"
                        )

                    # Check if batch should be processed
                    should_process = session.should_process_batch(
                        self.config.batch_delay_seconds,
                        self.config.max_batch_timeout_seconds,
                    )

                    # Check if max batch size reached
                    if len(session.pending_messages) >= self.config.max_batch_size:
                        logger.info(
                            f"[BATCH_PROCESSOR] 📏 Max batch size ({self.config.max_batch_size}) reached for {phone_number}, processing immediately"
                        )
                        should_process = True

                    # Check if batch has expired
                    if session.is_batch_expired(self.config.max_batch_timeout_seconds):
                        logger.info(
                            f"[BATCH_PROCESSOR] ⏰ Batch expired for {phone_number}, processing immediately"
                        )
                        should_process = True

                    # Log the decision for debugging
                    if iteration_count <= 20 or should_process:
                        logger.info(
                            f"[BATCH_PROCESSOR] Should process batch for {phone_number}: {should_process} (iteration {iteration_count}, messages={len(session.pending_messages)})"
                        )

                    if should_process:
                        logger.info(
                            f"[BATCH_PROCESSOR] 🚀 Batch ready for processing for {phone_number} (condition met after {iteration_count} iterations)"
                        )
                        logger.info(
                            f"[BATCH_PROCESSOR] About to call _process_message_batch with {len(self._response_callbacks)} callbacks registered"
                        )

                        await self._process_message_batch(
                            phone_number, session, processing_token
                        )

                        logger.info(
                            f"[BATCH_PROCESSOR] ✅ _process_message_batch completed for {phone_number}"
                        )
                        batch_processed = True
                        break

                except Exception as e:
                    logger.error(
                        f"[BATCH_PROCESSOR_ERROR] ❌ Error in batch processing loop for {phone_number}: {e}",
                        exc_info=True,
                    )
                    # Try to clean up the session state
                    try:
                        session = await self.provider.get_session(phone_number)
                        if session:
                            logger.debug(
                                f"[BATCH_PROCESSOR] 🧹 Cleaning up session state for {phone_number}"
                            )
                            session.finish_batch_processing(processing_token)
                            await self.provider.update_session(session)
                    except Exception as cleanup_error:
                        logger.error(
                            f"[BATCH_PROCESSOR] ❌ Failed to cleanup session for {phone_number}: {cleanup_error}"
                        )
                    break

                # Add delay between iterations
                await asyncio.sleep(0.1)  # Small polling interval

            # Log why we exited the loop
            logger.info(
                f"[BATCH_PROCESSOR] Loop exit for {phone_number}: self._running={self._running}, batch_processed={batch_processed}, iterations={iteration_count}, max_iterations={max_iterations}"
            )

        except asyncio.CancelledError:
            logger.info(
                f"[BATCH_PROCESSOR] ⚠️ Batch processor for {phone_number} was cancelled"
            )
            raise
        except Exception as e:
            logger.error(
                f"[BATCH_PROCESSOR_CRITICAL] ❌ Critical error in batch processor for {phone_number}: {e}",
                exc_info=True,
            )
        finally:
            # Clean up
            logger.info(
                f"[BATCH_PROCESSOR] 🧹 Cleaning up batch processor for {phone_number}"
            )
            if phone_number in self._batch_processors:
                del self._batch_processors[phone_number]
                logger.debug(
                    f"[BATCH_PROCESSOR] ✅ Removed batch processor task for {phone_number}"
                )

            # Ensure session is not left in processing state
            try:
                cleanup_session = await self.provider.get_session(phone_number)
                if cleanup_session and cleanup_session.is_processing:
                    logger.warning(
                        f"[BATCH_PROCESSOR] 🧹 Cleaning up processing state for {phone_number}"
                    )
                    cleanup_session.finish_batch_processing(processing_token)
                    await self.provider.update_session(cleanup_session)
            except Exception as cleanup_error:
                logger.error(
                    f"[BATCH_PROCESSOR] ❌ Final cleanup error for {phone_number}: {cleanup_error}"
                )

            logger.info("[BATCH_PROCESSOR] ═══════════ BATCH PROCESSOR END ═══════════")

    async def _process_message_batch(
        self, phone_number: PhoneNumber, session: WhatsAppSession, processing_token: str
    ) -> GeneratedAssistantMessage[T_Schema] | None:
        """Process a batch of messages for a user with enhanced timeout protection.

        This method processes multiple messages that were received in quick succession
        as a single batch. It applies batch-specific delays and combines all messages
        into a single conversation context for more coherent responses.

        Batch Processing Flow:
            1. Validate pending messages exist
            2. Mark session as sending to prevent cleanup
            3. Apply batch read delay (if human delays enabled) - simulates reading all messages
            4. Convert message batch to agent input
            5. Generate single response for entire batch
            6. Send response to user
            7. Mark all messages as read
            8. Update session state
            9. Execute response callbacks

        Human-Like Delays:
            When enable_human_delays is True, this method applies a batch read delay
            at the start of processing. The delay simulates the time a human would take
            to read multiple messages in sequence, accounting for:
            - Individual reading time for each message
            - Brief pauses between messages (0.5s each)
            - Compression factor (default 0.7x) for faster batch reading

            This creates a realistic gap before the batch is processed, making the bot
            appear more human-like when handling rapid message sequences.

        Args:
            phone_number: Phone number of the user whose messages are being processed.
            session: The user's WhatsApp session containing pending messages.
            processing_token: Unique token to prevent duplicate batch processing.

        Returns:
            Generated assistant response for the batch, or None if processing failed
            or no messages were pending.

        Raises:
            Exceptions are caught and logged. Session state is cleaned up on errors.

        Example:
            >>> # Called automatically by batch processor task
            >>> response = await self._process_message_batch(
            ...     phone_number="1234567890",
            ...     session=session,
            ...     processing_token="batch_123"
            ... )
        """
        logger.info("[BATCH_PROCESSING] ═══════════ BATCH PROCESSING START ═══════════")
        logger.info(
            f"[BATCH_PROCESSING] Phone: {phone_number}, Token: {processing_token}"
        )

        chat_id = session.context_data.get("custom_chat_id")
        logger.info(f"[BATCH_PROCESSING] Retrieved chat_id from session: {chat_id}")
        logger.info(
            f"[BATCH_PROCESSING] Session context_data keys: {list(session.context_data.keys())}"
        )

        # DEBUG: Log all context data for troubleshooting
        if chat_id is None:
            logger.warning(
                f"[BATCH_PROCESSING] ⚠️ custom_chat_id is None! Full context_data: {session.context_data}"
            )
        else:
            logger.info(f"[BATCH_PROCESSING] ✅ Using custom chat_id: {chat_id}")

        if not session.pending_messages:
            logger.warning(
                f"[BATCH_PROCESSING] ⚠️ No pending messages for {phone_number}, finishing batch processing"
            )
            session.finish_batch_processing(processing_token)
            await self.provider.update_session(session)
            return None

        try:
            # IMPORTANT: Mark session as "sending messages" to prevent cleanup during sending
            session.context_data["is_sending_messages"] = True
            session.context_data["sending_started_at"] = datetime.now().isoformat()
            await self.provider.update_session(session)

            # Note: Typing indicator is now sent in _send_response after TTS decision
            # to avoid sending it before determining if audio should be sent

            # Get all pending messages
            pending_messages = session.clear_pending_messages()
            logger.info(
                f"[BATCH_PROCESSING] 📦 Processing batch of {len(pending_messages)} messages for {phone_number}"
            )

            # Apply batch read delay before processing (simulates human reading multiple messages)
            await self._apply_batch_read_delay(list(pending_messages))

            # Convert message batch to agent input
            logger.debug(
                f"[BATCH_PROCESSING] Converting message batch to agent input for {phone_number}"
            )
            agent_input = await self._convert_message_batch_to_input(
                pending_messages, session
            )

            # Check if batch conversion returned None (empty batch)
            if not agent_input:
                logger.warning(
                    f"[BATCH_PROCESSING] Batch conversion returned None for {phone_number} - skipping empty batch"
                )
                # Clear sending state and finish batch processing
                session.context_data["is_sending_messages"] = False
                session.context_data["sending_completed_at"] = (
                    datetime.now().isoformat()
                )
                session.finish_batch_processing(processing_token)
                await self.provider.update_session(session)
                return None

            # Process with agent
            logger.info(f"[BATCH_PROCESSING] 🤖 Running agent for {phone_number}")
            response, input_tokens, output_tokens = await self._process_with_agent(
                agent_input, session, chat_id=chat_id
            )
            logger.info(
                f"[BATCH_PROCESSING] ✅ Agent processing complete for {phone_number}"
            )

            if response:
                logger.info(
                    f"[BATCH_PROCESSING] Response text length: {len(response.text)}"
                )

            # Send response (use the first message ID for reply if quoting is enabled)
            first_message_id = (
                pending_messages[0].get("id")
                if pending_messages and self.config.quote_messages
                else None
            )
            logger.info(
                f"[BATCH_PROCESSING] 📤 Sending response to {phone_number} (quote_messages={self.config.quote_messages}, reply to: {first_message_id})"
            )

            # CRITICAL: Send response with enhanced error handling
            await self._send_response(phone_number, response, first_message_id)

            # Update session - clear sending state
            session.message_count += len(pending_messages)
            session.last_activity = datetime.now()
            session.context_data["is_sending_messages"] = False
            session.context_data["sending_completed_at"] = datetime.now().isoformat()

            # Finish batch processing with token validation
            session.finish_batch_processing(processing_token)
            await self.provider.update_session(session)

            logger.info(
                f"[BATCH_PROCESSING] ✅ Successfully processed batch for {phone_number}. Total messages processed: {session.message_count}"
            )

            # Call response callbacks
            await self._call_response_callbacks(
                phone_number,
                response,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                chat_id=chat_id,
                processing_status="completed",
            )
            return response

        except Exception as e:
            logger.error(
                f"[BATCH_PROCESSING_ERROR] ❌ Error processing message batch for {phone_number}: {e}",
                exc_info=True,
            )

            # Clear sending state on error
            try:
                session.context_data["is_sending_messages"] = False
                session.context_data["sending_error_at"] = datetime.now().isoformat()
                session.context_data["sending_error"] = str(e)
            except Exception:
                pass

            await self._send_error_message(phone_number)

            # Ensure session state is cleaned up even on error
            session.finish_batch_processing(processing_token)
            await self.provider.update_session(session)

            # Call response callbacks with None response on error
            await self._call_response_callbacks(
                phone_number,
                None,
                input_tokens=0,
                output_tokens=0,
                chat_id=chat_id,
                processing_status="failed",
            )
            raise

    async def _process_single_message(
        self,
        message: WhatsAppMessage,
        session: WhatsAppSession,
        chat_id: ChatId | None = None,
    ) -> GeneratedAssistantMessage[T_Schema]:
        """Process a single message immediately with quote message support."""
        logger.info(
            "[SINGLE_MESSAGE] ═══════════ SINGLE MESSAGE PROCESSING START ═══════════"
        )
        logger.info(f"[SINGLE_MESSAGE] Phone: {message.from_number}")
        logger.info(
            f"[SINGLE_MESSAGE] Current response callbacks count: {len(self._response_callbacks)}"
        )

        try:
            # Show typing indicator
            if self.config.typing_indicator:
                logger.debug(
                    f"[SINGLE_MESSAGE] Sending typing indicator to {message.from_number}"
                )
                await self.provider.send_typing_indicator(
                    message.from_number, self.config.typing_duration
                )

            # Convert WhatsApp message to agent input
            logger.debug(
                f"[SINGLE_MESSAGE] Converting message to agent input for {message.from_number}"
            )
            agent_input = await self._convert_message_to_input(message, session)

            # Process with agent
            logger.info(f"[SINGLE_MESSAGE] 🤖 Running agent for {message.from_number}")
            response, input_tokens, output_tokens = await self._process_with_agent(
                agent_input, session, chat_id=chat_id
            )
            logger.info(
                f"[SINGLE_MESSAGE] ✅ Agent processing complete for {message.from_number}"
            )
            logger.info(f"[SINGLE_MESSAGE] Response generated: {response is not None}")  # type: ignore

            if response:
                logger.info(
                    f"[SINGLE_MESSAGE] Response text length: {len(response.text)}"
                )

            # Send response (quote message if enabled)
            quote_message_id = message.id if self.config.quote_messages else None
            logger.info(
                f"[SINGLE_MESSAGE] 📤 Sending response to {message.from_number} (quote_messages={self.config.quote_messages}, quote_id={quote_message_id})"
            )
            await self._send_response(message.from_number, response, quote_message_id)

            # Update session
            session.message_count += 1
            session.last_activity = datetime.now()
            await self.provider.update_session(session)

            logger.info(
                f"[SINGLE_MESSAGE] ✅ Successfully processed single message for {message.from_number}. Total messages processed: {session.message_count}"
            )

            # Call response callbacks - THIS IS CRITICAL
            logger.info(
                f"[SINGLE_MESSAGE] 📞 About to call response callbacks for {message.from_number}"
            )
            await self._call_response_callbacks(
                message.from_number,
                response,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                chat_id=chat_id,
                processing_status="completed",
            )
            logger.info(
                f"[SINGLE_MESSAGE] ✅ Response callbacks completed for {message.from_number}"
            )

            return response

        except Exception as e:
            logger.error(
                f"[SINGLE_MESSAGE_ERROR] ❌ Error processing single message: {e}",
                exc_info=True,
            )

            # Call response callbacks with None response on error - THIS IS CRITICAL
            logger.info(
                "[SINGLE_MESSAGE] 📞 Calling response callbacks with None response due to error"
            )
            await self._call_response_callbacks(
                phone_number=message.from_number,
                response=None,
                input_tokens=0,
                output_tokens=0,
                chat_id=chat_id,
                processing_status="failed",
            )
            logger.info("[SINGLE_MESSAGE] ✅ Error response callbacks completed")
            raise
        finally:
            logger.info(
                "[SINGLE_MESSAGE] ═══════════ SINGLE MESSAGE PROCESSING END ═══════════"
            )

    async def _message_to_dict(self, message: WhatsAppMessage) -> dict[str, Any]:
        """Convert WhatsApp message to dictionary for storage."""
        message_data: dict[str, Any] = {
            "id": message.id,
            "type": message.__class__.__name__,
            "from_number": message.from_number,
            "to_number": message.to_number,
            "timestamp": message.timestamp.isoformat(),
            "push_name": message.push_name,
        }

        # Add type-specific data
        if isinstance(message, WhatsAppTextMessage):
            message_data["text"] = message.text
        elif isinstance(message, WhatsAppMediaMessage):
            message_data.update(
                {
                    "media_url": message.media_url,
                    "media_mime_type": message.media_mime_type,
                    "caption": message.caption,
                    "filename": getattr(message, "filename", None),
                    "base64_data": getattr(
                        message, "base64_data", None
                    ),  # Include base64 if available
                }
            )

        logger.debug(f"[MESSAGE_TO_DICT] Converted message {message.id} to dict")
        return message_data

    async def _convert_message_batch_to_input(
        self, message_batch: Sequence[dict[str, Any]], session: WhatsAppSession
    ) -> Any:
        """Convert a batch of messages to agent input using phone number as chat_id."""
        logger.info(
            f"[BATCH_CONVERSION] Converting batch of {len(message_batch)} messages to agent input"
        )

        parts: MutableSequence[
            TextPart
            | FilePart
            | Tool[Any]
            | ToolExecutionSuggestion
            | ToolExecutionResult
        ] = []

        # Add batch header if multiple messages
        if len(message_batch) > 1:
            parts.append(
                TextPart(
                    text=f"[Batch of {len(message_batch)} messages received together]"
                )
            )

        # Process each message in the batch
        for i, msg_data in enumerate(message_batch):
            logger.debug(
                f"[BATCH_CONVERSION] Processing message {i + 1}/{len(message_batch)}: {msg_data.get('id')}"
            )

            if i > 0:  # Add separator between messages
                parts.append(TextPart(text="\n\n"))

            # Handle text messages
            if msg_data["type"] == "WhatsAppTextMessage":
                text = msg_data.get("text", "")
                if text:
                    parts.append(TextPart(text=text))
                    logger.debug(f"[BATCH_CONVERSION] Added text part: {text[:50]}...")

            # Handle media messages
            elif msg_data["type"] in [
                "WhatsAppImageMessage",
                "WhatsAppDocumentMessage",
                "WhatsAppAudioMessage",
                "WhatsAppVideoMessage",
            ]:
                try:
                    # CRITICAL FIX: Check if media has base64 data already (for audio messages)
                    # This avoids unnecessary download attempts when media is already available
                    base64_data = msg_data.get("base64_data")
                    if base64_data:
                        logger.info(
                            f"[BATCH_CONVERSION] 🎵 Using base64 data directly for message {msg_data['id']} (no download needed)"
                        )
                        import base64

                        media_bytes = base64.b64decode(base64_data)
                        mime_type = msg_data.get(
                            "media_mime_type", "application/octet-stream"
                        )
                        parts.append(FilePart(data=media_bytes, mime_type=mime_type))
                        logger.debug(
                            f"[BATCH_CONVERSION] Successfully decoded base64 media for {msg_data['id']} ({len(media_bytes)} bytes)"
                        )
                    else:
                        logger.debug(
                            f"[BATCH_CONVERSION] Downloading media for message {msg_data['id']}"
                        )
                        media_data = await self.provider.download_media(msg_data["id"])
                        parts.append(
                            FilePart(
                                data=media_data.data, mime_type=media_data.mime_type
                            )
                        )
                        logger.debug(
                            f"[BATCH_CONVERSION] Successfully downloaded media for {msg_data['id']}"
                        )

                    # Add caption if present
                    caption = msg_data.get("caption")
                    if caption:
                        parts.append(TextPart(text=f"Caption: {caption}"))
                        logger.debug(f"[BATCH_CONVERSION] Added caption: {caption}")

                except Exception as e:
                    logger.error(
                        f"[BATCH_CONVERSION] Failed to download media from batch: {e}"
                    )
                    parts.append(TextPart(text="[Media file - failed to download]"))

        # If no parts were added, skip processing instead of creating placeholder
        if not parts:
            logger.warning(
                "[BATCH_CONVERSION] No parts were created - skipping batch processing to avoid empty message"
            )
            # Return None to indicate this batch should be skipped
            # This prevents the agent from receiving empty content
            return None

        # Create user message with first message's push name
        first_message = message_batch[0] if message_batch else {}
        push_name = first_message.get("push_name", "User")
        user_message = UserMessage.create_named(parts=parts, name=push_name)
        logger.debug(f"[BATCH_CONVERSION] Created user message with name: {push_name}")

        # Simply return the user message - Agent will handle conversation history via chat_id
        return user_message

    async def _call_response_callbacks(
        self,
        phone_number: PhoneNumber,
        response: GeneratedAssistantMessage[Any] | None,
        input_tokens: float,
        output_tokens: float,
        *,
        chat_id: ChatId | None = None,
        processing_status: str = "completed",
    ) -> None:
        """Call all registered response callbacks with (phone_number, chat_id, response, context)."""
        logger.info("[CALLBACKS] ═══════════ CALLING RESPONSE CALLBACKS ═══════════")
        logger.info(f"[CALLBACKS] Phone number: {phone_number}")
        logger.info(f"[CALLBACKS] Response provided: {response is not None}")
        logger.info(f"[CALLBACKS] chat_id: {chat_id}")
        logger.info(f"[CALLBACKS] Processing status: {processing_status}")

        scope_key = self._build_callback_scope_key(
            chat_id=chat_id,
            phone_number=phone_number,
        )
        callbacks_to_call = [
            cb
            for cb in self._response_callbacks
            if cb.scope_key is None or cb.scope_key == scope_key
        ]
        logger.info(
            f"[CALLBACKS] Total callbacks to call: {len(callbacks_to_call)}"
        )

        if response:
            logger.info(f"[CALLBACKS] Response text length: {len(response.text)}")
            logger.debug(f"[CALLBACKS] Response text preview: {response.text[:100]}...")

        if not callbacks_to_call:
            logger.warning("[CALLBACKS] ⚠️ No callbacks registered to call!")
            return

        callbacks_to_remove: list[CallbackWithContext] = []

        for i, cb in enumerate(callbacks_to_call):
            logger.info(
                f"[CALLBACKS] 🔄 Calling callback {i + 1}/{len(callbacks_to_call)}"
            )
            logger.info(
                f"[CALLBACKS] Callback function: {getattr(cb.callback, '__name__', 'unnamed')}"
            )
            logger.info(
                f"[CALLBACKS] Callback context keys: {list(cb.context.keys()) if cb.context else 'None'}"
            )

            callback_context = dict(cb.context)
            callback_context.update(
                {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "processing_status": processing_status,
                }
            )

            try:
                if inspect.iscoroutinefunction(cb.callback):
                    await cb.callback(phone_number, chat_id, response, callback_context)
                else:
                    cb.callback(phone_number, chat_id, response, callback_context)
            except Exception as e:
                logger.error(
                    f"[CALLBACKS] ❌ Error calling callback {i + 1} for {phone_number}: {e}",
                    exc_info=True,
                )
            finally:
                if not cb.persistent:
                    callbacks_to_remove.append(cb)

        if callbacks_to_remove:
            self._response_callbacks = [
                existing
                for existing in self._response_callbacks
                if existing not in callbacks_to_remove
            ]
            logger.info(
                "[CALLBACKS] Removed %s one-shot callback(s). Remaining=%s",
                len(callbacks_to_remove),
                len(self._response_callbacks),
            )

        logger.info(
            f"[CALLBACKS] ✅ All {len(callbacks_to_call)} callbacks processed"
        )
        logger.info("[CALLBACKS] ═══════════ CALLBACKS COMPLETE ═══════════")

    async def _convert_message_to_input(
        self, message: WhatsAppMessage, session: WhatsAppSession
    ) -> Any:
        """Convert WhatsApp message to agent input using phone number as chat_id."""
        logger.info(
            f"[SINGLE_CONVERSION] Converting single message to agent input for {message.from_number}"
        )

        parts: MutableSequence[
            TextPart
            | FilePart
            | Tool[Any]
            | ToolExecutionSuggestion
            | ToolExecutionResult
        ] = []

        # Handle text messages
        if isinstance(message, WhatsAppTextMessage):
            parts.append(TextPart(text=message.text))
            logger.debug(f"[SINGLE_CONVERSION] Added text part: {message.text[:50]}...")

        # Handle media messages
        elif isinstance(message, WhatsAppMediaMessage):
            try:
                # CRITICAL FIX: Check if media has base64 data already (for audio messages)
                # This avoids unnecessary download attempts when media is already available
                if message.base64_data:
                    logger.info(
                        f"[SINGLE_CONVERSION] 🎵 Using base64 data directly for message {message.id} (no download needed)"
                    )
                    import base64

                    media_bytes = base64.b64decode(message.base64_data)
                    parts.append(
                        FilePart(data=media_bytes, mime_type=message.media_mime_type)
                    )
                    logger.debug(
                        f"[SINGLE_CONVERSION] Successfully decoded base64 media for {message.id} ({len(media_bytes)} bytes)"
                    )
                else:
                    logger.debug(
                        f"[SINGLE_CONVERSION] Downloading media for message {message.id}"
                    )
                    media_data = await self.provider.download_media(message.id)
                    parts.append(
                        FilePart(data=media_data.data, mime_type=media_data.mime_type)
                    )
                    logger.debug(
                        f"[SINGLE_CONVERSION] Successfully downloaded media for {message.id}"
                    )

                # Add caption if present
                if message.caption:
                    parts.append(TextPart(text=f"Caption: {message.caption}"))
                    logger.debug(
                        f"[SINGLE_CONVERSION] Added caption: {message.caption}"
                    )

            except Exception as e:
                logger.error(f"[SINGLE_CONVERSION] Failed to download media: {e}")
                parts.append(TextPart(text="[Media file - failed to download]"))

        # Create user message
        user_message = UserMessage.create_named(parts=parts, name=message.push_name)
        logger.debug(
            f"[SINGLE_CONVERSION] Created user message with name: {message.push_name}"
        )

        # Simply return the user message - Agent will handle conversation history via chat_id
        return user_message

    async def _process_with_agent(
        self,
        agent_input: AgentInput,
        session: WhatsAppSession,
        chat_id: ChatId | None = None,
    ) -> tuple[GeneratedAssistantMessage[Any], int, int]:
        """Process input with agent using custom chat_id for conversation persistence."""
        logger.info("[AGENT_PROCESSING] Starting agent processing")

        # MUDANÇA CRÍTICA: Recuperar chat_id personalizado da sessão se não fornecido
        effective_chat_id = chat_id
        if not effective_chat_id:
            effective_chat_id = session.context_data.get("custom_chat_id")
        if not effective_chat_id:
            effective_chat_id = session.phone_number

        logger.info(f"[AGENT_PROCESSING] Using effective chat_id: {effective_chat_id}")
        logger.info(
            f"[AGENT_PROCESSING] Chat ID type: {'CUSTOM' if chat_id else 'FALLBACK'}"
        )

        try:
            async with self.agent.start_mcp_servers_async():
                logger.debug("[AGENT_PROCESSING] Started MCP servers")

                # Run agent with effective chat_id for conversation persistence
                result = await self.agent.run_async(
                    agent_input,
                    chat_id=effective_chat_id,
                )
                input_tokens = result.input_tokens
                output_tokens = result.output_tokens

                logger.debug(f"Input tokens: {input_tokens}")
                logger.debug(f"Output tokens: {output_tokens}")

                logger.info("[AGENT_PROCESSING] Agent run completed successfully")

            if result.generation:
                generated_message = result.generation.message
                logger.info(
                    f"[AGENT_PROCESSING] Generated response (length: {len(generated_message.text)})"
                )

                # FIXED: Always clean thinking tags from all parts
                cleaned_parts: list[TextPart | ToolExecutionSuggestion] = []

                for part in generated_message.parts:
                    if part.type == "text":
                        from agentle.generations.models.message_parts.text import (
                            TextPart,
                        )

                        part_text = str(part.text) if part.text else ""
                        cleaned_part_text = self._remove_thinking_tags(part_text)
                        cleaned_parts.append(TextPart(text=cleaned_part_text))
                    else:
                        # FIXED: Keep non-text parts
                        cleaned_parts.append(part)

                # Always return cleaned message
                return (
                    GeneratedAssistantMessage[Any](
                        parts=cleaned_parts,
                        parsed=generated_message.parsed,
                    ),
                    input_tokens,
                    output_tokens,
                )

            logger.warning("[AGENT_PROCESSING] No generation found in result")
            from agentle.generations.models.message_parts.text import TextPart

            return (
                GeneratedAssistantMessage[Any](
                    parts=[
                        TextPart(
                            text="Desculpe, não consegui processar sua mensagem no momento. Tente novamente."
                        )
                    ],
                    parsed=None,
                ),
                input_tokens,
                output_tokens,
            )

        except Exception as e:
            logger.error(
                f"[AGENT_PROCESSING_ERROR] Agent processing error: {e}", exc_info=True
            )
            raise

    def _remove_thinking_tags(self, text: str) -> str:
        """Remove thinking tags and their content from the response text.

        This method handles:
        - Multiple occurrences of thinking tags
        - Tags spanning multiple lines
        - Malformed or incomplete tags
        - Case-insensitive matching
        - Responses with no thinking tags

        Args:
            text: The original response text that may contain thinking tags

        Returns:
            The cleaned text with thinking tags and their content removed
        """
        if not text:
            return text

        original_text = text

        # Pattern 1: Complete thinking tags (case-insensitive, multiline)
        # Use re.DOTALL flag to make . match newlines as well
        text = re.sub(
            r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL | re.IGNORECASE
        )

        # Pattern 2: Handle malformed tags or incomplete tags
        # Remove opening thinking tags without closing tags (to the end of text)
        text = re.sub(r"<thinking>.*?$", "", text, flags=re.DOTALL | re.IGNORECASE)

        # Pattern 3: Remove any remaining orphaned closing tags
        text = re.sub(r"</thinking>", "", text, flags=re.IGNORECASE)

        # Pattern 4: Handle variations with attributes or whitespace
        text = re.sub(
            r"<thinking[^>]*>.*?</thinking[^>]*>",
            "",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )

        # Clean up any extra whitespace that might be left after removing thinking tags
        # Replace multiple consecutive newlines with double newlines
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

        # Remove excessive spaces
        text = re.sub(r"[ \t]+", " ", text)

        # Clean up leading/trailing whitespace
        text = text.strip()

        # Log if thinking tags were found and removed
        if original_text != text:
            thinking_tags_removed = len(
                re.findall(
                    r"<thinking[^>]*>.*?</thinking[^>]*>",
                    original_text,
                    flags=re.DOTALL | re.IGNORECASE,
                )
            )
            logger.warning(
                f"[THINKING_CLEANUP] Removed {thinking_tags_removed} thinking tag(s). "
                + f"Original length: {len(original_text)}, Cleaned length: {len(text)}"
            )

            # Additional debug info for persistent issues
            if self.config.debug_mode:
                logger.debug(
                    f"[THINKING_CLEANUP] Original text preview: {original_text[:200]}..."
                )
                logger.debug(
                    f"[THINKING_CLEANUP] Cleaned text preview: {text[:200]}..."
                )

        return text

    def _create_whatsapp_renderer(self) -> mistune.HTMLRenderer:
        """Create a mistune renderer for WhatsApp formatting."""

        class WhatsAppRenderer(mistune.HTMLRenderer):
            """Custom renderer for WhatsApp markdown format."""

            def heading(self, text: str, level: int, **attrs: Any) -> str:
                """Render headings as bold text with separators."""
                if level == 1:
                    return f"\n*{text}*\n{'═' * min(len(text), 30)}\n"
                elif level == 2:
                    return f"\n*{text}*\n{'─' * min(len(text), 30)}\n"
                else:
                    return f"\n*{text}*\n"

            def strong(self, text: str) -> str:
                """Render bold as WhatsApp bold."""
                return f"*{text}*"

            def emphasis(self, text: str) -> str:
                """Render italic as WhatsApp italic."""
                return f"_{text}_"

            def strikethrough(self, text: str) -> str:
                """Render strikethrough as WhatsApp strikethrough."""
                return f"~{text}~"

            def codespan(self, text: str) -> str:
                """Render inline code as WhatsApp monospace."""
                return f"```{text}```"

            def block_code(self, code: str, info: str | None = None) -> str:
                """Render code blocks."""
                return f"```{code}```\n"

            def link(self, text: str, url: str, title: str | None = None) -> str:
                """Render links as text (url)."""
                return f"{text} ({url})"

            def image(self, text: str, url: str, title: str | None = None) -> str:
                """Render images as descriptive text."""
                return f"[Imagem: {text}]" if text else "[Imagem]"

            def block_quote(self, text: str) -> str:
                """Render blockquotes with indentation."""
                lines = text.strip().split("\n")
                return "\n".join(f"  ┃ {line}" for line in lines) + "\n"

            def list(self, text: str, ordered: bool, **attrs: Any) -> str:
                """Render lists."""
                return f"\n{text}"

            def list_item(self, text: str, **attrs: Any) -> str:
                """Render list items."""
                return f"{text}\n"

            def paragraph(self, text: str) -> str:
                """Render paragraphs."""
                return f"{text}\n\n"

            def thematic_break(self) -> str:
                """Render horizontal rules."""
                return "─" * 30 + "\n"

            def linebreak(self) -> str:
                """Render line breaks."""
                return "\n"

            def text(self, text: str) -> str:
                """Render plain text."""
                return text

        return WhatsAppRenderer()

    def _format_whatsapp_markdown(self, text: str) -> str:
        """Convert standard markdown to WhatsApp-compatible formatting using mistune.

        WhatsApp supports:
        - *bold* for bold text
        - _italic_ for italic text
        - ~strikethrough~ for strikethrough text
        - ```code``` for monospace text
        - No support for headers, tables, or complex markdown structures

        This method converts:
        - Headers (# ## ###) to bold text with separators
        - Tables to formatted text
        - Markdown lists to plain text lists (preserving line breaks)
        - Links to "text (url)" format
        - Images to descriptive text
        - Blockquotes to indented text
        """
        if not text:
            return text

        # Use mistune for markdown parsing
        try:
            renderer = self._create_whatsapp_renderer()
            markdown = mistune.create_markdown(
                renderer=renderer, plugins=["strikethrough", "table"]
            )
            result = markdown(text)
            # Ensure result is a string
            if isinstance(result, str):
                # Clean up extra newlines
                result = re.sub(r"\n{3,}", "\n\n", result)
                return result.strip()
            else:
                logger.warning(
                    f"[MARKDOWN] Mistune returned non-string result: {type(result)}"
                )
                return text
        except Exception as e:
            logger.warning(
                f"[MARKDOWN] Mistune conversion failed, returning original text: {e}"
            )
            return text

    async def _send_response(
        self,
        to: PhoneNumber,
        response: GeneratedAssistantMessage[T_Schema] | str,
        reply_to: str | None = None,
    ) -> None:
        """Send response message(s) to user with enhanced error handling and retry logic.

        This method handles the complete response sending flow including text-to-speech,
        human-like delays, typing indicators, message splitting, and error handling.

        Response Sending Flow:
            1. Extract and format response text
            2. Attempt TTS audio generation (if configured and chance succeeds)
            3. Apply typing delay (if human delays enabled and TTS not sent)
            4. Show typing indicator (if configured and not already shown during delay)
            5. Split long messages if needed
            6. Send each message part with send delay between parts
            7. Handle errors with retry logic

        Human-Like Delays:
            When enable_human_delays is True, this method applies two types of delays:

            1. Typing Delay: Applied before sending the response to simulate the time
               a human would take to compose and type the message. The delay is based
               on response length and includes composition planning time.

            2. Send Delay: Applied immediately before each message transmission to
               simulate the brief final review time before hitting send. This delay
               is applied to each message part independently.

            If TTS audio is successfully sent, the typing delay is skipped since the
            audio generation time already provides a natural delay.

        Args:
            to: Phone number of the recipient.
            response: The response to send. Can be a GeneratedAssistantMessage or string.
            reply_to: Optional message ID to reply to (for message quoting).

        Raises:
            Exceptions are caught and logged. Failed messages trigger retry logic
            if configured.

        Example:
            >>> response = GeneratedAssistantMessage(text="Hello! How can I help?")
            >>> await self._send_response(
            ...     to="1234567890",
            ...     response=response,
            ...     reply_to="msg_123"
            ... )
        """
        response_text = ""

        if isinstance(response, GeneratedAssistantMessage):
            # Check if we have structured output (parsed)
            if response.parsed:
                # Use the 'response' field from structured output
                response_text = response.parsed.response
                logger.debug(
                    "[SEND_RESPONSE] Using structured output 'response' field "
                    + f"(schema: {type(response.parsed).__name__})"
                )
            else:
                # Fallback to text field
                response_text = response.text
                logger.debug("[SEND_RESPONSE] Using standard text response")
        else:
            # Direct string
            response_text = response

        # Apply WhatsApp-specific markdown formatting
        response_text = self._format_whatsapp_markdown(response_text)

        logger.info(
            f"[SEND_RESPONSE] Sending response to {to} (length: {len(response_text)}, reply_to: {reply_to})"
        )

        # Track if TTS was successfully sent (to skip typing delay for audio)
        tts_sent_successfully = False

        # Check if we should send audio via TTS
        should_attempt_tts = (
            self.tts_provider
            and self.config.speech_config
            and self.config.speech_play_chance > 0
            and self._validate_tts_configuration()
        )

        if should_attempt_tts:
            import random

            # Determine if we should play speech based on chance
            should_play_speech = random.random() < self.config.speech_play_chance

            if should_play_speech:
                logger.info(
                    f"[TTS] Attempting to send audio response to {to} (chance: {self.config.speech_play_chance * 100}%)"
                )
                try:
                    # Show recording indicator while synthesizing
                    if self.config.typing_indicator:
                        logger.debug(
                            f"[TTS] Sending recording indicator to {to} during synthesis"
                        )
                        # Use a more appropriate duration for recording indicator
                        # Based on text length: minimum 2s, maximum 10s, or estimated synthesis time
                        estimated_duration = max(
                            2, min(10, len(response_text) // 50 + 2)
                        )
                        await self.provider.send_recording_indicator(
                            to, estimated_duration
                        )

                    # Synthesize speech
                    # We know these are not None due to validation above
                    assert self.tts_provider is not None
                    assert self.config.speech_config is not None
                    speech_result = await self.tts_provider.synthesize_async(
                        response_text, config=self.config.speech_config
                    )

                    # Try to upload to file storage if available
                    audio_url = None
                    if self.file_storage_manager:
                        try:
                            import base64
                            import time

                            # Decode base64 to bytes
                            audio_bytes = base64.b64decode(speech_result.audio)

                            # Generate unique filename
                            timestamp = int(time.time())
                            extension = self._get_audio_extension(speech_result.format)
                            filename = f"tts_{timestamp}.{extension}"

                            # Upload to storage
                            audio_url = await self.file_storage_manager.upload_file(
                                file_data=audio_bytes,
                                filename=filename,
                                mime_type=str(speech_result.mime_type),
                            )

                            logger.info(f"[TTS] Audio uploaded to storage: {audio_url}")

                        except Exception as e:
                            logger.warning(
                                f"[TTS] Failed to upload to storage, falling back to base64: {e}"
                            )
                            audio_url = None

                    # Send audio message (URL or base64)
                    if audio_url:
                        # Try URL method first
                        try:
                            await self.provider.send_audio_message_by_url(
                                to=to,
                                audio_url=audio_url,
                                quoted_message_id=reply_to
                                if self.config.quote_messages
                                else None,
                            )
                            logger.info(f"[TTS] Audio sent via URL to {to}")
                        except Exception as e:
                            logger.warning(
                                f"[TTS] URL method failed, falling back to base64: {e}"
                            )
                            # Fallback to base64
                            await self.provider.send_audio_message(
                                to=to,
                                audio_base64=speech_result.audio,
                                quoted_message_id=reply_to
                                if self.config.quote_messages
                                else None,
                            )
                            logger.info(f"[TTS] Audio sent via base64 to {to}")
                    else:
                        # Use base64 method (current behavior)
                        await self.provider.send_audio_message(
                            to=to,
                            audio_base64=speech_result.audio,
                            quoted_message_id=reply_to
                            if self.config.quote_messages
                            else None,
                        )
                        logger.info(f"[TTS] Audio sent via base64 to {to}")

                    logger.info(
                        f"[TTS] Successfully sent audio response to {to}",
                        extra={
                            "to_number": to,
                            "text_length": len(response_text),
                            "mime_type": str(speech_result.mime_type),
                            "format": str(speech_result.format),
                        },
                    )
                    # Audio sent successfully, mark flag and return early
                    tts_sent_successfully = True
                    logger.info(
                        "[TTS] Skipping typing delay since TTS audio was sent successfully"
                    )
                    return

                except Exception as e:
                    # Check if this is a specific Evolution API media upload error
                    error_message = str(e).lower()
                    if "media upload failed" in error_message or "400" in error_message:
                        logger.warning(
                            f"[TTS] Evolution API media upload failed for {to}, falling back to text: {e}",
                            extra={
                                "to_number": to,
                                "error_type": type(e).__name__,
                                "error": str(e),
                                "fallback_reason": "evolution_api_media_upload_failed",
                            },
                        )
                    else:
                        logger.warning(
                            f"[TTS] Failed to send audio response to {to}, falling back to text: {e}",
                            extra={
                                "to_number": to,
                                "error_type": type(e).__name__,
                                "error": str(e),
                                "fallback_reason": "tts_synthesis_or_send_failed",
                            },
                        )
                    # Fall through to send text message instead

        # Split messages by line breaks and length
        messages = self._split_message_by_line_breaks(response_text)
        logger.info(f"[SEND_RESPONSE] Split response into {len(messages)} parts")

        # Apply typing delay before sending messages (simulates human typing time)
        # This should be done before the typing indicator to coordinate properly
        # Note: This is only reached if TTS was not used or if TTS failed and fell back to text
        if should_attempt_tts and not tts_sent_successfully:
            logger.info(
                "[SEND_RESPONSE] TTS failed, applying typing delay for text fallback"
            )
        await self._apply_typing_delay(response_text, to)

        # Show typing indicator ONCE before sending all messages
        # Only send typing indicator if we're not attempting TTS or if TTS failed
        # Skip if typing delay already handled the indicator
        typing_delay_handled_indicator = (
            self.config.enable_human_delays
            and self.config.show_typing_during_delay
            and self.config.typing_indicator
        )

        if typing_delay_handled_indicator:
            logger.debug(
                "[SEND_RESPONSE] Skipping redundant typing indicator - already sent during typing delay"
            )

        if (
            self.config.typing_indicator
            and not should_attempt_tts
            and not typing_delay_handled_indicator
        ):
            try:
                logger.debug(
                    f"[SEND_RESPONSE] Sending typing indicator to {to} before sending {len(messages)} message(s)"
                )
                await self.provider.send_typing_indicator(
                    to, self.config.typing_duration
                )
            except Exception as e:
                # Don't let typing indicator failures break message sending
                logger.warning(f"[SEND_RESPONSE] Failed to send typing indicator: {e}")
        elif (
            self.config.typing_indicator
            and should_attempt_tts
            and not typing_delay_handled_indicator
        ):
            # TTS was attempted but failed, send typing indicator for text fallback
            # Skip if typing delay already handled the indicator
            try:
                logger.debug(
                    f"[SEND_RESPONSE] TTS failed, sending typing indicator to {to} for text fallback"
                )
                await self.provider.send_typing_indicator(
                    to, self.config.typing_duration
                )
            except Exception as e:
                # Don't let typing indicator failures break message sending
                logger.warning(f"[SEND_RESPONSE] Failed to send typing indicator: {e}")

        # Track sending state to handle partial failures
        successfully_sent_count = 0
        failed_parts: list[dict[str, Any]] = []

        for i, msg in enumerate(messages):
            logger.debug(
                f"[SEND_RESPONSE] Sending message part {i + 1}/{len(messages)} to {to}"
            )

            # Only quote the first message if quote_messages is enabled
            quoted_id = reply_to if i == 0 else None

            # Retry logic for individual message parts
            max_retries = 3
            retry_delay = 1.0
            sent_successfully = False

            for attempt in range(max_retries + 1):
                try:
                    # Apply send delay before transmitting message (simulates final review)
                    await self._apply_send_delay()

                    sent_message = await self.provider.send_text_message(
                        to=to, text=msg, quoted_message_id=quoted_id
                    )
                    logger.debug(
                        f"[SEND_RESPONSE] Successfully sent message part {i + 1} to {to}: {sent_message.id}"
                    )
                    sent_successfully = True
                    successfully_sent_count += 1
                    break

                except Exception as e:
                    if attempt < max_retries:
                        # Calculate exponential backoff delay
                        delay = retry_delay * (2**attempt)
                        logger.warning(
                            f"[SEND_RESPONSE] Failed to send message part {i + 1} to {to} (attempt {attempt + 1}/{max_retries + 1}), retrying in {delay}s: {e}"
                        )
                        await asyncio.sleep(delay)
                    else:
                        # Final failure - log but continue with next parts
                        logger.error(
                            f"[SEND_RESPONSE_ERROR] Failed to send message part {i + 1} to {to} after {max_retries + 1} attempts: {e}"
                        )
                        failed_parts.append(
                            {
                                "part_number": i + 1,
                                "text": msg[:100] + "..." if len(msg) > 100 else msg,
                                "error": str(e),
                            }
                        )

            # If this part failed, continue with next parts instead of stopping
            if not sent_successfully:
                logger.warning(
                    f"[SEND_RESPONSE] Message part {i + 1} failed, continuing with remaining parts"
                )

            # Delay between messages (respecting typing duration + small buffer)
            if i < len(messages) - 1:
                # Use typing duration if typing indicator is enabled, otherwise use a small delay
                inter_message_delay = (
                    self.config.typing_duration + 0.5
                    if self.config.typing_indicator
                    else 1.0
                )

                # Calculate total delay including send delay if human delays are enabled
                if self.config.enable_human_delays and self._delay_calculator:
                    # Send delay will be applied before next message, so log total expected delay
                    estimated_send_delay = (
                        self.config.min_send_delay_seconds
                        + self.config.max_send_delay_seconds
                    ) / 2
                    total_delay = inter_message_delay + estimated_send_delay
                    logger.debug(
                        f"[SEND_RESPONSE] Inter-message delay: {inter_message_delay:.2f}s "
                        + f"(+ ~{estimated_send_delay:.2f}s send delay = ~{total_delay:.2f}s total)"
                    )
                else:
                    logger.debug(
                        f"[SEND_RESPONSE] Waiting {inter_message_delay}s before sending next message part"
                    )

                await asyncio.sleep(inter_message_delay)

        # Log final sending results
        if failed_parts:
            logger.error(
                f"[SEND_RESPONSE] Completed sending with {successfully_sent_count}/{len(messages)} parts successful, {len(failed_parts)} failed"
            )
            logger.error(f"[SEND_RESPONSE] Failed parts details: {failed_parts}")

            # Optionally send error notification for partial failures
            if successfully_sent_count == 0:
                # Total failure - send error message
                await self._send_error_message(to, reply_to)
            elif len(failed_parts) > 0:
                # Partial failure - optionally notify user
                try:
                    error_msg = f"⚠️ Algumas partes da mensagem podem não ter sido enviadas devido a problemas técnicos. {len(failed_parts)} de {len(messages)} partes falharam."
                    formatted_error_msg = self._format_whatsapp_markdown(error_msg)
                    await self.provider.send_text_message(
                        to=to, text=formatted_error_msg
                    )
                except Exception as e:
                    logger.warning(
                        f"[SEND_RESPONSE] Failed to send partial failure notification: {e}"
                    )
        else:
            logger.info(
                f"[SEND_RESPONSE] Successfully sent all {len(messages)} message parts to {to}"
            )

    def _validate_tts_configuration(self) -> bool:
        """Validate TTS configuration before attempting synthesis."""
        try:
            if not self.config.speech_config:
                logger.debug("[TTS_VALIDATION] No speech_config provided")
                return False

            # Check if voice_id is provided
            if not self.config.speech_config.voice_id:
                logger.warning(
                    "[TTS_VALIDATION] speech_config.voice_id is required but not provided"
                )
                return False

            # Check if TTS provider is properly configured
            if not self.tts_provider:
                logger.warning("[TTS_VALIDATION] TTS provider is not configured")
                return False

            logger.debug(
                f"[TTS_VALIDATION] TTS configuration is valid: voice_id={self.config.speech_config.voice_id}"
            )
            return True

        except Exception as e:
            logger.warning(
                f"[TTS_VALIDATION] Failed to validate TTS configuration: {e}"
            )
            return False

    def _get_audio_extension(self, format_type: Any) -> str:
        """Get file extension from TTS format."""
        format_str = str(format_type)
        if "mp3" in format_str:
            return "mp3"
        elif "wav" in format_str:
            return "wav"
        elif "ogg" in format_str:
            return "ogg"
        else:
            return "mp3"  # default

    def _split_message_by_line_breaks(self, text: str) -> Sequence[str]:
        """Split message by line breaks first, then by length if needed with enhanced validation.

        CRITICAL: This method must preserve line breaks within messages for proper WhatsApp formatting.
        """
        if not text or not text.strip():
            return ["[Mensagem vazia]"]  # Portuguese: "Empty message"

        try:
            # Check if entire text fits in one message - if so, return it as-is
            if len(text) <= self.config.max_message_length:
                logger.debug(
                    f"[SPLIT_MESSAGE] Message fits in single message ({len(text)} chars), returning as-is"
                )
                return [text]

            logger.info(
                f"[SPLIT_MESSAGE] Message too long ({len(text)} chars), splitting into multiple messages"
            )

            # First split by double line breaks (paragraphs)
            paragraphs = text.split("\n\n")
            messages: MutableSequence[str] = []

            for paragraph in paragraphs:
                if not paragraph.strip():
                    continue

                # If paragraph fits, keep it intact with all line breaks
                if len(paragraph) <= self.config.max_message_length:
                    messages.append(paragraph)
                    continue

                # Paragraph is too long - need to split it
                # Check if paragraph is a list (has list markers)
                lines = paragraph.split("\n")
                is_list_paragraph = self._is_list_content(lines)

                if is_list_paragraph:
                    # Group list items together, preserving line breaks
                    grouped_list = self._group_list_items(lines)
                    messages.extend(grouped_list)
                else:
                    # For non-list paragraphs, try to keep lines together
                    current_chunk = ""
                    for line in lines:
                        if not line.strip():
                            # Preserve empty lines for spacing
                            if current_chunk:
                                current_chunk += "\n"
                            continue

                        # Try adding this line to current chunk
                        test_chunk = (
                            current_chunk + ("\n" if current_chunk else "") + line
                        )

                        if len(test_chunk) <= self.config.max_message_length:
                            current_chunk = test_chunk
                        else:
                            # Current chunk is full, save it
                            if current_chunk:
                                messages.append(current_chunk)

                            # Check if single line is too long
                            if len(line) > self.config.max_message_length:
                                # Split long line by length
                                split_lines = self._split_long_line(line)
                                messages.extend(split_lines)
                                current_chunk = ""
                            else:
                                current_chunk = line

                    # Add remaining chunk
                    if current_chunk:
                        messages.append(current_chunk)

            # Filter out empty messages and validate
            final_messages = []
            for msg in messages:
                if msg and msg.strip():
                    # Ensure message doesn't exceed WhatsApp's absolute limit
                    if len(msg) > 65536:  # WhatsApp's hard limit
                        # Split even further if needed
                        for i in range(0, len(msg), 65536):
                            chunk = msg[i : i + 65536]
                            if chunk.strip():
                                # Don't strip - preserve line breaks
                                final_messages.append(chunk)
                    else:
                        # Don't strip - preserve line breaks within the message
                        final_messages.append(msg)

            # If no valid messages were created, return a placeholder
            if not final_messages:
                final_messages = [
                    "[Não foi possível processar a mensagem]"
                ]  # Portuguese: "Could not process message"

            # Apply max split messages limit
            if len(final_messages) > self.config.max_split_messages:
                logger.info(
                    f"[SPLIT_MESSAGE] Limiting messages from {len(final_messages)} to {self.config.max_split_messages}"
                )
                final_messages = self._apply_message_limit(
                    final_messages, self.config.max_split_messages
                )

            logger.debug(
                f"[SPLIT_MESSAGE] Split message of {len(text)} chars into {len(final_messages)} parts"
            )

            # Log if we have many parts (potential performance issue)
            if len(final_messages) > 10:
                logger.warning(
                    f"[SPLIT_MESSAGE] Large message split into {len(final_messages)} parts - this may take time to send"
                )

            return final_messages

        except Exception as e:
            logger.error(f"[SPLIT_MESSAGE_ERROR] Error splitting message: {e}")
            # Fallback: return original message truncated if needed
            if len(text) <= self.config.max_message_length:
                return [text]
            else:
                return [text[: self.config.max_message_length]]

    def _split_long_line(self, line: str) -> Sequence[str]:
        """Split a single long line into chunks that fit within message length limits."""
        if len(line) <= self.config.max_message_length:
            return [line]

        chunks: MutableSequence[str] = []

        # Try to split by sentences first (by periods, exclamation marks, question marks)
        sentence_endings = [". ", "! ", "? "]
        sentences: MutableSequence[str] = []
        current_sentence = ""

        i = 0
        while i < len(line):
            current_sentence += line[i]

            # Check if we hit a sentence ending
            for ending in sentence_endings:
                if line[i : i + len(ending)] == ending:
                    sentences.append(current_sentence)
                    current_sentence = ""
                    i += len(ending) - 1
                    break

            i += 1

        # Add remaining text as last sentence
        if current_sentence:
            sentences.append(current_sentence)

        # If we couldn't split by sentences effectively, fall back to word splitting
        if len(sentences) <= 1:
            sentences = line.split(" ")

        # Group sentences/words into chunks that fit
        current_chunk = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            test_chunk = current_chunk + (" " if current_chunk else "") + sentence

            if len(test_chunk) <= self.config.max_message_length:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = sentence
                else:
                    # Single sentence/word is too long, hard split it
                    for i in range(0, len(sentence), self.config.max_message_length):
                        chunk = sentence[i : i + self.config.max_message_length]
                        chunks.append(chunk)
                    current_chunk = ""

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _is_list_content(self, lines: Sequence[str]) -> bool:
        """Check if lines contain list markers (numbered or bullet points).

        Detects various list formats including:
        - Numbered lists: "1.", "2)", "1 -"
        - Bullet points: "•", "*", "-", "→"
        - Indented sub-items
        """
        if not lines:
            return False

        list_markers = 0
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # Check for numbered lists: "1.", "2)", "1 -", etc.
            if re.match(r"^\d+[\.\)]\s", stripped) or re.match(
                r"^\d+\s[-–—]\s", stripped
            ):
                list_markers += 1
            # Check for bullet points: "•", "*", "-", "→", etc.
            elif re.match(r"^[•\*\-→▪►]\s", stripped):
                list_markers += 1
            # Check for indented items (common in nested lists)
            elif re.match(r"^\s+[•\*\-→▪►]\s", line):
                list_markers += 1

        # If more than 30% of non-empty lines are list items, consider it a list
        # Lowered threshold to catch lists with headers/descriptions
        non_empty_lines = sum(1 for line in lines if line.strip())
        return non_empty_lines > 0 and (list_markers / non_empty_lines) >= 0.3

    def _group_list_items(self, lines: Sequence[str]) -> Sequence[str]:
        """Group list items together to avoid splitting each item into a separate message.

        CRITICAL: Preserves line breaks between list items for proper WhatsApp formatting.
        """
        if not lines:
            return []

        messages: MutableSequence[str] = []
        current_group: MutableSequence[str] = []
        current_length = 0

        for line in lines:
            # Don't strip the line completely - preserve leading spaces for indentation
            # But remove trailing whitespace
            line = line.rstrip()

            # Skip completely empty lines but preserve them in the group for spacing
            if not line:
                # Add empty line to group for spacing between items
                if current_group:
                    current_group.append("")
                continue

            # Calculate potential length if we add this line
            potential_length = current_length + len(line)
            if current_group:
                potential_length += 1  # For the newline

            # If adding this line would exceed the limit, save current group and start new one
            if potential_length > self.config.max_message_length and current_group:
                # Join with newlines to preserve line breaks
                messages.append("\n".join(current_group))
                current_group = [line]
                current_length = len(line)
            else:
                current_group.append(line)
                current_length = potential_length

        # Add remaining group
        if current_group:
            # Join with newlines to preserve line breaks
            messages.append("\n".join(current_group))

        return messages

    def _apply_message_limit(
        self, messages: Sequence[str], max_messages: int
    ) -> Sequence[str]:
        """
        Apply limit to number of split messages.
        If exceeded, group remaining messages together.
        """
        if len(messages) <= max_messages:
            return messages

        # Keep first (max_messages - 1) messages as-is
        limited_messages = list(messages[: max_messages - 1])

        # Group all remaining messages into one
        remaining = messages[max_messages - 1 :]

        # Try to join remaining messages with double line breaks
        grouped_remaining = "\n\n".join(remaining)

        # If the grouped message is too long, split it more intelligently
        if len(grouped_remaining) > self.config.max_message_length:
            # Split by chunks that fit
            chunks: MutableSequence[str] = []
            current_chunk = ""

            for msg in remaining:
                test_chunk = current_chunk + ("\n\n" if current_chunk else "") + msg

                if len(test_chunk) <= self.config.max_message_length:
                    current_chunk = test_chunk
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = msg

                    # If single message is too long, hard split it
                    if len(current_chunk) > self.config.max_message_length:
                        for i in range(
                            0, len(current_chunk), self.config.max_message_length
                        ):
                            chunk = current_chunk[
                                i : i + self.config.max_message_length
                            ]
                            chunks.append(chunk)
                        current_chunk = ""

            if current_chunk:
                chunks.append(current_chunk)

            limited_messages.extend(chunks)
        else:
            limited_messages.append(grouped_remaining)

        return limited_messages

    async def _send_error_message(
        self, to: PhoneNumber, reply_to: str | None = None
    ) -> None:
        """Send error message to user."""
        if self.config.error_message is None:
            logger.debug(f"[SEND_ERROR] Error message configuration is None, not sending error message to {to}")
            return

        logger.warning(f"[SEND_ERROR] Sending error message to {to}")
        try:
            # Only quote if quote_messages is enabled
            quoted_id = reply_to if self.config.quote_messages else None
            formatted_error = self._format_whatsapp_markdown(self.config.error_message)
            await self.provider.send_text_message(
                to=to, text=formatted_error, quoted_message_id=quoted_id
            )
            logger.debug(f"[SEND_ERROR] Successfully sent error message to {to}")
        except Exception as e:
            logger.error(
                f"[SEND_ERROR_ERROR] Failed to send error message to {to}: {e}"
            )

    def _is_user_facing_error(self, error: Exception) -> bool:
        """Determine if an error should be communicated to the user."""
        # Don't show technical errors to users
        technical_errors = [
            ValueError,
            TypeError,
            AttributeError,
            KeyError,
            ImportError,
            ConnectionError,
        ]

        # Show only user-relevant errors like rate limiting
        user_relevant_errors = [
            "rate limit",
            "quota exceeded",
            "service unavailable",
        ]

        error_str = str(error).lower()

        # Don't show technical errors
        if any(isinstance(error, err_type) for err_type in technical_errors):
            return False

        # Show user-relevant errors
        if any(keyword in error_str for keyword in user_relevant_errors):
            return True

        # Default to not showing the error to users
        return False

    async def _send_rate_limit_message(self, to: PhoneNumber) -> None:
        """Send rate limit notification to user."""
        message = "You're sending messages too quickly. Please wait a moment before sending more messages."
        logger.info(f"[RATE_LIMIT] Sending rate limit message to {to}")
        try:
            formatted_message = self._format_whatsapp_markdown(message)
            await self.provider.send_text_message(to=to, text=formatted_message)
            logger.debug(f"[RATE_LIMIT] Successfully sent rate limit message to {to}")
        except Exception as e:
            logger.error(
                f"[RATE_LIMIT_ERROR] Failed to send rate limit message to {to}: {e}"
            )

    async def _apply_read_delay(self, message: WhatsAppMessage) -> None:
        """Apply human-like read delay before marking message as read.

        This method simulates the time a human would take to read and comprehend
        an incoming message. The delay is calculated based on message content length
        and includes reading time, context switching, and comprehension time.

        The delay is applied BEFORE marking the message as read, creating a realistic
        gap between message receipt and read receipt that matches human behavior.

        Behavior:
            - Skips delay if enable_human_delays is False
            - Extracts text content from message (text or media caption)
            - Calculates delay using HumanDelayCalculator
            - Applies delay using asyncio.sleep (non-blocking)
            - Logs delay start and completion
            - Handles cancellation and errors gracefully

        Args:
            message: The WhatsApp message to process. Can be text or media message.

        Raises:
            asyncio.CancelledError: Re-raised to allow proper task cancellation.
            Other exceptions are caught and logged, processing continues without delay.

        Example:
            >>> # Called automatically in handle_message() before marking as read
            >>> await self._apply_read_delay(message)
            >>> await self.provider.mark_message_as_read(message.id)
        """
        if not self.config.enable_human_delays or not self._delay_calculator:
            logger.debug("[HUMAN_DELAY] ⏱️  Read delay skipped (delays disabled)")
            return

        try:
            # Extract text content from message
            text_content = ""
            message_type = type(message).__name__
            if isinstance(message, WhatsAppTextMessage):
                text_content = message.text
            elif isinstance(message, WhatsAppMediaMessage):
                # For media messages, use caption if available
                text_content = message.caption or ""

            # Calculate read delay
            delay = self._delay_calculator.calculate_read_delay(text_content)

            # Log delay start
            logger.info(
                f"[HUMAN_DELAY] ⏱️  Starting read delay: {delay:.2f}s "
                + f"for {len(text_content)} chars (message_type={message_type}, message_id={message.id})"
            )

            # Apply delay
            await asyncio.sleep(delay)

            # Log delay completion
            logger.info(
                f"[HUMAN_DELAY] ⏱️  Read delay completed: {delay:.2f}s "
                + f"(message_id={message.id})"
            )

        except asyncio.CancelledError:
            logger.warning(
                f"[HUMAN_DELAY] ⏱️  Read delay cancelled for message {message.id}"
            )
            raise  # Re-raise to allow proper cancellation
        except Exception as e:
            logger.error(
                f"[HUMAN_DELAY] ⏱️  Error applying read delay for message {message.id}: {e}",
                exc_info=True,
            )
            # Continue without delay on error

    async def _apply_typing_delay(self, response_text: str, to: PhoneNumber) -> None:
        """Apply human-like typing delay before sending response.

        This method simulates the time a human would take to compose and type
        a response. The delay is calculated based on response content length
        and includes composition planning, typing time, and multitasking overhead.

        The delay is applied AFTER response generation but BEFORE sending the message,
        creating a realistic gap that matches human typing behavior.

        Behavior:
            - Skips delay if enable_human_delays is False
            - Calculates delay using HumanDelayCalculator based on response length
            - Optionally sends typing indicator during delay (if show_typing_during_delay is True)
            - Applies delay using asyncio.sleep (non-blocking)
            - Logs delay start and completion
            - Handles typing indicator failures gracefully
            - Handles cancellation and errors gracefully

        Args:
            response_text: The response text that will be sent to the user.
            to: The phone number of the recipient.

        Raises:
            asyncio.CancelledError: Re-raised to allow proper task cancellation.
            Other exceptions are caught and logged, processing continues without delay.

        Example:
            >>> # Called automatically in _send_response() before sending
            >>> response_text = "Hello! How can I help you?"
            >>> await self._apply_typing_delay(response_text, phone_number)
            >>> await self.provider.send_text_message(phone_number, response_text)
        """
        if not self.config.enable_human_delays or not self._delay_calculator:
            logger.debug("[HUMAN_DELAY] ⌨️  Typing delay skipped (delays disabled)")
            return

        try:
            # Calculate typing delay
            delay = self._delay_calculator.calculate_typing_delay(response_text)

            # Log delay start
            logger.info(
                f"[HUMAN_DELAY] ⌨️  Starting typing delay: {delay:.2f}s "
                + f"for {len(response_text)} chars (to={to})"
            )

            # Show typing indicator during delay if configured
            if self.config.show_typing_during_delay and self.config.typing_indicator:
                try:
                    logger.debug(
                        f"[HUMAN_DELAY] ⌨️  Sending typing indicator for {int(delay)}s to {to}"
                    )
                    # Send typing indicator for the duration of the delay
                    await self.provider.send_typing_indicator(to, int(delay))
                except Exception as indicator_error:
                    logger.warning(
                        f"[HUMAN_DELAY] ⌨️  Failed to send typing indicator during delay to {to}: "
                        + f"{indicator_error}"
                    )
                    # Continue with delay even if indicator fails

            # Apply delay
            await asyncio.sleep(delay)

            # Log delay completion
            logger.info(
                f"[HUMAN_DELAY] ⌨️  Typing delay completed: {delay:.2f}s (to={to})"
            )

        except asyncio.CancelledError:
            logger.warning(f"[HUMAN_DELAY] ⌨️  Typing delay cancelled for {to}")
            raise  # Re-raise to allow proper cancellation
        except Exception as e:
            logger.error(
                f"[HUMAN_DELAY] ⌨️  Error applying typing delay for {to}: {e}",
                exc_info=True,
            )
            # Continue without delay on error

    async def _apply_send_delay(self) -> None:
        """Apply brief delay before sending message.

        This method simulates the final review time before a human sends a message.
        The delay is a random value within configured bounds, representing the brief
        moment a human takes to review their message before hitting send.

        The delay is applied immediately BEFORE each message transmission, creating
        a small gap that adds to the natural feel of the conversation.

        Behavior:
            - Skips delay if enable_human_delays is False
            - Generates random delay within configured send delay bounds
            - Applies optional jitter if enabled
            - Applies delay using asyncio.sleep (non-blocking)
            - Logs delay start and completion
            - Handles cancellation and errors gracefully

        Raises:
            asyncio.CancelledError: Re-raised to allow proper task cancellation.
            Other exceptions are caught and logged, processing continues without delay.

        Example:
            >>> # Called automatically before each message transmission
            >>> for message_part in message_parts:
            ...     await self._apply_send_delay()
            ...     await self.provider.send_text_message(phone_number, message_part)
        """
        if not self.config.enable_human_delays or not self._delay_calculator:
            logger.debug("[HUMAN_DELAY] 📤 Send delay skipped (delays disabled)")
            return

        try:
            # Calculate send delay
            delay = self._delay_calculator.calculate_send_delay()

            # Log delay start
            logger.info(f"[HUMAN_DELAY] 📤 Starting send delay: {delay:.2f}s")

            # Apply delay
            await asyncio.sleep(delay)

            # Log delay completion
            logger.debug(f"[HUMAN_DELAY] 📤 Send delay completed: {delay:.2f}s")

        except asyncio.CancelledError:
            logger.warning("[HUMAN_DELAY] 📤 Send delay cancelled")
            raise  # Re-raise to allow proper cancellation
        except Exception as e:
            logger.error(
                f"[HUMAN_DELAY] 📤 Error applying send delay: {e}", exc_info=True
            )
            # Continue without delay on error

    async def _apply_batch_read_delay(self, messages: list[dict[str, Any]]) -> None:
        """Apply human-like read delay for batch of messages.

        This method simulates the time a human would take to read multiple messages
        in sequence. The delay accounts for reading each message individually, with
        brief pauses between messages, and applies a compression factor to simulate
        faster batch reading compared to reading messages one at a time.

        The delay is applied at the START of batch processing, before any message
        processing begins, creating a realistic gap that matches human batch reading.

        Behavior:
            - Skips delay if enable_human_delays is False
            - Extracts text content from all messages (text and media captions)
            - Calculates individual read delays for each message
            - Adds 0.5s pause between each message
            - Applies compression factor (default 0.7x for 30% faster reading)
            - Clamps to reasonable bounds (2-20 seconds suggested)
            - Applies delay using asyncio.sleep (non-blocking)
            - Logs delay start and completion with message count
            - Handles cancellation and errors gracefully

        Args:
            messages: List of message dictionaries from the batch. Each dict should
                     contain 'type' and either 'text' or 'caption' fields.

        Raises:
            asyncio.CancelledError: Re-raised to allow proper task cancellation.
            Other exceptions are caught and logged, processing continues without delay.

        Example:
            >>> # Called automatically in _process_message_batch() before processing
            >>> pending_messages = [msg1_dict, msg2_dict, msg3_dict]
            >>> await self._apply_batch_read_delay(pending_messages)
            >>> # Now process the batch...
        """
        if not self.config.enable_human_delays or not self._delay_calculator:
            logger.debug("[HUMAN_DELAY] 📚 Batch read delay skipped (delays disabled)")
            return

        try:
            # Extract text content from all messages in batch
            message_texts: list[str] = []
            total_chars = 0
            for msg in messages:
                if msg.get("type") == "WhatsAppTextMessage":
                    text = msg.get("text", "")
                    if text:
                        message_texts.append(text)
                        total_chars += len(text)
                elif msg.get("type") in [
                    "WhatsAppImageMessage",
                    "WhatsAppDocumentMessage",
                    "WhatsAppAudioMessage",
                    "WhatsAppVideoMessage",
                ]:
                    # For media messages, use caption if available
                    caption = msg.get("caption", "")
                    if caption:
                        message_texts.append(caption)
                        total_chars += len(caption)

            # Calculate batch read delay
            delay = self._delay_calculator.calculate_batch_read_delay(message_texts)

            # Log delay start
            logger.info(
                f"[HUMAN_DELAY] 📚 Starting batch read delay: {delay:.2f}s "
                + f"for {len(messages)} messages ({total_chars} total chars)"
            )

            # Apply delay
            await asyncio.sleep(delay)

            # Log delay completion
            logger.info(
                f"[HUMAN_DELAY] 📚 Batch read delay completed: {delay:.2f}s "
                + f"for {len(messages)} messages"
            )

        except asyncio.CancelledError:
            logger.warning(
                f"[HUMAN_DELAY] 📚 Batch read delay cancelled for {len(messages)} messages"
            )
            raise  # Re-raise to allow proper cancellation
        except Exception as e:
            logger.error(
                f"[HUMAN_DELAY] 📚 Error applying batch read delay for {len(messages)} messages: {e}",
                exc_info=True,
            )
            # Continue without delay on error

    def _split_message(self, text: str) -> Sequence[str]:
        """Split long message into chunks."""
        if len(text) <= self.config.max_message_length:
            return [text]

        # Split by paragraphs first
        paragraphs = text.split("\n\n")
        messages: MutableSequence[str] = []
        current = ""

        for para in paragraphs:
            if len(current) + len(para) + 2 <= self.config.max_message_length:
                if current:
                    current += "\n\n"
                current += para
            else:
                if current:
                    messages.append(current)
                current = para

        if current:
            messages.append(current)

        # Further split if any message is still too long
        final_messages = []
        for msg in messages:
            if len(msg) <= self.config.max_message_length:
                final_messages.append(msg)
            else:
                # Hard split
                for i in range(0, len(msg), self.config.max_message_length):
                    final_messages.append(msg[i : i + self.config.max_message_length])

        return final_messages

    async def _handle_message_upsert(
        self, payload: WhatsAppWebhookPayload, chat_id: ChatId | None = None
    ) -> MessageHandlingResult:
        """Handle new message event."""
        logger.info("[MESSAGE_UPSERT] ═══════════ MESSAGE UPSERT START ═══════════")
        logger.debug("[MESSAGE_UPSERT] Processing message upsert event")
        logger.info(
            f"[MESSAGE_UPSERT] Current response callbacks count: {len(self._response_callbacks)}"
        )

        # Ensure bot is running before processing messages
        if not self._running:
            logger.warning(
                "[MESSAGE_UPSERT] ⚠️ Bot is not running, skipping message processing"
            )
            return None

        # Check if this is Evolution API format
        if payload.event == "messages.upsert" and payload.data:
            logger.info("[MESSAGE_UPSERT] Processing Evolution API format")
            # Evolution API format - single message in data field
            data = payload.data

            # Skip outgoing messages
            if data.key.fromMe:
                logger.debug("[MESSAGE_UPSERT] Skipping outgoing message")
                return None

            logger.info("[MESSAGE_UPSERT] Parsing message from Evolution API data")
            # Parse message directly from data (which contains the message info)
            # When remoteJid contains @lid, use remoteJidAlt which has the correct @s.whatsapp.net format
            from_number = payload.data.key.remoteJid
            if "@lid" in payload.data.key.remoteJid and payload.data.key.remoteJidAlt:
                from_number = payload.data.key.remoteJidAlt
                logger.info(
                    f"[MESSAGE_UPSERT] 🔄 Using remoteJidAlt instead of @lid: {from_number}"
                )

            message = self._parse_evolution_message_from_data(
                data,
                from_number=from_number,
            )

            if message:
                logger.info(
                    f"[MESSAGE_UPSERT] ✅ Parsed message: {message.id} from {message.from_number}"
                )

                # Store the remoteJid for later use when sending messages back
                message.remote_jid = from_number

                logger.info(
                    f"[MESSAGE_UPSERT] About to call handle_message with {len(self._response_callbacks)} callbacks"
                )

                result = await self.handle_message(message, chat_id=chat_id)

                logger.info(
                    "[MESSAGE_UPSERT] ✅ handle_message completed. Result status: %s",
                    self._describe_message_handling_result(result),
                )
                return result
            else:
                logger.warning(
                    "[MESSAGE_UPSERT] ❌ Failed to parse message or message was skipped (empty/placeholder content)"
                )
                return None

        # Check if this is Meta API format
        elif payload.entry:
            # Meta API format - handle through provider
            logger.debug("[MESSAGE_UPSERT] Processing Meta API message upsert")
            await self.provider.validate_webhook(payload)
            return None
        else:
            logger.warning(
                "[MESSAGE_UPSERT] ⚠️ Unknown webhook format in message upsert"
            )
            return None

    async def _handle_message_update(self, payload: WhatsAppWebhookPayload) -> None:
        """Handle message update event (status changes)."""
        if payload.event == "messages.update" and payload.data:
            logger.debug(f"[MESSAGE_UPDATE] Message update: {payload.data}")
        elif payload.entry:
            logger.debug(f"[MESSAGE_UPDATE] Message update: {payload.entry}")
        else:
            logger.debug(f"[MESSAGE_UPDATE] Message update: {payload}")

    async def _handle_connection_update(self, payload: WhatsAppWebhookPayload) -> None:
        """Handle connection status update."""
        if payload.event == "connection.update" and payload.data:
            logger.info(
                f"[CONNECTION_UPDATE] WhatsApp connection update: {payload.data}"
            )
        elif payload.entry:
            logger.info(
                f"[CONNECTION_UPDATE] WhatsApp connection update: {payload.entry}"
            )
        else:
            logger.info(f"[CONNECTION_UPDATE] WhatsApp connection update: {payload}")

    def _parse_evolution_message_from_data(
        self, data: Data, from_number: str
    ) -> WhatsAppMessage | None:
        """Parse Evolution API message from webhook data field."""
        logger.debug("[PARSE_EVOLUTION] Parsing Evolution message from data")

        try:
            # Extract key information
            key = data.key
            message_id = key.id

            if not message_id or not from_number:
                logger.warning("[PARSE_EVOLUTION] Missing message ID or from_number")
                return None

            logger.debug(
                f"[PARSE_EVOLUTION] Message ID: {message_id}, From: {from_number}"
            )

            # Get message type from the data
            message_type = data.messageType or ""
            logger.info(f"[PARSE_EVOLUTION] Message type: {message_type}")

            # Handle different message types
            if message_type == "editedMessage":
                logger.info(
                    "[PARSE_EVOLUTION] Handling editedMessage - treating as text message"
                )
                # For edited messages, we might not have the content, but we should still process it
                return WhatsAppTextMessage(
                    id=message_id,
                    push_name=data.pushName or "Unknown",
                    from_number=from_number,
                    to_number=self.provider.get_instance_identifier(),
                    timestamp=datetime.fromtimestamp(
                        (data.messageTimestamp or 0) / 1000  # Convert from milliseconds
                    ),
                    text="[Message was edited]",  # Placeholder text for edited messages
                )

            # Check if there's a message field
            if data.message:
                msg_content = data.message

                # Handle text messages
                if msg_content.conversation:
                    text = msg_content.conversation
                    logger.debug(
                        f"[PARSE_EVOLUTION] Found conversation text: {text[:50] if text else 'None'}..."
                    )

                    return WhatsAppTextMessage(
                        id=message_id,
                        push_name=data.pushName or "Unknown",
                        from_number=from_number,
                        to_number=self.provider.get_instance_identifier(),
                        timestamp=datetime.fromtimestamp(
                            (data.messageTimestamp or 0)
                            / 1000  # Convert from milliseconds
                        ),
                        text=text or ".",
                    )

                # Handle extended text messages (extendedTextMessage is not in Message BaseModel, needs dict access)
                # TODO: Add extendedTextMessage to Message BaseModel if needed
                elif hasattr(msg_content, "__dict__") and msg_content.__dict__.get(
                    "extendedTextMessage"
                ):
                    extended_text_message = msg_content.__dict__.get(
                        "extendedTextMessage"
                    )
                    text = (
                        extended_text_message.get("text", "")
                        if extended_text_message
                        else ""
                    )
                    logger.debug(
                        f"[PARSE_EVOLUTION] Found extended text: {text[:50] if text else 'None'}..."
                    )

                    return WhatsAppTextMessage(
                        id=message_id,
                        from_number=from_number,
                        push_name=data.pushName or "Unknown",
                        to_number=self.provider.get_instance_identifier(),
                        timestamp=datetime.fromtimestamp(
                            (data.messageTimestamp or 0) / 1000
                        ),
                        text=text,
                    )

                # Handle image messages
                elif msg_content.imageMessage:
                    logger.debug("[PARSE_EVOLUTION] Found image message")
                    image_msg = msg_content.imageMessage
                    audio_base64 = msg_content.base64 if msg_content.base64 else None

                    return WhatsAppImageMessage(
                        id=message_id,
                        from_number=from_number,
                        push_name=data.pushName or "Unknown",
                        to_number=self.provider.get_instance_identifier(),
                        timestamp=datetime.fromtimestamp(
                            (data.messageTimestamp or 0) / 1000
                        ),
                        media_url=image_msg.url if image_msg.url else "",
                        media_mime_type=image_msg.mimetype
                        if image_msg and image_msg.mimetype
                        else "image/jpeg",
                        caption=image_msg.caption
                        if image_msg and image_msg.caption
                        else "",
                        base64_data=audio_base64
                    )

                # Handle document messages
                elif msg_content.documentMessage:
                    logger.debug("[PARSE_EVOLUTION] Found document message")
                    doc_msg = msg_content.documentMessage
                    return WhatsAppDocumentMessage(
                        id=message_id,
                        from_number=from_number,
                        push_name=data.pushName or "Unknown",
                        to_number=self.provider.get_instance_identifier(),
                        timestamp=datetime.fromtimestamp(
                            (data.messageTimestamp or 0) / 1000
                        ),
                        media_url=doc_msg.url if doc_msg else "",
                        media_mime_type=doc_msg.mimetype
                        if doc_msg and doc_msg.mimetype
                        else "application/octet-stream",
                        filename=doc_msg.fileName
                        if doc_msg and doc_msg.fileName
                        else "",
                        caption=doc_msg.caption if doc_msg and doc_msg.caption else "",
                    )

                # Handle audio messages
                elif msg_content.audioMessage:
                    logger.debug("[PARSE_EVOLUTION] Found audio message")
                    audio_msg = msg_content.audioMessage

                    # CRITICAL FIX: Check if audio comes with base64 data instead of URL
                    # This happens when WhatsApp sends audio directly in the webhook
                    audio_url = audio_msg.url if audio_msg else ""
                    audio_base64 = msg_content.base64 if msg_content.base64 else None

                    if not audio_url and audio_base64:
                        logger.info(
                            "[PARSE_EVOLUTION] 🎵 Audio message has base64 data but no URL - using base64"
                        )
                    elif audio_url:
                        logger.debug(
                            f"[PARSE_EVOLUTION] Audio message has URL: {audio_url[:50]}..."
                        )
                    else:
                        logger.warning(
                            "[PARSE_EVOLUTION] ⚠️ Audio message has neither URL nor base64 data"
                        )

                    return WhatsAppAudioMessage(
                        id=message_id,
                        from_number=from_number,
                        push_name=data.pushName or "Unknown",
                        to_number=self.provider.get_instance_identifier(),
                        timestamp=datetime.fromtimestamp(
                            (data.messageTimestamp or 0) / 1000
                        ),
                        media_url=audio_url or "",
                        media_mime_type=audio_msg.mimetype
                        if audio_msg and audio_msg.mimetype
                        else "audio/ogg",
                        base64_data=audio_base64,  # Store base64 data if available
                    )
                elif msg_content.videoMessage:
                    logger.debug("[PARSE_EVOLUTION] Found video message")
                    video_msg = msg_content.videoMessage
                    return WhatsAppVideoMessage(
                        id=message_id,
                        from_number=from_number,
                        push_name=data.pushName or "Unknown",
                        caption=video_msg.caption
                        if video_msg and video_msg.caption
                        else None,
                        to_number=self.provider.get_instance_identifier(),
                        timestamp=datetime.fromtimestamp(
                            (data.messageTimestamp or 0) / 1000
                        ),
                        media_url=video_msg.url if video_msg else "",
                        media_mime_type=video_msg.mimetype
                        if video_msg and video_msg.mimetype
                        else "",
                    )
                else:
                    logger.warning(
                        f"[PARSE_EVOLUTION] Unknown message type in content: {msg_content.__class__.__name__}"
                    )

            # If we get here and message is empty but we have messageType info, skip processing
            elif message_type and message_type != "":
                logger.info(
                    f"[PARSE_EVOLUTION] Empty message content with messageType '{message_type}' - skipping to avoid empty message processing"
                )
                # Return None to skip processing instead of creating placeholder messages
                # This prevents the agent from receiving empty/placeholder content
                return None

            logger.warning("[PARSE_EVOLUTION] No recognizable message content found")
            return None

        except Exception as e:
            logger.error(
                f"[PARSE_EVOLUTION_ERROR] Error parsing Evolution message from data: {e}",
                exc_info=True,
            )

        return None

    async def _handle_meta_webhook(
        self, payload: WhatsAppWebhookPayload
    ) -> GeneratedAssistantMessage[Any] | None:
        """Handle Meta WhatsApp Business API webhooks."""
        logger.debug("[META_WEBHOOK] Processing Meta webhook")

        try:
            if not payload.entry:
                logger.warning("[META_WEBHOOK] No entry data in Meta webhook")
                return None

            response = None

            for entry_item in payload.entry:
                changes = entry_item.get("changes", [])
                for change in changes:
                    field = change.get("field")
                    value = change.get("value", {})

                    if field == "messages":
                        logger.debug("[META_WEBHOOK] Processing messages field")
                        # Process incoming messages
                        messages = value.get("messages", [])
                        for msg_data in messages:
                            # Skip outgoing messages
                            if (
                                msg_data.get("from")
                                == self.provider.get_instance_identifier()
                            ):
                                logger.debug("[META_WEBHOOK] Skipping outgoing message")
                                continue

                            message = await self._parse_meta_message(msg_data)
                            if message:
                                logger.info(
                                    f"[META_WEBHOOK] Parsed message: {message.id} from {message.from_number}"
                                )
                                # Return the response from the last processed message
                                response = await self.handle_message(message)

            return response

        except Exception as e:
            logger.error(
                f"[META_WEBHOOK_ERROR] Error handling Meta webhook: {e}", exc_info=True
            )
            return None

    async def _parse_meta_message(
        self, msg_data: dict[str, Any]
    ) -> WhatsAppMessage | None:
        """Parse Meta API message format."""
        logger.debug("[PARSE_META] Parsing Meta API message")

        try:
            message_id = msg_data.get("id")
            from_number = msg_data.get("from")
            timestamp_str = msg_data.get("timestamp")

            if not message_id or not from_number:
                logger.warning("[PARSE_META] Missing message ID or from_number")
                return None

            logger.debug(f"[PARSE_META] Message ID: {message_id}, From: {from_number}")

            # Convert timestamp
            timestamp = (
                datetime.fromtimestamp(int(timestamp_str))
                if timestamp_str
                else datetime.now()
            )

            # Handle different message types
            msg_type = msg_data.get("type")
            logger.debug(f"[PARSE_META] Message type: {msg_type}")

            if msg_type == "text":
                text_data = msg_data.get("text", {})
                text = text_data.get("body", "")

                return WhatsAppTextMessage(
                    id=message_id,
                    from_number=from_number,
                    push_name=msg_data.get("pushName", "user"),
                    to_number=self.provider.get_instance_identifier(),
                    timestamp=timestamp,
                    text=text,
                )

            elif msg_type == "image":
                image_data = msg_data.get("image", {})

                return WhatsAppImageMessage(
                    id=message_id,
                    from_number=from_number,
                    push_name=msg_data.get("pushName", "user"),
                    to_number=self.provider.get_instance_identifier(),
                    timestamp=timestamp,
                    media_url=image_data.get("id", ""),  # Meta uses ID for media
                    media_mime_type=image_data.get("mime_type", "image/jpeg"),
                    caption=image_data.get("caption"),
                )

            elif msg_type == "document":
                doc_data = msg_data.get("document", {})

                return WhatsAppDocumentMessage(
                    id=message_id,
                    from_number=from_number,
                    push_name=msg_data.get("pushName", "user"),
                    to_number=self.provider.get_instance_identifier(),
                    timestamp=timestamp,
                    media_url=doc_data.get("id", ""),  # Meta uses ID for media
                    media_mime_type=doc_data.get(
                        "mime_type", "application/octet-stream"
                    ),
                    filename=doc_data.get("filename"),
                    caption=doc_data.get("caption"),
                )

            elif msg_type == "audio":
                audio_data = msg_data.get("audio", {})

                return WhatsAppAudioMessage(
                    id=message_id,
                    from_number=from_number,
                    push_name=msg_data.get("pushName", "user"),
                    to_number=self.provider.get_instance_identifier(),
                    timestamp=timestamp,
                    media_url=audio_data.get("id", ""),  # Meta uses ID for media
                    media_mime_type=audio_data.get("mime_type", "audio/ogg"),
                )

        except Exception as e:
            logger.error(
                f"[PARSE_META_ERROR] Error parsing Meta message: {e}", exc_info=True
            )

        return None

    async def send_message(
        self,
        to: PhoneNumber,
        message: str,
        reply_to: str | None = None,
    ) -> bool:
        """
        Send a message independently to a WhatsApp number.

        Args:
            to: The phone number to send the message to (e.g., "5511999999999")
            message: The message text to send
            reply_to: Optional message ID to reply to

        Returns:
            bool: True if message was sent successfully, False otherwise
        """
        logger.info(f"[SEND_MESSAGE] Sending independent message to {to}")

        if not self._running:
            logger.error("[SEND_MESSAGE] Bot is not running")
            return False

        if not message or not message.strip():
            logger.error("[SEND_MESSAGE] Message is empty")
            return False

        try:
            if "@" not in to:
                to += "@s.whatsapp.net"

            await self._send_response(to, message, reply_to)
            logger.info(f"[SEND_MESSAGE] ✅ Message sent successfully to {to}")
            return True
        except Exception as e:
            logger.error(
                f"[SEND_MESSAGE] ❌ Failed to send message to {to}: {e}", exc_info=True
            )
            return False

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the bot's current state."""
        return {
            "running": self._running,
            "active_batch_processors": len(self._batch_processors),
            "processing_locks": len(self._processing_locks),
            "agent_has_conversation_store": self.agent.conversation_store is not None,
            "config": {
                "message_batching_enabled": self.config.enable_message_batching,
                "spam_protection_enabled": self.config.spam_protection_enabled,
                "quote_messages": self.config.quote_messages,
                "batch_delay_seconds": self.config.batch_delay_seconds,
                "max_batch_size": self.config.max_batch_size,
                "max_messages_per_minute": self.config.max_messages_per_minute,
                "debug_mode": self.config.debug_mode,
            },
        }
