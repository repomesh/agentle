from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable, MutableMapping, MutableSequence
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from rsb.coroutines.run_sync import run_sync

from agentle.agents.agent import Agent
from agentle.agents.channels.channel_bot_config import ChannelBotConfig
from agentle.agents.channels.models.channel_message import ChannelMessage
from agentle.agents.channels.models.channel_response_base import ChannelResponseBase
from agentle.agents.channels.models.channel_session import ChannelSession
from agentle.agents.channels.providers.base import ChannelProvider
from agentle.generations.models.message_parts.file import FilePart
from agentle.generations.models.message_parts.text import TextPart
from agentle.generations.models.messages.generated_assistant_message import (
    GeneratedAssistantMessage,
)
from agentle.generations.models.messages.user_message import UserMessage
from agentle.storage.file_storage_manager import FileStorageManager
from agentle.tts.tts_provider import TtsProvider

logger = logging.getLogger(__name__)

T_Schema = TypeVar("T_Schema", bound=ChannelResponseBase)
CallbackFunction = (
    Callable[
        [str, str | None, GeneratedAssistantMessage[Any] | None, dict[str, Any]],
        None,
    ]
    | Callable[
        [str, str | None, GeneratedAssistantMessage[Any] | None, dict[str, Any]],
        Awaitable[None],
    ]
)


@dataclass
class CallbackWithContext:
    callback: CallbackFunction
    context: dict[str, Any] = field(default_factory=dict)
    scope_key: str | None = None
    persistent: bool = True


@dataclass(frozen=True)
class QueuedChannelMessageResult:
    contact_identifier: str
    chat_id: str | None = None
    pending_messages: int = 0
    processing_token: str | None = None
    status: str = "queued"
    reason: str = "message_batched"


class ChannelBot(Generic[T_Schema]):
    """Provider-neutral bot for messaging channels."""

    def __init__(
        self,
        *,
        agent: Agent[Any],
        provider: ChannelProvider,
        config: ChannelBotConfig | None = None,
        tts_provider: TtsProvider | None = None,
        file_storage_manager: FileStorageManager | None = None,
    ):
        if agent.conversation_store is None:
            raise ValueError("Agent must have a conversation_store configured.")

        self.agent = agent
        self.provider = provider
        self.config = config or ChannelBotConfig()
        self.tts_provider = tts_provider
        self.file_storage_manager = file_storage_manager
        self._running = False
        self._response_callbacks: MutableSequence[CallbackWithContext] = []
        self._batch_processors: MutableMapping[str, asyncio.Task[Any]] = {}
        self._processing_locks: MutableMapping[str, asyncio.Lock] = {}

    def start(self) -> None:
        run_sync(self.start_async)

    def stop(self) -> None:
        run_sync(self.stop_async)

    async def start_async(self) -> None:
        await self.provider.initialize()
        self._running = True

    async def stop_async(self) -> None:
        self._running = False
        for task in list(self._batch_processors.values()):
            task.cancel()
        self._batch_processors.clear()
        self._processing_locks.clear()
        await self.provider.shutdown()

    @staticmethod
    def _callback_scope_key(
        *, chat_id: str | None = None, contact_identifier: str | None = None
    ) -> str | None:
        if chat_id:
            return f"chat:{chat_id}"
        if contact_identifier:
            return f"contact:{contact_identifier}"
        return None

    def add_response_callback(
        self,
        callback: CallbackFunction,
        context: dict[str, Any] | None = None,
        *,
        chat_id: str | None = None,
        contact_identifier: str | None = None,
        persistent: bool = True,
        allow_duplicates: bool = False,
    ) -> None:
        scope_key = self._callback_scope_key(
            chat_id=chat_id,
            contact_identifier=contact_identifier,
        )
        normalized_context = dict(context or {})
        if not allow_duplicates:
            for existing in self._response_callbacks:
                if (
                    existing.callback == callback
                    and existing.context == normalized_context
                    and existing.scope_key == scope_key
                    and existing.persistent == persistent
                ):
                    return

        self._response_callbacks.append(
            CallbackWithContext(
                callback=callback,
                context=normalized_context,
                scope_key=scope_key,
                persistent=persistent,
            )
        )

    async def handle_channel_message(
        self,
        message: ChannelMessage,
        *,
        callback: CallbackFunction | list[CallbackFunction] | None = None,
        callback_context: dict[str, Any] | None = None,
        chat_id: str | None = None,
    ) -> GeneratedAssistantMessage[Any] | QueuedChannelMessageResult | None:
        if callback:
            callbacks = callback if isinstance(callback, list) else [callback]
            for callback_function in callbacks:
                self.add_response_callback(
                    callback_function,
                    context=callback_context,
                    contact_identifier=message.contact_identifier,
                    persistent=False,
                )

        contact_identifier = message.contact_identifier
        session = await self.provider.get_session(contact_identifier)
        if session is None:
            logger.error("Failed to create channel session for %s", contact_identifier)
            return None

        if self.config.spam_protection_enabled and not session.update_rate_limiting(
            self.config.max_messages_per_minute,
            self.config.rate_limit_cooldown_seconds,
        ):
            await self.provider.update_session(session)
            return None

        if self.config.auto_read_messages and self.provider.capabilities.supports_read_receipt:
            await self.provider.mark_message_as_read(message.id)

        effective_chat_id = chat_id or message.conversation_id
        session.context_data["custom_chat_id"] = effective_chat_id

        if self.config.enable_message_batching:
            return await self._handle_message_with_batching(
                message,
                session,
                chat_id=effective_chat_id,
            )

        return await self._process_single_message(
            message,
            session,
            chat_id=effective_chat_id,
        )

    async def _handle_message_with_batching(
        self,
        message: ChannelMessage,
        session: ChannelSession,
        *,
        chat_id: str,
    ) -> QueuedChannelMessageResult | None:
        contact_identifier = message.contact_identifier
        lock = self._processing_locks.setdefault(contact_identifier, asyncio.Lock())

        async with lock:
            current_session = await self.provider.get_session(contact_identifier)
            if current_session is None:
                return None

            current_session.context_data["custom_chat_id"] = chat_id
            current_session.add_pending_message(self._message_to_pending_dict(message))

            token = current_session.processing_token
            if not current_session.is_processing:
                token = current_session.start_batch_processing(
                    self.config.max_batch_timeout_seconds
                )
                self._batch_processors[contact_identifier] = asyncio.create_task(
                    self._batch_processor(contact_identifier, token)
                )

            await self.provider.update_session(current_session)
            return QueuedChannelMessageResult(
                contact_identifier=contact_identifier,
                chat_id=chat_id,
                pending_messages=len(current_session.pending_messages),
                processing_token=token,
            )

    async def _batch_processor(self, contact_identifier: str, token: str) -> None:
        try:
            while self._running:
                session = await self.provider.get_session(contact_identifier)
                if session is None or not session.is_processing:
                    return

                if session.processing_token != token:
                    return

                if session.should_process_batch(
                    self.config.batch_delay_seconds,
                    self.config.max_batch_timeout_seconds,
                ):
                    await self._process_message_batch(session, token)
                    return

                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Error processing channel message batch")
            session = await self.provider.get_session(contact_identifier)
            if session is not None:
                session.finish_batch_processing(token)
                await self.provider.update_session(session)
                await self._call_response_callbacks(
                    contact_identifier,
                    None,
                    input_tokens=0,
                    output_tokens=0,
                    chat_id=session.context_data.get("custom_chat_id"),
                    processing_status="failed",
                )
        finally:
            self._batch_processors.pop(contact_identifier, None)

    async def _process_message_batch(
        self,
        session: ChannelSession,
        token: str,
    ) -> GeneratedAssistantMessage[Any] | None:
        chat_id = str(session.context_data.get("custom_chat_id") or session.contact_identifier)
        pending_messages = session.clear_pending_messages()
        messages = [ChannelMessage.model_validate(item) for item in pending_messages]

        response: GeneratedAssistantMessage[Any] | None = None
        input_tokens = 0
        output_tokens = 0
        try:
            agent_input = await self._messages_to_user_input(messages)
            await self._send_typing_indicator_if_supported(session.contact_identifier)
            response, input_tokens, output_tokens = await self._process_with_agent(
                agent_input,
                session,
                chat_id=chat_id,
            )
            if response:
                reply_to = (
                    messages[-1].id
                    if self.config.quote_messages
                    and self.provider.capabilities.supports_quoting
                    else None
                )
                await self._send_response(session.contact_identifier, response, reply_to)
            session.message_count += len(messages)
            session.finish_batch_processing(token)
            await self.provider.update_session(session)
            await self._call_response_callbacks(
                session.contact_identifier,
                response,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                chat_id=chat_id,
                processing_status="completed",
            )
            return response
        except Exception:
            logger.exception("Failed to process channel message batch")
            session.finish_batch_processing(token)
            await self.provider.update_session(session)
            await self._call_response_callbacks(
                session.contact_identifier,
                None,
                input_tokens=0,
                output_tokens=0,
                chat_id=chat_id,
                processing_status="failed",
            )
            return None

    async def _process_single_message(
        self,
        message: ChannelMessage,
        session: ChannelSession,
        *,
        chat_id: str,
    ) -> GeneratedAssistantMessage[Any] | None:
        contact_identifier = message.contact_identifier
        try:
            agent_input = await self._messages_to_user_input([message])
            await self._send_typing_indicator_if_supported(contact_identifier)
            response, input_tokens, output_tokens = await self._process_with_agent(
                agent_input,
                session,
                chat_id=chat_id,
            )
            if response:
                reply_to = (
                    message.id
                    if self.config.quote_messages
                    and self.provider.capabilities.supports_quoting
                    else None
                )
                await self._send_response(contact_identifier, response, reply_to)
            session.message_count += 1
            await self.provider.update_session(session)
            await self._call_response_callbacks(
                contact_identifier,
                response,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                chat_id=chat_id,
                processing_status="completed",
            )
            return response
        except Exception:
            logger.exception("Failed to process channel message")
            await self._call_response_callbacks(
                contact_identifier,
                None,
                input_tokens=0,
                output_tokens=0,
                chat_id=chat_id,
                processing_status="failed",
            )
            return None

    async def _messages_to_user_input(self, messages: list[ChannelMessage]) -> UserMessage:
        parts: MutableSequence[TextPart | FilePart] = []
        for index, message in enumerate(messages):
            if index > 0:
                parts.append(TextPart(text="\n\n"))

            if message.text:
                parts.append(TextPart(text=message.text))

            if message.media:
                try:
                    if message.media.base64_data:
                        parts.append(
                            FilePart(
                                data=message.media.base64_data,
                                mime_type=message.media.mime_type or "application/octet-stream",
                            )
                        )
                    elif message.media.media_id:
                        media = await self.provider.download_media(message.media.media_id)
                        parts.append(FilePart(data=media.data, mime_type=media.mime_type))
                    if message.media.caption:
                        parts.append(TextPart(text=f"Caption: {message.media.caption}"))
                except Exception:
                    logger.exception("Failed to download channel media")
                    parts.append(TextPart(text="[Media file - failed to download]"))

        if not parts:
            parts.append(TextPart(text=""))

        display_name = messages[0].sender_display_name if messages else None
        return UserMessage.create_named(parts=parts, name=display_name)

    async def _process_with_agent(
        self,
        agent_input: UserMessage,
        session: ChannelSession,
        *,
        chat_id: str,
    ) -> tuple[GeneratedAssistantMessage[Any], int, int]:
        async with self.agent.start_mcp_servers_async():
            result = await self.agent.run_async(agent_input, chat_id=chat_id)

        if not result.generation:
            raise RuntimeError("Agent generation returned no message.")

        return (
            result.generation.message,
            int(result.input_tokens or 0),
            int(result.output_tokens or 0),
        )

    async def _send_response(
        self,
        recipient: str,
        response: GeneratedAssistantMessage[Any] | str,
        reply_to: str | None = None,
    ) -> None:
        response_text = response if isinstance(response, str) else response.text
        if not isinstance(response, str) and response.parsed is not None:
            response_text = getattr(response.parsed, "response", response.text)

        for part in self._split_response(str(response_text)):
            await self.provider.send_text_message(recipient, part, reply_to)
            reply_to = None

    async def _send_typing_indicator_if_supported(self, recipient: str) -> None:
        if (
            not self.config.typing_indicator
            or not self.provider.capabilities.supports_typing_indicator
        ):
            return
        try:
            await self.provider.send_typing_indicator(
                recipient,
                self.config.typing_duration,
            )
        except Exception:
            logger.exception("Failed to send channel typing indicator")

    def _split_response(self, text: str) -> list[str]:
        limit = max(1, int(self.config.max_message_length or 4096))
        if len(text) <= limit:
            return [text]

        parts = [text[index : index + limit] for index in range(0, len(text), limit)]
        return parts[: max(1, self.config.max_split_messages)]

    async def _call_response_callbacks(
        self,
        contact_identifier: str,
        response: GeneratedAssistantMessage[Any] | None,
        input_tokens: int,
        output_tokens: int,
        *,
        chat_id: str | None,
        processing_status: str,
    ) -> None:
        scope_key = self._callback_scope_key(
            chat_id=chat_id,
            contact_identifier=contact_identifier,
        )
        contact_scope_key = self._callback_scope_key(contact_identifier=contact_identifier)
        callbacks = [
            callback
            for callback in self._response_callbacks
            if callback.scope_key is None
            or callback.scope_key == scope_key
            or callback.scope_key == contact_scope_key
        ]
        callbacks_to_remove: list[CallbackWithContext] = []

        for callback in callbacks:
            context = dict(callback.context)
            context.update(
                {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "processing_status": processing_status,
                }
            )
            try:
                if inspect.iscoroutinefunction(callback.callback):
                    await callback.callback(contact_identifier, chat_id, response, context)
                else:
                    callback.callback(contact_identifier, chat_id, response, context)
            except Exception:
                logger.exception("Channel response callback failed")
            finally:
                if not callback.persistent:
                    callbacks_to_remove.append(callback)

        if callbacks_to_remove:
            self._response_callbacks = [
                existing
                for existing in self._response_callbacks
                if existing not in callbacks_to_remove
            ]

    @staticmethod
    def _message_to_pending_dict(message: ChannelMessage) -> dict[str, Any]:
        return message.model_dump(mode="python")
