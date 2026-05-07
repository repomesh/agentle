from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import pytest

from agentle.agents.channels.channel_bot import ChannelBot, QueuedChannelMessageResult
from agentle.agents.channels.channel_bot_config import ChannelBotConfig
from agentle.agents.channels.models.channel_capabilities import ChannelCapabilities
from agentle.agents.channels.models.channel_message import ChannelMessage
from agentle.agents.channels.models.channel_session import ChannelSession
from agentle.generations.models.message_parts.text import TextPart
from agentle.generations.models.messages.generated_assistant_message import (
    GeneratedAssistantMessage,
)


class SlowFakeAgent:
    def __init__(self) -> None:
        self.conversation_store = object()
        self.calls: list[str] = []
        self.first_call_started = asyncio.Event()

    @asynccontextmanager
    async def start_mcp_servers_async(self):
        yield

    async def run_async(
        self,
        agent_input: Any,
        *,
        chat_id: str | None = None,
        **_: Any,
    ) -> Any:
        del chat_id
        text = "".join(
            str(part.text)
            for part in getattr(agent_input, "parts", ())
            if isinstance(part, TextPart)
        )
        self.calls.append(text)
        self.first_call_started.set()
        await asyncio.sleep(0.05)

        message = GeneratedAssistantMessage(
            parts=[TextPart(text=f"reply:{len(self.calls)}")],
            parsed=None,
        )
        return SimpleNamespace(
            generation=SimpleNamespace(message=message),
            input_tokens=1,
            output_tokens=1,
        )


class MemoryChannelProvider:
    capabilities = ChannelCapabilities()

    def __init__(self) -> None:
        self.sessions: dict[str, ChannelSession] = {}
        self.sent_texts: list[str] = []

    def get_resource_identifier(self) -> str:
        return "memory"

    async def initialize(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None

    async def send_text_message(
        self,
        to: str,
        text: str,
        quoted_message_id: str | None = None,
    ) -> None:
        del to, quoted_message_id
        self.sent_texts.append(text)

    async def send_media_message(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    async def send_typing_indicator(self, to: str, duration: int = 3) -> None:
        del to, duration

    async def send_recording_indicator(self, to: str, duration: int = 3) -> None:
        del to, duration

    async def mark_message_as_read(self, message_id: str) -> None:
        del message_id

    async def get_session(self, contact_identifier: str) -> ChannelSession | None:
        session = self.sessions.get(contact_identifier)
        if session is None:
            session = ChannelSession(
                session_id=f"memory:{contact_identifier}",
                contact_identifier=contact_identifier,
            )
            self.sessions[contact_identifier] = session
        return session

    async def update_session(self, session: ChannelSession) -> None:
        self.sessions[session.contact_identifier] = session

    async def download_media(self, media_id: str) -> None:
        del media_id
        return None


def make_message(message_id: str, text: str) -> ChannelMessage:
    return ChannelMessage(
        id=message_id,
        provider="memory",
        resource_id="memory",
        conversation_id="chat:contact-1",
        sender_id="contact-1",
        text=text,
    )


async def wait_for(predicate, timeout_seconds: float = 2.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("Timed out waiting for condition")


@pytest.mark.asyncio
async def test_pending_messages_added_during_batch_start_followup_batch() -> None:
    provider = MemoryChannelProvider()
    agent = SlowFakeAgent()
    bot = ChannelBot(
        agent=agent,
        provider=provider,
        config=ChannelBotConfig(
            enable_message_batching=True,
            batch_delay_seconds=0.01,
            max_batch_timeout_seconds=1.0,
            spam_protection_enabled=False,
            auto_read_messages=False,
            typing_indicator=False,
        ),
    )

    await bot.start_async()
    try:
        first = await bot.handle_channel_message(make_message("m1", "first"))
        assert isinstance(first, QueuedChannelMessageResult)

        await asyncio.wait_for(agent.first_call_started.wait(), timeout=1)

        second = await bot.handle_channel_message(make_message("m2", "second"))
        assert isinstance(second, QueuedChannelMessageResult)

        await wait_for(lambda: len(agent.calls) == 2)
        await wait_for(lambda: not bot._batch_processors)

        assert agent.calls == ["first", "second"]
        assert provider.sent_texts == ["reply:1", "reply:2"]

        session = await provider.get_session("contact-1")
        assert session is not None
        assert session.message_count == 2
        assert not session.pending_messages
        assert not session.is_processing
        assert session.processing_token is None
    finally:
        await bot.stop_async()
