from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from agentle.agents.channels.channel_bot import ChannelBot
from agentle.agents.channels.models.channel_capabilities import ChannelCapabilities
from agentle.agents.channels.models.channel_message import ChannelMessage
from agentle.generations.models.message_parts.text import TextPart


class MinimalProvider:
    capabilities = ChannelCapabilities()

    def get_resource_identifier(self) -> str:
        return "memory"

    async def download_media(self, media_id: str) -> Any:
        del media_id
        return None


def _make_message(text: str) -> ChannelMessage:
    return ChannelMessage(
        id="m1",
        provider="memory",
        resource_id="memory",
        conversation_id="chat:contact-1",
        sender_id="contact-1",
        text=text,
    )


def _make_bot(runtime_context_provider: Any) -> ChannelBot[Any]:
    agent = SimpleNamespace(conversation_store=object())
    return ChannelBot(
        agent=agent,  # type: ignore[arg-type]
        provider=MinimalProvider(),  # type: ignore[arg-type]
        runtime_context_provider=runtime_context_provider,
    )


def _joined_text(user_message: Any) -> str:
    return "".join(
        str(p.text) for p in user_message.parts if isinstance(p, TextPart)
    )


@pytest.mark.asyncio
async def test_runtime_context_is_prepended_to_user_turn() -> None:
    bot = _make_bot(lambda _messages: "CONTEXTO-RUNTIME")

    user_message = await bot._messages_to_user_input([_make_message("oi")])

    text_parts = [p for p in user_message.parts if isinstance(p, TextPart)]
    assert text_parts[0].text == "CONTEXTO-RUNTIME"
    assert "oi" in _joined_text(user_message)


@pytest.mark.asyncio
async def test_async_runtime_context_provider_is_awaited() -> None:
    async def provider(_messages: Any) -> str:
        return "CTX-ASYNC"

    bot = _make_bot(provider)

    user_message = await bot._messages_to_user_input([_make_message("oi")])

    assert any(
        isinstance(p, TextPart) and p.text == "CTX-ASYNC" for p in user_message.parts
    )


@pytest.mark.asyncio
async def test_no_provider_leaves_user_turn_unchanged() -> None:
    bot = _make_bot(None)

    user_message = await bot._messages_to_user_input([_make_message("oi")])

    assert _joined_text(user_message).strip() == "oi"


@pytest.mark.asyncio
async def test_empty_runtime_context_is_ignored() -> None:
    bot = _make_bot(lambda _messages: "   ")

    user_message = await bot._messages_to_user_input([_make_message("oi")])

    assert _joined_text(user_message).strip() == "oi"


@pytest.mark.asyncio
async def test_provider_exception_is_swallowed() -> None:
    def boom(_messages: Any) -> str:
        raise RuntimeError("provider failed")

    bot = _make_bot(boom)

    user_message = await bot._messages_to_user_input([_make_message("oi")])

    assert _joined_text(user_message).strip() == "oi"
