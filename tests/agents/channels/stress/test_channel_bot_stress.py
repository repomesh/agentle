from __future__ import annotations

import asyncio
import os
from collections import Counter
from datetime import datetime, timedelta
import pytest

from agentle.agents.channels.channel_bot import (
    ChannelBot,
    QueuedChannelMessageResult,
)
from agentle.agents.channels.channel_bot_config import ChannelBotConfig
from agentle.agents.channels.models.channel_media import ChannelMedia
from agentle.agents.channels.models.channel_message import ChannelMessage
from agentle.agents.channels.models.downloaded_channel_media import (
    DownloadedChannelMedia,
)
from agentle.generations.models.message_parts.file import FilePart

from .conftest import (
    PROVIDER_NAMES,
    CallbackRecord,
    FakeStressAgent,
    assert_no_exceptions,
    collect_callbacks,
    make_offline_provider,
    wait_for,
)


pytestmark = [
    pytest.mark.stress,
    pytest.mark.skipif(
        os.environ.get("AGENTLE_RUN_CHANNEL_STRESS") != "1",
        reason="Set AGENTLE_RUN_CHANNEL_STRESS=1 to run channel stress tests.",
    ),
]


WORKLOADS = (
    pytest.param(1, 100, id="1x100"),
    pytest.param(16, 25, id="16x25"),
    pytest.param(64, 16, id="64x16"),
)


@pytest.mark.parametrize("provider_name", PROVIDER_NAMES)
@pytest.mark.parametrize(("contact_count", "messages_per_contact"), WORKLOADS)
async def test_concurrent_channel_messages_without_batching(
    provider_name: str,
    contact_count: int,
    messages_per_contact: int,
) -> None:
    provider, probe, make_message = make_offline_provider(provider_name)
    agent = FakeStressAgent(delay_seconds=0.001)
    callbacks: list[CallbackRecord] = []
    bot = ChannelBot(
        agent=agent,
        provider=provider,
        config=ChannelBotConfig(
            enable_message_batching=False,
            spam_protection_enabled=False,
            auto_read_messages=False,
            typing_indicator=True,
        ),
    )
    bot.add_response_callback(collect_callbacks(callbacks))

    messages = [
        make_message(contact_index, message_index, None)
        for contact_index in range(contact_count)
        for message_index in range(messages_per_contact)
    ]

    await bot.start_async()
    try:
        results = await asyncio.gather(
            *(bot.handle_channel_message(message) for message in messages),
            return_exceptions=True,
        )

        await assert_no_exceptions(results)
        assert all(result is not None for result in results)
        assert len(agent.calls) == len(messages)
        assert len(callbacks) == len(messages)
        assert len(probe.text_sends) == len(messages)
        assert agent.max_active_calls > 1

        expected_by_chat = Counter(message.conversation_id for message in messages)
        actual_by_chat = Counter(call.chat_id for call in agent.calls)
        assert actual_by_chat == expected_by_chat

        expected_by_contact = Counter(message.contact_identifier for message in messages)
        for contact_identifier, expected_count in expected_by_contact.items():
            session = await provider.get_session(contact_identifier)
            assert session is not None
            assert session.message_count == expected_count

        assert {
            record.context["processing_status"] for record in callbacks
        } == {"completed"}
        assert all(
            send.text.startswith(f"reply:{send.payload.get('conversation', {}).get('id')}")
            or send.text.startswith("reply:")
            for send in probe.text_sends
        )
    finally:
        await bot.stop_async()


@pytest.mark.parametrize("provider_name", PROVIDER_NAMES)
async def test_same_contact_concurrency_preserves_session_count(
    provider_name: str,
) -> None:
    provider, probe, make_message = make_offline_provider(provider_name)
    contact_index = 7
    message_count = 250
    agent = FakeStressAgent(delay_seconds=0.001)
    callbacks: list[CallbackRecord] = []
    bot = ChannelBot(
        agent=agent,
        provider=provider,
        config=ChannelBotConfig(
            enable_message_batching=False,
            spam_protection_enabled=False,
            auto_read_messages=False,
        ),
    )
    bot.add_response_callback(collect_callbacks(callbacks))
    messages = [
        make_message(contact_index, message_index, None)
        for message_index in range(message_count)
    ]

    await bot.start_async()
    try:
        results = await asyncio.gather(
            *(bot.handle_channel_message(message) for message in messages),
            return_exceptions=True,
        )

        await assert_no_exceptions(results)
        assert all(result is not None for result in results)
        assert len(agent.calls) == message_count
        assert len(callbacks) == message_count
        assert len(probe.text_sends) == message_count

        session = await provider.get_session(messages[0].contact_identifier)
        assert session is not None
        assert session.message_count == message_count
        assert not session.pending_messages
        assert not session.is_processing
    finally:
        await bot.stop_async()


@pytest.mark.parametrize("provider_name", PROVIDER_NAMES)
async def test_batching_under_parallel_bursts_isolated_by_contact(
    provider_name: str,
) -> None:
    provider, probe, make_message = make_offline_provider(provider_name)
    contact_count = 8
    messages_per_contact = 12
    agent = FakeStressAgent(delay_seconds=0.001)
    callbacks: list[CallbackRecord] = []
    bot = ChannelBot(
        agent=agent,
        provider=provider,
        config=ChannelBotConfig(
            enable_message_batching=True,
            batch_delay_seconds=0.01,
            max_batch_timeout_seconds=1.0,
            spam_protection_enabled=False,
            auto_read_messages=False,
            typing_indicator=True,
        ),
    )
    bot.add_response_callback(collect_callbacks(callbacks))

    messages = [
        make_message(
            contact_index,
            message_index,
            f"burst {contact_index}:{message_index}",
        )
        for contact_index in range(contact_count)
        for message_index in range(messages_per_contact)
    ]

    await bot.start_async()
    try:
        results = await asyncio.gather(
            *(bot.handle_channel_message(message) for message in messages),
            return_exceptions=True,
        )

        await assert_no_exceptions(results)
        assert all(isinstance(result, QueuedChannelMessageResult) for result in results)

        await wait_for(lambda: len(callbacks) == contact_count, timeout_seconds=5)
        await wait_for(lambda: not bot._batch_processors, timeout_seconds=2)

        assert len(agent.calls) == contact_count
        assert len(probe.text_sends) == contact_count

        calls_by_chat = {call.chat_id: call for call in agent.calls}
        for contact_index in range(contact_count):
            conversation_id = messages[contact_index * messages_per_contact].conversation_id
            call = calls_by_chat[conversation_id]
            for message_index in range(messages_per_contact):
                assert f"burst {contact_index}:{message_index}" in call.text
            for other_contact in range(contact_count):
                if other_contact != contact_index:
                    assert f"burst {other_contact}:" not in call.text

        for contact_identifier in {
            message.contact_identifier for message in messages
        }:
            session = await provider.get_session(contact_identifier)
            assert session is not None
            assert session.message_count == messages_per_contact
            assert not session.pending_messages
            assert not session.is_processing
            assert session.processing_token is None
    finally:
        await bot.stop_async()


@pytest.mark.parametrize("provider_name", PROVIDER_NAMES)
async def test_rate_limit_blocks_excess_messages_and_recovers_after_cooldown(
    provider_name: str,
) -> None:
    provider, probe, make_message = make_offline_provider(provider_name)
    agent = FakeStressAgent(delay_seconds=0)
    callbacks: list[CallbackRecord] = []
    bot = ChannelBot(
        agent=agent,
        provider=provider,
        config=ChannelBotConfig(
            enable_message_batching=False,
            spam_protection_enabled=True,
            max_messages_per_minute=3,
            rate_limit_cooldown_seconds=60,
            auto_read_messages=False,
            typing_indicator=False,
        ),
    )
    bot.add_response_callback(collect_callbacks(callbacks))

    await bot.start_async()
    try:
        first_results = [
            await bot.handle_channel_message(make_message(0, index, None))
            for index in range(5)
        ]

        assert sum(result is not None for result in first_results) == 3
        assert first_results[3] is None
        assert first_results[4] is None
        assert len(agent.calls) == 3
        assert len(callbacks) == 3
        assert len(probe.text_sends) == 3

        session = await provider.get_session(make_message(0, 0, None).contact_identifier)
        assert session is not None
        assert session.is_rate_limited
        assert session.rate_limit_until is not None

        session.rate_limit_until = datetime.now() - timedelta(seconds=1)
        await provider.update_session(session)

        recovered = await bot.handle_channel_message(make_message(0, 99, None))
        assert recovered is not None
        assert len(agent.calls) == 4
        assert len(probe.text_sends) == 4

        recovered_session = await provider.get_session(
            make_message(0, 0, None).contact_identifier
        )
        assert recovered_session is not None
        assert not recovered_session.is_rate_limited
    finally:
        await bot.stop_async()


@pytest.mark.parametrize("provider_name", PROVIDER_NAMES)
async def test_provider_send_and_parse_contracts_offline(provider_name: str) -> None:
    provider, probe, make_message = make_offline_provider(provider_name)
    messages = [make_message(index % 5, index, f"provider contract {index}") for index in range(30)]

    await provider.initialize()
    try:
        sessions = await asyncio.gather(
            *(provider.get_session(message.contact_identifier) for message in messages)
        )
        results = await asyncio.gather(
            *(
                provider.send_text_message(
                    message.contact_identifier,
                    f"outbound {index}",
                    quoted_message_id=message.id,
                )
                for index, message in enumerate(messages)
            )
        )

        assert all(session is not None for session in sessions)
        assert all(result.provider == provider_name for result in results)
        assert len(probe.text_sends) == len(messages)

        if provider_name == "whatsapp_cloud":
            assert all("/v24.0/phone-number-1/messages" in send.url for send in probe.text_sends)
            assert all(send.quoted_message_id is not None for send in probe.text_sends)
            assert messages[0].sender_display_name == "WhatsApp 0"
        elif provider_name == "instagram_direct":
            assert all("/v24.0/ig-business-1/messages" in send.url for send in probe.text_sends)
            assert all(send.quoted_message_id is None for send in probe.text_sends)
            assert messages[0].text == "provider contract 0"
        else:
            assert all("/v3/conversations/" in send.url for send in probe.text_sends)
            assert all(send.quoted_message_id is not None for send in probe.text_sends)
            assert messages[0].text == "provider contract 0"
            assert messages[0].metadata["teams_reference"]["conversation"]["id"]
    finally:
        await provider.shutdown()


@pytest.mark.parametrize("provider_name", PROVIDER_NAMES)
async def test_agent_and_send_failures_report_failed_callbacks(
    provider_name: str,
) -> None:
    provider, probe, make_message = make_offline_provider(provider_name)
    failing_message = make_message(1, 1, "agent should fail")
    agent = FakeStressAgent(
        delay_seconds=0,
        fail_for_chat_ids={failing_message.conversation_id},
    )
    callbacks: list[CallbackRecord] = []
    bot = ChannelBot(
        agent=agent,
        provider=provider,
        config=ChannelBotConfig(
            enable_message_batching=False,
            spam_protection_enabled=False,
            auto_read_messages=False,
            typing_indicator=False,
        ),
    )
    bot.add_response_callback(collect_callbacks(callbacks))

    await bot.start_async()
    try:
        assert await bot.handle_channel_message(failing_message) is None
        assert callbacks[-1].context["processing_status"] == "failed"
        assert callbacks[-1].response is None

        probe.fail_text_sends = 1
        send_failure_message = make_message(2, 1, "send should fail")
        assert await bot.handle_channel_message(send_failure_message) is None
        assert callbacks[-1].context["processing_status"] == "failed"
        assert callbacks[-1].response is None

        session = await provider.get_session(send_failure_message.contact_identifier)
        assert session is not None
        assert not session.is_processing
    finally:
        await bot.stop_async()


@pytest.mark.parametrize("provider_name", PROVIDER_NAMES)
async def test_batch_failure_finishes_processing_and_stop_cleans_tasks(
    provider_name: str,
) -> None:
    provider, _, make_message = make_offline_provider(provider_name)
    failing_message = make_message(3, 0, "batch should fail")
    agent = FakeStressAgent(
        delay_seconds=0,
        fail_for_chat_ids={failing_message.conversation_id},
    )
    callbacks: list[CallbackRecord] = []
    bot = ChannelBot(
        agent=agent,
        provider=provider,
        config=ChannelBotConfig(
            enable_message_batching=True,
            batch_delay_seconds=0.01,
            max_batch_timeout_seconds=5.0,
            spam_protection_enabled=False,
            auto_read_messages=False,
            typing_indicator=False,
        ),
    )
    bot.add_response_callback(collect_callbacks(callbacks))

    await bot.start_async()
    try:
        queued = await bot.handle_channel_message(failing_message)
        assert isinstance(queued, QueuedChannelMessageResult)
        assert bot._batch_processors

        await wait_for(lambda: len(callbacks) == 1, timeout_seconds=5)
        await wait_for(lambda: not bot._batch_processors, timeout_seconds=2)

        assert callbacks[0].context["processing_status"] == "failed"
        session = await provider.get_session(failing_message.contact_identifier)
        assert session is not None
        assert not session.pending_messages
        assert not session.is_processing
        assert session.processing_token is None

        held_message = make_message(4, 0, "hold until stop")
        held = await bot.handle_channel_message(held_message)
        assert isinstance(held, QueuedChannelMessageResult)
        tasks = list(bot._batch_processors.values())
        assert tasks

        await bot.stop_async()
        await asyncio.sleep(0)

        assert not bot._batch_processors
        assert not bot._processing_locks
        assert all(task.cancelled() or task.done() for task in tasks)
    finally:
        if bot._running:
            await bot.stop_async()


async def test_media_batch_and_split_response_use_file_parts_and_send_chunks() -> None:
    provider, probe, _ = make_offline_provider("whatsapp_cloud")

    async def download_media(media_id: str) -> DownloadedChannelMedia:
        assert media_id == "download-media-1"
        return DownloadedChannelMedia(data=b"downloaded-bytes", mime_type="text/plain")

    provider.download_media = download_media

    agent = FakeStressAgent(
        delay_seconds=0,
        response_factory=lambda _: "abcdefghijklmnopqrstuvwxyz",
    )
    callbacks: list[CallbackRecord] = []
    bot = ChannelBot(
        agent=agent,
        provider=provider,
        config=ChannelBotConfig(
            enable_message_batching=True,
            batch_delay_seconds=0.01,
            max_batch_timeout_seconds=1.0,
            spam_protection_enabled=False,
            auto_read_messages=False,
            max_message_length=10,
            max_split_messages=3,
        ),
    )
    bot.add_response_callback(collect_callbacks(callbacks))

    inline_message = ChannelMessage(
        id="media-inline",
        provider="whatsapp_cloud",
        resource_id="phone-number-1",
        conversation_id="media-contact",
        sender_id="media-contact",
        sender_display_name="Media Contact",
        message_type="media",
        text="inline text",
        media=ChannelMedia(
            media_type="document",
            base64_data="aW5saW5lLWJ5dGVz",
            mime_type="text/plain",
            caption="inline caption",
        ),
    )
    downloaded_message = ChannelMessage(
        id="media-downloaded",
        provider="whatsapp_cloud",
        resource_id="phone-number-1",
        conversation_id="media-contact",
        sender_id="media-contact",
        sender_display_name="Media Contact",
        message_type="media",
        text="download text",
        media=ChannelMedia(
            media_type="document",
            media_id="download-media-1",
            mime_type="text/plain",
            caption="download caption",
        ),
    )

    await bot.start_async()
    try:
        await asyncio.gather(
            bot.handle_channel_message(inline_message),
            bot.handle_channel_message(downloaded_message),
        )
        await wait_for(lambda: len(callbacks) == 1, timeout_seconds=5)

        assert len(agent.calls) == 1
        file_parts = [part for part in agent.calls[0].parts if isinstance(part, FilePart)]
        assert len(file_parts) == 2
        assert file_parts[0].data == "aW5saW5lLWJ5dGVz"
        assert file_parts[1].data == b"downloaded-bytes"
        assert "Caption: inline caption" in agent.calls[0].text
        assert "Caption: download caption" in agent.calls[0].text

        assert [send.text for send in probe.text_sends] == [
            "abcdefghij",
            "klmnopqrst",
            "uvwxyz",
        ]
    finally:
        await bot.stop_async()
