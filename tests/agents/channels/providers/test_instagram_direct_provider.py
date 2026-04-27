from __future__ import annotations

from typing import Any

import pytest

from agentle.agents.channels.providers.instagram_direct import (
    InstagramDirectConfig,
    InstagramDirectProvider,
)


def _webhook_payload() -> dict[str, Any]:
    return {
        "object": "instagram",
        "entry": [
            {
                "id": "ig-business-1",
                "time": 1777152892000,
                "messaging": [
                    {
                        "sender": {"id": "igsid-user-1"},
                        "recipient": {"id": "ig-business-1"},
                        "timestamp": 1777152892123,
                        "message": {
                            "mid": "mid.1",
                            "text": "oi",
                        },
                    },
                    {
                        "sender": {"id": "igsid-user-1"},
                        "recipient": {"id": "ig-business-1"},
                        "timestamp": 1777152893000,
                        "message": {
                            "mid": "mid.echo",
                            "is_echo": True,
                            "text": "mensagem enviada pela conta",
                        },
                    },
                    {
                        "sender": {"id": "igsid-user-1"},
                        "recipient": {"id": "ig-business-1"},
                        "timestamp": 1777152894000,
                        "read": {"mid": "mid.1"},
                    },
                ],
            }
        ],
    }


def test_webhook_to_channel_messages_parses_inbound_text() -> None:
    messages = InstagramDirectProvider.webhook_to_channel_messages(_webhook_payload())

    assert len(messages) == 1
    message = messages[0]
    assert message.id == "mid.1"
    assert message.provider == "instagram_direct"
    assert message.resource_id == "ig-business-1"
    assert message.conversation_id == "igsid-user-1"
    assert message.contact_identifier == "igsid-user-1"
    assert message.sender_display_name == "igsid-user-1"
    assert message.text == "oi"
    assert message.metadata["instagram_sender_id"] == "igsid-user-1"
    assert message.metadata["instagram_recipient_id"] == "ig-business-1"


def test_messaging_event_to_channel_message_parses_first_attachment() -> None:
    event = {
        "sender": {"id": "igsid-user-1"},
        "recipient": {"id": "ig-business-1"},
        "timestamp": "2026-04-25T21:35:00.000Z",
        "message": {
            "mid": "mid.2",
            "attachments": [
                {
                    "type": "image",
                    "payload": {"url": "https://cdn.example/image.jpg"},
                }
            ],
        },
    }

    message = InstagramDirectProvider.messaging_event_to_channel_message(
        event,
        resource_id="ig-business-1",
    )

    assert message.message_type == "media"
    assert message.media is not None
    assert message.media.media_type == "image"
    assert message.media.media_id == "https://cdn.example/image.jpg"
    assert message.media.url == "https://cdn.example/image.jpg"


@pytest.mark.asyncio
async def test_provider_session_and_send_text_use_igsid() -> None:
    provider = InstagramDirectProvider(
        InstagramDirectConfig(
            access_token="token",
            instagram_user_id="ig-business-1",
        )
    )
    captured: dict[str, Any] = {}

    async def fake_request(method, url, data=None, expected_status=200):
        captured.update(
            {
                "method": method,
                "url": url,
                "data": data,
                "expected_status": expected_status,
            }
        )
        return {"recipient_id": "igsid-user-1", "message_id": "mid.sent"}

    provider._make_request_with_retry = fake_request  # type: ignore[method-assign]

    session = await provider.get_session("igsid-user-1")
    result = await provider.send_text_message("igsid-user-1", "Olá")

    assert session is not None
    assert session.session_id == "instagram_direct:ig-business-1:igsid-user-1"
    assert result.id == "mid.sent"
    assert result.provider == "instagram_direct"
    assert captured["method"] == "POST"
    assert captured["url"] == "https://graph.instagram.com/v24.0/ig-business-1/messages"
    assert captured["data"] == {
        "recipient": {"id": "igsid-user-1"},
        "message": {"text": "Olá"},
    }

    await provider.shutdown()
