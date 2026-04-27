from __future__ import annotations

import pytest

from agentle.agents.channels.providers.microsoft_teams import (
    MicrosoftTeamsConfig,
    MicrosoftTeamsProvider,
)


def _message_activity() -> dict:
    return {
        "type": "message",
        "id": "activity-1",
        "timestamp": "2026-04-25T21:35:00.000Z",
        "serviceUrl": "https://smba.trafficmanager.net/amer/",
        "channelId": "msteams",
        "from": {"id": "user-1", "name": "Arthur"},
        "recipient": {"id": "bot-1", "name": "EaiZap"},
        "conversation": {"id": "conv-1", "conversationType": "personal"},
        "text": "<at>EaiZap</at> oi",
        "entities": [
            {
                "type": "mention",
                "text": "<at>EaiZap</at>",
                "mentioned": {"id": "bot-1", "name": "EaiZap"},
            }
        ],
        "channelData": {"tenant": {"id": "tenant-1"}},
    }


def test_activity_to_channel_message_uses_conversation_user_contact_by_default() -> None:
    message = MicrosoftTeamsProvider.activity_to_channel_message(
        _message_activity(),
        app_id="bot-1",
    )

    assert message.provider == "microsoft_teams"
    assert message.resource_id == "bot-1"
    assert message.conversation_id == "conv-1"
    assert message.sender_id == "conv-1:user-1"
    assert message.contact_identifier == "conv-1:user-1"
    assert message.sender_display_name == "Arthur"
    assert message.text == "oi"
    assert message.metadata["teams_user_id"] == "user-1"
    assert message.metadata["teams_tenant_id"] == "tenant-1"


def test_activity_to_channel_message_supports_other_contact_strategies() -> None:
    by_user = MicrosoftTeamsProvider.activity_to_channel_message(
        _message_activity(),
        app_id="bot-1",
        contact_identifier_strategy="user",
    )
    by_conversation = MicrosoftTeamsProvider.activity_to_channel_message(
        _message_activity(),
        app_id="bot-1",
        contact_identifier_strategy="conversation",
    )

    assert by_user.contact_identifier == "user-1"
    assert by_conversation.contact_identifier == "conv-1"


@pytest.mark.asyncio
async def test_provider_remembers_conversation_reference_for_session() -> None:
    provider = MicrosoftTeamsProvider(
        MicrosoftTeamsConfig(app_id="bot-1", access_token="token")
    )
    message = provider.parse_channel_message(_message_activity())

    session = await provider.get_session(message.contact_identifier)

    assert session is not None
    assert session.session_id == "microsoft_teams:bot-1:conv-1:user-1"
    assert session.context_data["teams_reference"]["conversation"]["id"] == "conv-1"
    assert session.context_data["teams_reference"]["user"]["id"] == "user-1"

    await provider.shutdown()
