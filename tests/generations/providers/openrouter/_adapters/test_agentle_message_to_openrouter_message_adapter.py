from agentle.generations.models.message_parts.text import CacheControl, TextPart
from agentle.generations.models.messages.developer_message import DeveloperMessage
from agentle.generations.providers.openrouter._adapters.agentle_message_to_openrouter_message_adapter import (
    AgentleMessageToOpenRouterMessageAdapter,
)


def test_developer_message_without_cache_control_stays_flat_string():
    adapter = AgentleMessageToOpenRouterMessageAdapter()

    result = adapter.adapt(DeveloperMessage(parts=[TextPart(text="persona estavel")]))

    assert result["role"] == "system"
    # Backward-compatible: plain string content when no caching is requested.
    assert result["content"] == "persona estavel"


def test_developer_message_with_cache_control_emits_array_with_marker():
    adapter = AgentleMessageToOpenRouterMessageAdapter()

    result = adapter.adapt(
        DeveloperMessage(
            parts=[
                TextPart(
                    text="persona estavel",
                    cache_control=CacheControl(type="ephemeral"),
                )
            ]
        )
    )

    assert result["role"] == "system"
    # The cache_control marker must survive (a flat string would drop it).
    assert result["content"] == [
        {
            "type": "text",
            "text": "persona estavel",
            "cache_control": {"type": "ephemeral"},
        }
    ]
