from agentle.generations.providers.openrouter._adapters.openrouter_message_to_generated_assistant_message_adapter import (
    OpenRouterMessageToGeneratedAssistantMessageAdapter,
)


def test_openrouter_message_adapter_preserves_reasoning_and_reasoning_details():
    adapter = OpenRouterMessageToGeneratedAssistantMessageAdapter()
    reasoning_details = [
        {
            "type": "reasoning.summary",
            "summary": "Need to call weather tool before answering.",
            "format": "anthropic-claude-v1",
            "id": "reasoning-summary-1",
            "index": 0,
        }
    ]

    message = adapter.adapt(
        {
            "role": "assistant",
            "content": "I will check the weather.",
            "reasoning": "Need to call weather tool before answering.",
            "reasoning_details": reasoning_details,
        }
    )

    assert message.text == "I will check the weather."
    assert message.reasoning == "Need to call weather tool before answering."
    assert message.reasoning_details == reasoning_details
