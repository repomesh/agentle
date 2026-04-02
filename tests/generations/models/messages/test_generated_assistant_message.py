from agentle.generations.models.message_parts.text import TextPart
from agentle.generations.models.messages.generated_assistant_message import (
    GeneratedAssistantMessage,
)


def test_to_assistant_message_preserves_reasoning_fields():
    reasoning_details = [
        {
            "type": "reasoning.text",
            "text": "Need the weather tool first.",
            "format": "anthropic-claude-v1",
            "id": "reasoning-1",
            "index": 0,
        }
    ]
    message = GeneratedAssistantMessage(
        parts=[TextPart(text="Answer incoming")],
        parsed=None,
        reasoning="Need the weather tool first.",
        reasoning_details=reasoning_details,
    )

    assistant_message = message.to_assistant_message()

    assert assistant_message.reasoning == "Need the weather tool first."
    assert assistant_message.reasoning_details == reasoning_details
    assert assistant_message.parts[0].text == "Answer incoming"
