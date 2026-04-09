from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, cast

import pytest

from agentle.agents.agent import Agent
from agentle.agents.conversations.conversation_store import ConversationStore
from agentle.generations.models.generation.choice import Choice
from agentle.generations.models.generation.generation import Generation
from agentle.generations.models.generation.usage import Usage
from agentle.generations.models.message_parts.text import TextPart
from agentle.generations.models.messages.assistant_message import AssistantMessage
from agentle.generations.models.messages.developer_message import DeveloperMessage
from agentle.generations.models.messages.generated_assistant_message import (
    GeneratedAssistantMessage,
)
from agentle.generations.models.messages.user_message import UserMessage
from agentle.generations.providers.base.generation_provider import GenerationProvider
from agentle.generations.providers.types.model_kind import ModelKind


class RecordingConversationStore(ConversationStore):
    def __init__(self, *, fail_on_add: int | None = None) -> None:
        super().__init__()
        self.fail_on_add = fail_on_add
        self.add_attempts = 0
        self.messages: dict[
            str, list[DeveloperMessage | UserMessage | AssistantMessage]
        ] = {}

    async def get_conversation_history_async(
        self, chat_id: str
    ) -> list[DeveloperMessage | UserMessage | AssistantMessage]:
        return list(self.messages.get(chat_id, []))

    async def add_message_async[T = Any](
        self,
        chat_id: str,
        message: DeveloperMessage
        | UserMessage
        | AssistantMessage
        | GeneratedAssistantMessage[T],
    ) -> None:
        self.add_attempts += 1
        if self.fail_on_add is not None and self.add_attempts == self.fail_on_add:
            raise RuntimeError("conversation store write failed")

        stored_message = (
            message.to_assistant_message()
            if isinstance(message, GeneratedAssistantMessage)
            else message
        )
        self.messages.setdefault(chat_id, []).append(stored_message)

    async def clear_conversation_async(self, chat_id: str) -> None:
        self.messages.pop(chat_id, None)


class StubGenerationProvider(GenerationProvider):
    def __init__(
        self,
        *,
        generation: Generation[Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        super().__init__()
        self._generation = generation
        self._error = error
        self.calls = 0

    @property
    def default_model(self) -> str:
        return "test-model"

    @property
    def organization(self) -> str:
        return "tests"

    async def generate_async[T = None](
        self,
        *,
        model: str | ModelKind | None = None,
        messages,
        response_schema: type[T] | None = None,
        generation_config=None,
        tools=None,
        fallback_models=None,
    ) -> Generation[T]:
        self.calls += 1
        if self._error is not None:
            raise self._error
        assert self._generation is not None
        return cast(Generation[T], self._generation)

    async def price_per_million_tokens_input(
        self, model: str, estimate_tokens: int | None = None
    ) -> float:
        return 0.0

    async def price_per_million_tokens_output(
        self, model: str, estimate_tokens: int | None = None
    ) -> float:
        return 0.0

    def map_model_kind_to_provider_model(self, model_kind: ModelKind) -> str:
        return str(model_kind)


def make_generation(text: str) -> Generation[Any]:
    return Generation(
        id=uuid.uuid4(),
        object="chat.generation",
        created=datetime.now(),
        model="test-model",
        choices=[
            Choice(
                index=0,
                message=GeneratedAssistantMessage(
                    parts=[TextPart(text=text)],
                    parsed=None,
                ),
            )
        ],
        usage=Usage(prompt_tokens=3, completion_tokens=5),
    )


@pytest.mark.asyncio
async def test_run_persists_user_message_before_generation_failure() -> None:
    store = RecordingConversationStore()
    provider = StubGenerationProvider(error=RuntimeError("generation failed"))
    agent = Agent(
        generation_provider=provider,
        model="test-model",
        instructions="Be helpful.",
        conversation_store=store,
    )

    with pytest.raises(RuntimeError, match="generation failed"):
        await agent.run_async("Hello from the user", chat_id="chat-1")

    stored_messages = store.messages["chat-1"]
    assert len(stored_messages) == 1
    assert isinstance(stored_messages[0], UserMessage)
    assert stored_messages[0].text == "Hello from the user"


@pytest.mark.asyncio
async def test_run_persists_user_and_assistant_once_on_success() -> None:
    store = RecordingConversationStore()
    provider = StubGenerationProvider(generation=make_generation("Assistant reply"))
    agent = Agent(
        generation_provider=provider,
        model="test-model",
        instructions="Be helpful.",
        conversation_store=store,
    )

    result = await agent.run_async("Need an answer", chat_id="chat-1")

    stored_messages = store.messages["chat-1"]
    assert len(stored_messages) == 2
    assert isinstance(stored_messages[0], UserMessage)
    assert stored_messages[0].text == "Need an answer"
    assert isinstance(stored_messages[1], AssistantMessage)
    assert stored_messages[1].parts[0].text == "Assistant reply"
    assert result.text == "Assistant reply"


@pytest.mark.asyncio
async def test_run_aborts_when_user_message_persistence_fails() -> None:
    store = RecordingConversationStore(fail_on_add=1)
    provider = StubGenerationProvider(generation=make_generation("Should not run"))
    agent = Agent(
        generation_provider=provider,
        model="test-model",
        instructions="Be helpful.",
        conversation_store=store,
    )

    with pytest.raises(RuntimeError, match="conversation store write failed"):
        await agent.run_async("Persist me first", chat_id="chat-1")

    assert provider.calls == 0
    assert store.messages == {}
