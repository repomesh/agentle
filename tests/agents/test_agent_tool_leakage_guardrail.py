import uuid
from datetime import datetime
from typing import Any

import pytest

from agentle.agents.agent import Agent
from agentle.generations.models.generation.choice import Choice
from agentle.generations.models.generation.generation import Generation
from agentle.generations.models.generation.usage import Usage
from agentle.generations.models.message_parts.text import TextPart
from agentle.generations.models.message_parts.tool_execution_suggestion import (
    ToolExecutionSuggestion,
)
from agentle.generations.models.messages.generated_assistant_message import (
    GeneratedAssistantMessage,
)
from agentle.generations.providers.base.generation_provider import GenerationProvider
from agentle.generations.tools.terminal import terminal
from agentle.guardrails.validators.tool_leakage_validator import ToolLeakageValidator


def registrar_agendamento(data_hora_inicio: str) -> str:
    return f"registrado: {data_hora_inicio}"


@terminal()
def stop_without_public_message(reason: str) -> str:
    return f"stopped: {reason}"


def make_generation(parts: list[Any]) -> Generation[None]:
    return Generation(
        id=uuid.uuid4(),
        object="chat.generation",
        created=datetime.now(),
        model="mock-model",
        choices=[
            Choice(
                index=0,
                message=GeneratedAssistantMessage(parts=parts, parsed=None),
            )
        ],
        usage=Usage(prompt_tokens=10, completion_tokens=10),
    )


class LeakyTextProvider(GenerationProvider):
    async def generate_async(self, messages, tools=None, **kwargs):
        return make_generation(
            [
                TextPart(
                    text="Tool: registrar_agendamento\n"
                    "Args: {'data_hora_inicio': '2026-05-19 11:15'}"
                )
            ]
        )

    @property
    def default_model(self) -> str:
        return "mock-model"

    @property
    def map_model_kind_to_provider_model(self) -> dict[str, str]:
        return {}

    @property
    def organization(self) -> str:
        return "mock-org"

    @property
    def price_per_million_tokens_input(self) -> float:
        return 0.0

    @property
    def price_per_million_tokens_output(self) -> float:
        return 0.0


class MixedPartsProvider(LeakyTextProvider):
    async def generate_async(self, messages, tools=None, **kwargs):
        return make_generation(
            [
                TextPart(text="Confirmado."),
                ToolExecutionSuggestion(
                    tool_name="registrar_agendamento",
                    args={"data_hora_inicio": "2026-05-19 11:15"},
                ),
            ]
        )


class TerminalWithoutMessageProvider(LeakyTextProvider):
    async def generate_async(self, messages, tools=None, **kwargs):
        return make_generation(
            [
                ToolExecutionSuggestion(
                    tool_name="stop_without_public_message",
                    args={"reason": "done"},
                )
            ]
        )


class DirectStreamingLeakyProvider(LeakyTextProvider):
    async def stream_async(self, messages, tools=None, **kwargs):
        yield make_generation(
            [
                TextPart(
                    text="Tool: registrar_agendamento\n"
                    "Args: {'data_hora_inicio': '2026-05-19 11:15'}"
                )
            ]
        )


@pytest.mark.asyncio
async def test_explicit_tool_leakage_guardrail_sanitizes_textual_tool_call() -> None:
    agent = Agent(
        generation_provider=LeakyTextProvider(),
        tools=[registrar_agendamento],
        output_guardrails=[
            ToolLeakageValidator(
                tools=[registrar_agendamento],
                block_on_detection=False,
                redact_leakage=True,
            ),
        ],
        instructions="Test agent",
    )

    output = await agent.run_async("Agende")

    assert output.text == ""
    assert "Tool:" not in output.generation_text
    assert "Args:" not in output.generation_text


@pytest.mark.asyncio
async def test_tool_leakage_guardrail_is_not_installed_by_default() -> None:
    agent = Agent(
        generation_provider=LeakyTextProvider(),
        tools=[registrar_agendamento],
        instructions="Test agent",
    )

    output = await agent.run_async("Agende")

    assert output.text.startswith("Tool: registrar_agendamento")


@pytest.mark.asyncio
async def test_public_text_ignores_tool_call_parts_in_final_generation() -> None:
    agent = Agent(
        generation_provider=MixedPartsProvider(),
        instructions="Test agent",
    )

    output = await agent.run_async("Agende")

    assert output.text == "Confirmado."
    assert output.generation is not None
    assert len(output.generation.tool_calls) == 1


@pytest.mark.asyncio
async def test_terminal_tool_without_message_param_does_not_leak_tool_call() -> None:
    agent = Agent(
        generation_provider=TerminalWithoutMessageProvider(),
        tools=[stop_without_public_message],
        instructions="Test agent",
    )

    output = await agent.run_async("Stop")

    assert output.text == ""
    assert "Tool:" not in output.generation_text
    assert "Args:" not in output.generation_text


@pytest.mark.asyncio
async def test_streaming_chunks_are_sanitized_before_yield() -> None:
    agent = Agent(
        generation_provider=DirectStreamingLeakyProvider(),
        output_guardrails=[
            ToolLeakageValidator(tool_names=["registrar_agendamento"]),
        ],
        instructions="Test agent",
    )

    stream = await agent.run_async("Agende", stream=True)
    chunks = [chunk async for chunk in stream]

    assert chunks
    assert all("Tool:" not in chunk.text for chunk in chunks)
    assert all("Args:" not in chunk.text for chunk in chunks)
