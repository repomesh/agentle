from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from agentle.generations.models.generation.generation_config import GenerationConfig
from agentle.generations.models.message_parts.text import TextPart
from agentle.generations.models.messages.user_message import UserMessage
from agentle.generations.providers.base.generation_provider import GenerationProvider
from agentle.generations.providers.ollama import ollama_generation_provider as provider_module
from agentle.generations.providers.ollama.ollama_generation_provider import (
    OllamaGenerationProvider,
)


class FakeClient:
    def __init__(self, chunks: Sequence[object]) -> None:
        self.chunks = chunks
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> AsyncIterator[object]:
        self.calls.append(kwargs)
        return self._stream()

    async def _stream(self) -> AsyncIterator[object]:
        for chunk in self.chunks:
            yield chunk


class FakeMessageToOllamaMessageAdapter:
    def adapt(self, message: object) -> dict[str, object]:
        return {"role": getattr(message, "role"), "content": getattr(message, "text")}


class FakeToolToOllamaToolAdapter:
    def adapt(self, tool: object) -> dict[str, object]:
        return {"type": "function", "function": {"name": str(tool)}}


def build_provider(
    monkeypatch: pytest.MonkeyPatch,
    chunks: Sequence[object],
    *,
    options: dict[str, object] | None = None,
    think: bool | None = None,
) -> tuple[OllamaGenerationProvider, FakeClient]:
    monkeypatch.setattr(
        provider_module,
        "MessageToOllamaMessageAdapter",
        FakeMessageToOllamaMessageAdapter,
    )
    monkeypatch.setattr(
        provider_module,
        "ToolToOllamaToolAdapter",
        FakeToolToOllamaToolAdapter,
    )

    fake_client = FakeClient(chunks)
    provider = OllamaGenerationProvider.__new__(OllamaGenerationProvider)
    GenerationProvider.__init__(provider)
    provider._client = fake_client
    provider.options = options
    provider.think = think

    return provider, fake_client


def make_chat_chunk(
    content: str = "",
    *,
    thinking: str | None = None,
    tool_calls: Sequence[object] | None = None,
    prompt_eval_count: int | None = None,
    eval_count: int | None = None,
) -> object:
    return SimpleNamespace(
        message=SimpleNamespace(
            content=content,
            thinking=thinking,
            tool_calls=tool_calls,
        ),
        prompt_eval_count=prompt_eval_count,
        eval_count=eval_count,
    )


def make_tool_call(name: str, arguments: object) -> object:
    return SimpleNamespace(
        function=SimpleNamespace(
            name=name,
            arguments=arguments,
        )
    )


@pytest.mark.asyncio
async def test_stream_async_calls_ollama_with_stream_and_accumulates_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool_call = make_tool_call("lookup", {"query": "agentle"})
    provider, fake_client = build_provider(
        monkeypatch,
        [
            make_chat_chunk("Hel", thinking="think "),
            make_chat_chunk(
                "lo",
                thinking="again",
                tool_calls=[tool_call],
                prompt_eval_count=3,
                eval_count=2,
            ),
        ],
        options={"temperature": 0.2},
        think=True,
    )

    tool = "search_tool"

    chunks = [
        chunk
        async for chunk in provider.stream_async(
            model="gemma3",
            messages=[UserMessage(parts=[TextPart(text="hi")])],
            generation_config=GenerationConfig(timeout_s=1),
            tools=[tool],
        )
    ]

    assert fake_client.calls == [
        {
            "model": "gemma3",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": str(tool)}}],
            "stream": True,
            "format": None,
            "options": {"temperature": 0.2},
            "think": True,
        }
    ]
    assert fake_client.calls[0]["stream"] is True
    assert fake_client.calls[0]["format"] is None
    assert fake_client.calls[0]["options"] == {"temperature": 0.2}
    assert fake_client.calls[0]["think"] is True

    assert len(chunks) == 2
    assert chunks[0].id == chunks[1].id
    assert chunks[0].created == chunks[1].created
    assert chunks[0].message.text == "Hel"
    assert chunks[0].usage.total_tokens == 0
    assert chunks[1].message.text == "Hello"
    assert chunks[1].message.reasoning == "think again"
    assert chunks[1].usage.prompt_tokens == 3
    assert chunks[1].usage.completion_tokens == 2
    assert len(chunks[1].message.tool_calls) == 1
    assert chunks[1].message.tool_calls[0].tool_name == "lookup"
    assert chunks[1].message.tool_calls[0].args == {"query": "agentle"}


@pytest.mark.asyncio
async def test_stream_async_parses_structured_output_progressively(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StructuredAnswer(BaseModel):
        answer: str

    provider, fake_client = build_provider(
        monkeypatch,
        [
            make_chat_chunk('{"answer":"Hel'),
            make_chat_chunk('lo"}', prompt_eval_count=4, eval_count=2),
        ],
    )

    chunks = [
        chunk
        async for chunk in provider.stream_async(
            model="gemma3",
            messages=[UserMessage(parts=[TextPart(text="hi")])],
            response_schema=StructuredAnswer,
            generation_config=GenerationConfig(timeout_s=1),
        )
    ]

    assert fake_client.calls[0]["stream"] is True
    assert fake_client.calls[0]["format"]["properties"]["answer"]["type"] == "string"
    assert chunks[0].message.parsed.answer == "Hel"
    assert chunks[1].message.parsed.answer == "Hello"
    assert chunks[1].message.text == '{"answer":"Hello"}'
    assert chunks[1].usage.prompt_tokens == 4
    assert chunks[1].usage.completion_tokens == 2
