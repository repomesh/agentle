from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, Mapping
from datetime import datetime
from typing import TYPE_CHECKING, Any, Sequence, cast, override
from uuid import uuid4

from agentle.generations.models.generation.choice import Choice
from agentle.generations.models.generation.generation import Generation
from agentle.generations.models.generation.generation_config import GenerationConfig
from agentle.generations.models.generation.generation_config_dict import (
    GenerationConfigDict,
)
from agentle.generations.models.generation.usage import Usage
from agentle.generations.models.message_parts.text import TextPart
from agentle.generations.models.message_parts.tool_execution_suggestion import (
    ToolExecutionSuggestion,
)
from agentle.generations.models.messages.assistant_message import AssistantMessage
from agentle.generations.models.messages.developer_message import DeveloperMessage
from agentle.generations.models.messages.generated_assistant_message import (
    GeneratedAssistantMessage,
)
from agentle.generations.models.messages.message import Message
from agentle.generations.models.messages.user_message import UserMessage
from agentle.generations.providers.base.generation_provider import GenerationProvider
from agentle.generations.providers.ollama.adapters.chat_response_to_generation_adapter import (
    ChatResponseToGenerationAdapter,
)
from agentle.generations.providers.ollama.adapters.message_to_ollama_message_adapter import (
    MessageToOllamaMessageAdapter,
)
from agentle.generations.providers.ollama.adapters.tool_to_ollama_tool_adapter import (
    ToolToOllamaToolAdapter,
)
from agentle.generations.providers.types.model_kind import ModelKind
from agentle.generations.tools.tool import Tool
from agentle.generations.tracing import observe

if TYPE_CHECKING:
    from ollama._types import Options

    from agentle.generations.tracing.otel_client import OtelClient

type WithoutStructuredOutput = None


class OllamaGenerationProvider(GenerationProvider):
    def __init__(
        self,
        *,
        otel_clients: Sequence[OtelClient] | OtelClient | None = None,
        provider_id: str | None = None,
        options: Mapping[str, Any] | Options | None = None,
        think: bool | None = None,
        host: str | None = None,
    ) -> None:
        from ollama._client import AsyncClient

        super().__init__(otel_clients=otel_clients, provider_id=provider_id)
        self._client = AsyncClient(host=host)
        self.options = options
        self.think = think

    @property
    @override
    def default_model(self) -> str:
        return "gemma3n:e4b"

    @property
    @override
    def organization(self) -> str:
        return "Ollama"

    @override
    async def stream_async[T = WithoutStructuredOutput](
        self,
        *,
        model: str | ModelKind | None = None,
        messages: Sequence[Message],
        response_schema: type[T] | None = None,
        generation_config: GenerationConfig | GenerationConfigDict | None = None,
        tools: Sequence[Tool] | None = None,
    ) -> AsyncGenerator[Generation[T], None]:
        from pydantic import BaseModel

        from agentle.utils.make_fields_optional import make_fields_optional
        from agentle.utils.parse_streaming_json import parse_streaming_json

        tool_adapter = ToolToOllamaToolAdapter()

        bm = cast(type[BaseModel], response_schema) if response_schema else None
        optional_response_schema = make_fields_optional(bm) if bm else None

        _generation_config = self._normalize_generation_config(generation_config)

        _model = self._resolve_model(model)
        message_adapter = MessageToOllamaMessageAdapter()
        _messages = [message_adapter.adapt(m) for m in messages]

        _tools = [tool_adapter.adapt(tool) for tool in tools] if tools else None

        generation_id = uuid4()
        created = datetime.now()
        accumulated_content = ""
        accumulated_thinking = ""
        accumulated_tool_calls: list[ToolExecutionSuggestion] = []
        seen_tool_calls: set[str] = set()
        usage = Usage.zero()

        def parse_tool_arguments(arguments: object) -> Mapping[str, object]:
            if isinstance(arguments, Mapping):
                return cast(Mapping[str, object], arguments)

            if isinstance(arguments, str):
                try:
                    parsed_arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    return {}

                if isinstance(parsed_arguments, Mapping):
                    return cast(Mapping[str, object], parsed_arguments)

            return {}

        def tool_call_signature(tool_name: str, args: Mapping[str, object]) -> str:
            try:
                serialized_args = json.dumps(args, sort_keys=True, default=str)
            except TypeError:
                serialized_args = str(args)

            return f"{tool_name}:{serialized_args}"

        try:
            async with asyncio.timeout(_generation_config.timeout_in_seconds):
                response_stream = await self._client.chat(
                    model=_model,
                    messages=_messages,
                    tools=_tools,
                    stream=True,
                    format=bm.model_json_schema() if bm else None,
                    options=self.options,
                    think=self.think,
                )

                async for chunk in response_stream:
                    message = chunk.message

                    text_content = message.content
                    if text_content:
                        accumulated_content += text_content

                    thinking = getattr(message, "thinking", None)
                    if thinking:
                        accumulated_thinking += thinking

                    tool_calls = message.tool_calls
                    if tool_calls:
                        for tool_call in tool_calls:
                            function = tool_call.function
                            args = parse_tool_arguments(function.arguments)
                            signature = tool_call_signature(function.name, args)

                            if signature in seen_tool_calls:
                                continue

                            seen_tool_calls.add(signature)
                            accumulated_tool_calls.append(
                                ToolExecutionSuggestion(
                                    id=str(uuid4()),
                                    tool_name=function.name,
                                    args=args,
                                )
                            )

                    prompt_eval_count = getattr(chunk, "prompt_eval_count", None)
                    eval_count = getattr(chunk, "eval_count", None)
                    if prompt_eval_count is not None or eval_count is not None:
                        usage = Usage(
                            prompt_tokens=prompt_eval_count or 0,
                            completion_tokens=eval_count or 0,
                        )

                    parts: list[TextPart | ToolExecutionSuggestion] = []
                    if accumulated_content:
                        parts.append(TextPart(text=accumulated_content))

                    parts.extend(accumulated_tool_calls)

                    parsed = (
                        parse_streaming_json(
                            accumulated_content,
                            model=optional_response_schema,
                        )
                        if optional_response_schema
                        else None
                    )

                    generation = Generation[Any](
                        id=generation_id,
                        object="chat.generation",
                        created=created,
                        choices=[
                            Choice[Any](
                                index=0,
                                message=GeneratedAssistantMessage[Any](
                                    parts=parts,
                                    parsed=cast(T, parsed),
                                    reasoning=accumulated_thinking or None,
                                ),
                            )
                        ],
                        model=_model,
                        usage=usage,
                    )

                    yield cast(Generation[T], generation)
        except asyncio.TimeoutError as e:
            e.add_note(
                f"Content generation timed out after {_generation_config.timeout_in_seconds}s"
            )
            raise

    @observe
    @override
    async def generate_async[T](
        self,
        *,
        model: str | ModelKind | None = None,
        messages: Sequence[AssistantMessage | DeveloperMessage | UserMessage],
        response_schema: type[T] | None = None,
        generation_config: GenerationConfig | GenerationConfigDict | None = None,
        tools: Sequence[Tool[Any]] | None = None,
        fallback_models: Sequence[str] | None = None,
    ) -> Generation[T]:
        """Note: Ollama does not support fallback models. Parameter ignored."""
        from pydantic import BaseModel

        tool_adapter = ToolToOllamaToolAdapter()

        bm = cast(BaseModel, response_schema) if response_schema else None  # type: ignore

        _generation_config = self._normalize_generation_config(generation_config)

        _model = self._resolve_model(model)
        message_adapter = MessageToOllamaMessageAdapter()
        _messages = [message_adapter.adapt(m) for m in messages]

        _tools = [tool_adapter.adapt(tool) for tool in tools] if tools else None

        try:
            async with asyncio.timeout(_generation_config.timeout_in_seconds):
                response = await self._client.chat(
                    model=_model,
                    messages=_messages,
                    tools=_tools,
                    format=bm.model_json_schema() if bm else None,
                    options=self.options,
                    think=self.think,
                )
        except asyncio.TimeoutError as e:
            e.add_note(
                f"Content generation timed out after {_generation_config.timeout_in_seconds}s"
            )
            raise

        return ChatResponseToGenerationAdapter(
            model=_model, response_schema=response_schema
        ).adapt(response)  # type: ignore

    @override
    async def price_per_million_tokens_input(
        self, model: str, estimate_tokens: int | None = None
    ) -> float:
        return 0.0

    @override
    async def price_per_million_tokens_output(
        self, model: str, estimate_tokens: int | None = None
    ) -> float:
        return 0.0

    @override
    def map_model_kind_to_provider_model(
        self,
        model_kind: ModelKind,
    ) -> str:
        """
        Maps abstract ModelKind categories to specific Ollama model names.

        This mapping is based on the latest available Ollama models as of July 2025,
        focusing on the most capable and well-supported models in each category.
        """
        mapping: Mapping[ModelKind, str] = {
            # Nano: Smallest, fastest, most cost-effective models
            "category_nano": "llama3.2:1b",
            "category_nano_experimental": "smollm2:135m",
            # Mini: Small but capable models
            "category_mini": "llama3.2:3b",
            "category_mini_experimental": "phi4:mini",
            # Standard: Mid-range, balanced performance models
            "category_standard": "llama3.1:8b",
            "category_standard_experimental": "qwen2.5:7b",
            # Pro: High performance models
            "category_pro": "llama3.1:70b",
            "category_pro_experimental": "qwen2.5:14b",
            # Flagship: Best available models from provider
            "category_flagship": "llama3.3:70b",
            "category_flagship_experimental": "qwen3:235b",
            # Reasoning: Specialized for complex reasoning
            "category_reasoning": "deepseek-r1:32b",
            "category_reasoning_experimental": "qwq:32b",
            # Vision: Multimodal capabilities for image/video processing
            "category_vision": "llama3.2-vision:11b",
            "category_vision_experimental": "qwen2-vl:7b",
            # Coding: Specialized for programming tasks
            "category_coding": "codellama:13b",
            "category_coding_experimental": "qwen2.5-coder:7b",
            # Instruct: Fine-tuned for instruction following
            "category_instruct": "dolphin-llama3:8b",
            "category_instruct_experimental": "openhermes:7b",
        }

        return mapping.get(model_kind, "llama3.1:8b")  # Default fallback
