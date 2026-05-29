# Adapter for OpenRouter streaming responses to generation
"""
Adapter for converting OpenRouter streaming responses to Agentle Generation objects.

This module handles the transformation of OpenRouter's SSE streaming format into
Agentle's standardized Generation format, processing delta chunks into complete
messages.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

from agentle.generations.models.generation.generation import Generation
from agentle.generations.models.generation.usage import Usage
from agentle.generations.models.message_parts.text import TextPart
from agentle.generations.models.message_parts.tool_execution_suggestion import (
    ToolExecutionSuggestion,
)
from agentle.generations.models.messages.generated_assistant_message import (
    GeneratedAssistantMessage,
)
from agentle.generations.models.generation.choice import Choice
from agentle.generations.providers.openrouter._types import (
    OpenRouterStreamResponse,
    OpenRouterStreamDelta,
)
from agentle.generations.providers.openrouter._adapters.openrouter_response_to_generation_adapter import (
    _build_openrouter_usage,
)
from agentle.utils.parse_streaming_json import parse_streaming_json
from agentle.utils.make_fields_optional import make_fields_optional

logger = logging.getLogger(__name__)


class OpenRouterStreamToGenerationAdapter[T]:
    """
    Adapter for converting OpenRouter streaming responses to Generation objects.

    Processes SSE streaming chunks and accumulates them into Generation objects
    that can be yielded as the stream progresses.

    Attributes:
        response_schema: Optional Pydantic model class for parsing structured data.
        model: The model identifier being used.
    """

    response_schema: type[T] | None
    model: str

    def __init__(
        self,
        *,
        response_schema: type[T] | None = None,
        model: str,
    ):
        """
        Initialize the streaming adapter.

        Args:
            response_schema: Optional Pydantic model class for structured output.
            model: The model identifier being used.
        """
        self.response_schema = response_schema
        self.model = model

    async def adapt(
        self,
        response_stream: AsyncGenerator[bytes, None],
    ) -> AsyncGenerator[Generation[None], None]:
        """
        Convert an OpenRouter SSE stream to an async generator of Generations.

        Args:
            response_stream: The async generator of raw response bytes.

        Yields:
            Generation objects as chunks arrive.
        """
        # Accumulate content across chunks
        accumulated_content = ""
        accumulated_reasoning = ""
        accumulated_reasoning_details: list[dict[str, Any]] = []
        accumulated_tool_calls: dict[str, dict[str, Any]] = {}
        accumulated_usage: dict[str, Any] | None = None
        accumulated_model = self.model
        buffer = ""

        async for chunk_bytes in response_stream:
            chunk_str = chunk_bytes.decode("utf-8")
            buffer += chunk_str

            # Process complete SSE lines
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()

                # Skip empty lines and comments
                if not line or line.startswith(":"):
                    continue

                # Parse SSE data
                if line.startswith("data: "):
                    data = line[6:]

                    # Check for stream end
                    if data == "[DONE]":
                        # Yield final generation if we have accumulated content
                        if (
                            accumulated_content
                            or accumulated_reasoning
                            or accumulated_reasoning_details
                            or accumulated_tool_calls
                            or accumulated_usage
                        ):
                            yield self._build_generation(
                                accumulated_content,
                                accumulated_reasoning,
                                accumulated_reasoning_details,
                                accumulated_tool_calls,
                                accumulated_usage,
                                accumulated_model,
                            )
                        return

                    try:
                        chunk_data: OpenRouterStreamResponse = json.loads(data)

                        # Check for errors in the chunk
                        if "error" in chunk_data:
                            error_msg = chunk_data.get("error", {}).get(
                                "message", "Unknown error"
                            )
                            logger.error(f"Stream error: {error_msg}")
                            raise RuntimeError(f"OpenRouter stream error: {error_msg}")

                        chunk_model = str(chunk_data.get("model") or "").strip()
                        if chunk_model:
                            accumulated_model = chunk_model

                        chunk_usage = chunk_data.get("usage")
                        if isinstance(chunk_usage, dict):
                            accumulated_usage = dict(chunk_usage)

                        # Extract delta from first choice
                        if chunk_data.get("choices"):
                            choice = chunk_data["choices"][0]
                            delta: OpenRouterStreamDelta = choice.get("delta", {})

                            # Accumulate content
                            if "content" in delta and delta["content"]:
                                accumulated_content += delta["content"]

                            # Accumulate reasoning
                            if "reasoning" in delta and delta["reasoning"]:
                                accumulated_reasoning += delta["reasoning"]

                            if (
                                "reasoning_details" in delta
                                and delta["reasoning_details"]
                            ):
                                accumulated_reasoning_details.extend(
                                    delta["reasoning_details"]
                                )

                            # Accumulate tool calls
                            if "tool_calls" in delta:
                                for tool_call in delta["tool_calls"]:
                                    tool_id = tool_call.get("id", "")
                                    if tool_id not in accumulated_tool_calls:
                                        accumulated_tool_calls[tool_id] = {
                                            "id": tool_id,
                                            "type": "function",
                                            "function": {
                                                "name": "",
                                                "arguments": "",
                                            },
                                        }

                                    function = tool_call.get("function", {})
                                    if "name" in function:
                                        accumulated_tool_calls[tool_id]["function"][
                                            "name"
                                        ] = function["name"]
                                    if "arguments" in function:
                                        accumulated_tool_calls[tool_id]["function"][
                                            "arguments"
                                        ] += function["arguments"]

                            # Check for finish_reason
                            finish_reason = choice.get("finish_reason")
                            if finish_reason:
                                # Stream is complete for this choice
                                yield self._build_generation(
                                    accumulated_content,
                                    accumulated_reasoning,
                                    accumulated_reasoning_details,
                                    accumulated_tool_calls,
                                    accumulated_usage,
                                    accumulated_model,
                                )
                                continue

                            # Yield intermediate generation
                            yield self._build_generation(
                                accumulated_content,
                                accumulated_reasoning,
                                accumulated_reasoning_details,
                                accumulated_tool_calls,
                                accumulated_usage,
                                accumulated_model,
                            )
                        elif accumulated_usage:
                            yield self._build_generation(
                                accumulated_content,
                                accumulated_reasoning,
                                accumulated_reasoning_details,
                                accumulated_tool_calls,
                                accumulated_usage,
                                accumulated_model,
                            )

                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse SSE chunk: {data}")
                        continue

    def _build_generation(
        self,
        content: str,
        reasoning: str,
        reasoning_details: list[dict[str, Any]],
        tool_calls: dict[str, dict[str, Any]],
        usage_data: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> Generation[None]:
        """
        Build a Generation object from accumulated chunks.

        Args:
            content: Accumulated text content.
            reasoning: Accumulated reasoning.
            reasoning_details: Accumulated structured reasoning details.
            tool_calls: Accumulated tool calls.

        Returns:
            A Generation object with the current state.
        """
        import datetime
        import uuid

        # Build message parts
        parts: list[TextPart | ToolExecutionSuggestion] = []

        if content:
            parts.append(TextPart(text=content))

        # Add tool execution suggestions
        for tool_call_data in tool_calls.values():
            function_data = tool_call_data.get("function", {})
            try:
                args_str = str(function_data.get("arguments", "{}"))
                args = json.loads(args_str)
            except json.JSONDecodeError:
                args = {}

            parts.append(
                ToolExecutionSuggestion(
                    id=str(tool_call_data.get("id", "")),
                    tool_name=str(function_data.get("name", "")),
                    args=args,
                )
            )

        # Parse accumulated content if response_schema is provided
        parsed_data: Any = None
        if self.response_schema is not None:
            try:
                # Try to parse if it looks like a Pydantic/BaseModel class
                if hasattr(self.response_schema, "model_fields"):
                    # Make fields optional for streaming partial results
                    optional_model = make_fields_optional(self.response_schema)  # type: ignore
                    parsed_data = parse_streaming_json(content, model=optional_model)
            except Exception as e:
                logger.warning(f"Failed to parse streaming JSON: {e}")

        # Create GeneratedAssistantMessage
        message = GeneratedAssistantMessage[Any](
            parts=parts,
            parsed=parsed_data,
            reasoning=reasoning if reasoning else None,
            reasoning_details=reasoning_details if reasoning_details else None,
        )

        # Create Choice with correct type
        choice = Choice[Any](
            index=0,
            message=message,
        )

        usage = (
            _build_openrouter_usage(usage_data)
            if usage_data
            else Usage(prompt_tokens=0, completion_tokens=0)
        )

        return Generation[Any](
            id=uuid.uuid4(),
            choices=[choice],
            object="chat.generation",
            created=datetime.datetime.now(),
            model=model or self.model,
            usage=usage,
        )
