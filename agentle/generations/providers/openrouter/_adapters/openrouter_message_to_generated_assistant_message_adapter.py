# Adapter for OpenRouter message to generated assistant message
"""
Adapter for converting OpenRouter response messages to Agentle's GeneratedAssistantMessage.

This module handles conversion of OpenRouter's response format into Agentle's
internal GeneratedAssistantMessage format, including structured output parsing.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import cast

from rsb.adapters.adapter import Adapter
from rsb.models.base_model import BaseModel

from agentle.generations.models.message_parts.text import TextPart
from agentle.generations.models.message_parts.tool_execution_suggestion import (
    ToolExecutionSuggestion,
)
from agentle.generations.models.messages.generated_assistant_message import (
    GeneratedAssistantMessage,
)
from agentle.generations.providers.openrouter._types import (
    OpenRouterResponseMessage,
    OpenRouterToolCall,
)


class OpenRouterMessageToGeneratedAssistantMessageAdapter[T](
    Adapter[
        OpenRouterResponseMessage,
        GeneratedAssistantMessage[T],
    ]
):
    """
    Adapter for converting OpenRouter response messages to GeneratedAssistantMessage.

    Handles conversion of text content, tool calls, and structured output parsing.

    Attributes:
        response_schema: Optional Pydantic model class for parsing structured data.
    """

    response_schema: type[T] | None

    def __init__(self, response_schema: type[T] | None = None):
        """
        Initialize the adapter with an optional response schema.

        Args:
            response_schema: Optional Pydantic model class for parsing structured data.
        """
        self.response_schema = response_schema

    def adapt(
        self,
        _f: OpenRouterResponseMessage,
    ) -> GeneratedAssistantMessage[T]:
        """
        Convert an OpenRouter response message to a GeneratedAssistantMessage.

        Args:
            _f: The OpenRouter response message to convert.

        Returns:
            GeneratedAssistantMessage with content and optional parsed data.
        """
        openrouter_message = _f

        # Extract content
        content = openrouter_message.get("content") or ""

        # Extract reasoning if present (some models support this)
        reasoning = openrouter_message.get("reasoning")
        reasoning_details = openrouter_message.get("reasoning_details")

        # Extract tool calls if present
        tool_calls_data: Sequence[OpenRouterToolCall] = openrouter_message.get(
            "tool_calls", []
        )

        # Convert tool calls to ToolExecutionSuggestions
        tool_parts: list[ToolExecutionSuggestion] = []
        for tool_call in tool_calls_data:
            function_data = tool_call.get("function", {})

            # Parse arguments with error handling for malformed JSON
            args_str = str(function_data.get("arguments", "{}"))
            args: dict[str, object] = {}
            try:
                args = json.loads(args_str)
            except json.JSONDecodeError as e:
                # Log the error and try to extract the first valid JSON object
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(
                    f"Malformed JSON in tool call arguments: {e}. "
                    + "Attempting to parse first valid JSON object. "
                    + f"Raw arguments: {args_str[:200]}..."
                )

                # Try to find the first complete JSON object
                try:
                    # Use JSONDecoder to parse incrementally
                    decoder = json.JSONDecoder()
                    args, idx = decoder.raw_decode(args_str)
                    if idx < len(args_str.strip()):
                        logger.warning(
                            f"Extra data found after position {idx}. "
                            + "Using first valid JSON object only."
                        )
                except (json.JSONDecodeError, ValueError) as e2:
                    logger.error(
                        f"Failed to parse tool call arguments even with recovery: {e2}. "
                        + "Using empty dict."
                    )
                    args = {}

            tool_parts.append(
                ToolExecutionSuggestion(
                    id=str(tool_call.get("id", "")),
                    tool_name=str(function_data.get("name", "")),
                    args=args,
                )
            )

        # Handle structured output parsing
        parsed_data: T | None = None
        if self.response_schema and content:
            try:
                content_obj = json.loads(content)
                parsed_data = cast(
                    T,
                    cast(BaseModel, self.response_schema).model_validate(content_obj),
                )
            except (json.JSONDecodeError, ValueError):
                # If parsing fails, leave parsed_data as None
                pass

        # Build parts list
        parts: list[TextPart | ToolExecutionSuggestion] = (
            [TextPart(text=content)] if content else []
        )
        parts.extend(tool_parts)

        return GeneratedAssistantMessage[T](
            parts=parts,
            parsed=cast(T, parsed_data),
            reasoning=reasoning,
            reasoning_details=reasoning_details,
        )
