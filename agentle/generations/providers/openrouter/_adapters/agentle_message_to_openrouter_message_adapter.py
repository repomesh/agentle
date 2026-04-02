# Adapter for Agentle message to OpenRouter message
"""
Adapter for converting Agentle messages to OpenRouter message format.

This module handles the conversion of Agentle's message types
(AssistantMessage, DeveloperMessage, UserMessage) into OpenRouter's
API message format.
"""

from __future__ import annotations

import json
from typing import override

from rsb.adapters.adapter import Adapter

from agentle.generations.models.message_parts.file import FilePart
from agentle.generations.models.message_parts.text import TextPart
from agentle.generations.models.message_parts.tool_execution_suggestion import (
    ToolExecutionSuggestion,
)
from agentle.generations.models.messages.assistant_message import AssistantMessage
from agentle.generations.models.messages.developer_message import DeveloperMessage
from agentle.generations.models.messages.user_message import UserMessage
from agentle.generations.providers.openrouter._adapters.agentle_part_to_openrouter_part_adapter import (
    AgentlePartToOpenRouterPartAdapter,
)
from agentle.generations.providers.openrouter._types import (
    OpenRouterAssistantMessage,
    OpenRouterMessage,
    OpenRouterSystemMessage,
    OpenRouterToolCall,
    OpenRouterToolMessage,
    OpenRouterUserMessage,
)
from agentle.generations.tools.tool_execution_result import ToolExecutionResult


class AgentleMessageToOpenRouterMessageAdapter(
    Adapter[
        AssistantMessage | DeveloperMessage | UserMessage,
        OpenRouterMessage | list[OpenRouterMessage],
    ]
):
    """
    Adapter for converting Agentle messages to OpenRouter format.

    Handles conversion of:
    - DeveloperMessage -> OpenRouterSystemMessage
    - UserMessage -> OpenRouterUserMessage (or OpenRouterToolMessage if contains tool results)
    - AssistantMessage -> OpenRouterAssistantMessage (with tool calls)

    Note: When a message contains ToolExecutionResult parts, they are extracted
    and returned as separate OpenRouterToolMessage objects.
    """

    @override
    def adapt(
        self,
        _f: AssistantMessage | DeveloperMessage | UserMessage,
    ) -> OpenRouterMessage | list[OpenRouterMessage]:
        """
        Convert an Agentle message to OpenRouter format.

        Args:
            _f: The Agentle message to convert.

        Returns:
            The corresponding OpenRouter message(s). Returns a list when the message
            contains ToolExecutionResult parts that need to be split into separate
            tool messages.
        """
        message = _f
        part_adapter = AgentlePartToOpenRouterPartAdapter()

        match message:
            case DeveloperMessage():
                # Developer messages become system messages
                # Concatenate all text parts
                content = "".join(str(p) for p in message.parts)
                return OpenRouterSystemMessage(
                    role="system",
                    content=content,
                )

            case UserMessage():
                # Check if this message contains tool execution results
                tool_results = [
                    p for p in message.parts if isinstance(p, ToolExecutionResult)
                ]

                if tool_results:
                    # Convert each tool result to a separate tool message
                    return [
                        OpenRouterToolMessage(
                            role="tool",
                            tool_call_id=result.suggestion.id,
                            content=self._serialize_tool_result(result.result),
                        )
                        for result in tool_results
                    ]

                # User messages can have multimodal content
                # Filter out non-content parts (like tool execution suggestions and results)
                content_parts = [
                    p
                    for p in message.parts
                    if not isinstance(p, (ToolExecutionSuggestion, ToolExecutionResult))
                ]

                # If only text parts, concatenate into a string
                if all(isinstance(p, TextPart) for p in content_parts):
                    return OpenRouterUserMessage(
                        role="user",
                        content="".join(str(p) for p in content_parts),
                    )

                # Otherwise, convert to multimodal format
                return OpenRouterUserMessage(
                    role="user",
                    content=[
                        part_adapter.adapt(p)
                        for p in content_parts
                        if isinstance(p, TextPart) or isinstance(p, FilePart)
                    ],
                )

            case AssistantMessage():
                # Check if this message contains tool execution results
                tool_results = [
                    p for p in message.parts if isinstance(p, ToolExecutionResult)
                ]

                if tool_results:
                    # If assistant message has tool results, we need to split it
                    # First, create the assistant message with tool calls (if any)
                    messages: list[OpenRouterMessage] = []

                    # Separate text content from tool calls
                    text_parts = [p for p in message.parts if isinstance(p, TextPart)]
                    tool_suggestions = [
                        p
                        for p in message.parts
                        if isinstance(p, ToolExecutionSuggestion)
                    ]

                    # Only create assistant message if there's content or tool calls
                    if text_parts or tool_suggestions:
                        content = (
                            "".join(str(p) for p in text_parts) if text_parts else None
                        )

                        tool_calls: list[OpenRouterToolCall] = [
                            OpenRouterToolCall(
                                id=suggestion.id,
                                type="function",
                                function={
                                    "name": suggestion.tool_name,
                                    "arguments": self._serialize_tool_arguments(
                                        suggestion.args
                                    ),
                                },
                            )
                            for suggestion in tool_suggestions
                        ]

                        assistant_msg = OpenRouterAssistantMessage(
                            role="assistant",
                            content=content,
                        )

                        if tool_calls:
                            assistant_msg["tool_calls"] = tool_calls

                        self._apply_reasoning_fields(
                            assistant_msg,
                            message,
                        )

                        messages.append(assistant_msg)

                    # Add tool result messages
                    for result in tool_results:
                        messages.append(
                            OpenRouterToolMessage(
                                role="tool",
                                tool_call_id=result.suggestion.id,
                                content=self._serialize_tool_result(result.result),
                            )
                        )

                    return messages

                # Separate text content from tool calls
                text_parts = [p for p in message.parts if isinstance(p, TextPart)]
                tool_suggestions = [
                    p for p in message.parts if isinstance(p, ToolExecutionSuggestion)
                ]

                # Build content string from text parts
                content = "".join(str(p) for p in text_parts) if text_parts else None

                # Convert tool suggestions to OpenRouter tool calls
                tool_calls: list[OpenRouterToolCall] = [
                    OpenRouterToolCall(
                        id=suggestion.id,
                        type="function",
                        function={
                            "name": suggestion.tool_name,
                            "arguments": self._serialize_tool_arguments(
                                suggestion.args
                            ),
                        },
                    )
                    for suggestion in tool_suggestions
                ]

                result = OpenRouterAssistantMessage(
                    role="assistant",
                    content=content,
                )

                if tool_calls:
                    result["tool_calls"] = tool_calls

                self._apply_reasoning_fields(result, message)

                return result

    def _apply_reasoning_fields(
        self,
        target: OpenRouterAssistantMessage,
        message: AssistantMessage,
    ) -> None:
        if hasattr(message, "reasoning") and message.reasoning:
            target["reasoning"] = message.reasoning

        if hasattr(message, "reasoning_details") and message.reasoning_details:
            target["reasoning_details"] = list(message.reasoning_details)

    def _serialize_tool_arguments(self, args: object) -> str:
        """
        Serialize tool arguments to JSON string.

        Args:
            args: The arguments to serialize.

        Returns:
            JSON string representation of the arguments.
        """
        if isinstance(args, str):
            return args
        return json.dumps(args)

    def _serialize_tool_result(self, result: object) -> str:
        """
        Serialize tool execution result to string.

        Args:
            result: The result to serialize.

        returns:
            String representation of the result.
        """
        if isinstance(result, str):
            return result
        try:
            return json.dumps(result)
        except (TypeError, ValueError):
            return str(result)
