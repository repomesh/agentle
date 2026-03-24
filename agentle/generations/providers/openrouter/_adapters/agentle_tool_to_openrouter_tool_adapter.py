"""
Adapter for converting Agentle Tool definitions to OpenRouter tool format.

This module handles the conversion of Agentle's Tool objects into
the OpenRouter API tool definition format.
"""

from __future__ import annotations

import inspect
import logging
import typing
from typing import Any, Union, get_args, get_origin, override, cast

from rsb.adapters.adapter import Adapter

from agentle.generations.json.json_schema_builder import JsonSchemaBuilder
from agentle.generations.tools.tool import Tool
from agentle.generations.providers.openrouter._types import (
    OpenRouterTool,
    OpenRouterToolFunction,
    OpenRouterToolFunctionParameters,
)

logger = logging.getLogger(__name__)


class AgentleToolToOpenRouterToolAdapter(Adapter[Tool, OpenRouterTool]):
    """
    Adapter for converting Agentle Tool objects to OpenRouter format.

    Converts tool definitions including name, description, and parameters
    to the format expected by OpenRouter's API.

    This adapter handles both flat parameter format (from Tool.from_callable)
    and JSON Schema format. It also handles complex types like BaseModel,
    TypedDict, dataclasses, Literal types, and Optional[Literal[...]].
    """

    def _is_optional_type(self, type_annotation: Any) -> bool:
        """
        Check if a type annotation is an Optional type (Union[X, None]).

        Args:
            type_annotation: The type annotation to check.

        Returns:
            True if the type is Optional.
        """
        try:
            origin = get_origin(type_annotation)
            if origin is Union:
                args = get_args(type_annotation)
                # Optional[X] is Union[X, None]
                return type(None) in args
            return False
        except Exception:
            return False

    def _extract_optional_inner_type(self, type_annotation: Any) -> Any | None:
        """
        Extract the inner type from an Optional type.

        Args:
            type_annotation: The Optional type annotation.

        Returns:
            The inner type (without None), or None if extraction fails.
        """
        try:
            args = get_args(type_annotation)
            # Filter out NoneType
            non_none_args = [arg for arg in args if arg is not type(None)]
            if len(non_none_args) == 1:
                return non_none_args[0]
            elif len(non_none_args) > 1:
                # Union with more than 2 types (excluding None)
                return Union[tuple(non_none_args)]  # type: ignore
            return None
        except Exception:
            return None

    def _is_literal_type(self, type_annotation: Any) -> bool:
        """
        Check if a type annotation is a Literal type.

        Args:
            type_annotation: The type annotation to check.

        Returns:
            True if the type is a Literal type.
        """
        try:
            return get_origin(type_annotation) is typing.Literal
        except Exception:
            return False

    def _extract_literal_values(self, type_annotation: Any) -> list[Any]:
        """
        Extract values from a Literal type.

        Args:
            type_annotation: The Literal type annotation.

        Returns:
            List of literal values.
        """
        try:
            return list(get_args(type_annotation))
        except Exception:
            return []

    def _is_complex_type(self, type_annotation: Any) -> bool:
        """
        Check if a type annotation represents a complex type that needs expansion.

        Args:
            type_annotation: The type annotation to check.

        Returns:
            True if the type is complex (BaseModel, TypedDict, dataclass, etc.).
        """
        if not inspect.isclass(type_annotation):
            return False

        # Check for common complex types
        try:
            # Check for BaseModel (Pydantic)
            if hasattr(type_annotation, "model_fields") or hasattr(
                type_annotation, "__fields__"
            ):
                return True

            # Check for TypedDict
            if hasattr(type_annotation, "__annotations__") and hasattr(
                type_annotation, "__required_keys__"
            ):
                return True

            # Check for dataclass
            if hasattr(type_annotation, "__dataclass_fields__"):
                return True

        except Exception:
            pass

        return False

    def _expand_complex_type(self, type_annotation: Any) -> dict[str, Any]:
        """
        Expand a complex type to its JSON schema representation.

        Args:
            type_annotation: The complex type to expand.

        Returns:
            JSON schema representation of the type.
        """
        try:
            schema = JsonSchemaBuilder(
                type_annotation,
                clean_output=True,
                use_defs_instead_of_definitions=True,
            ).build(dereference=True)

            # Remove the $defs key if present since we dereferenced
            schema.pop("$defs", None)
            schema.pop("definitions", None)

            logger.debug(
                f"Expanded complex type {type_annotation.__name__} to JSON schema"
            )
            return schema

        except Exception as e:
            logger.warning(
                f"Failed to expand complex type {type_annotation}: {e}. "
                + "Falling back to generic object type."
            )
            return {"type": "object"}

    def _resolve_type_annotation(self, type_str: str, tool: Tool) -> Any | None:
        """
        Resolve a type string to the actual type object.

        Args:
            type_str: String representation of the type.
            tool: The tool object (to access the callable's scope if needed).

        Returns:
            The resolved type object, or None if it cannot be resolved.
        """
        # Try to get the callable's module for type resolution
        if not tool.callable_ref:
            return None

        try:
            # Get the signature to access parameter annotations
            sig = inspect.signature(tool.callable_ref)
            for _, param in sig.parameters.items():
                if param.annotation != inspect.Parameter.empty:
                    # Check if this parameter's annotation matches our type string
                    annotation_str = (
                        str(param.annotation).replace("<class '", "").replace("'>", "")
                    )
                    if (
                        annotation_str == type_str
                        or getattr(param.annotation, "__name__", "")
                        == type_str.split(".")[-1]
                    ):
                        return param.annotation
        except Exception as e:
            logger.debug(f"Could not resolve type annotation {type_str}: {e}")

        return None

    def _convert_to_json_schema(
        self, agentle_params: dict[str, Any], tool: Tool
    ) -> dict[str, Any]:
        """
        Convert Agentle's flat parameter format to proper JSON Schema format.

        Agentle format:
        {
            'param1': {'type': 'str', 'required': True, 'description': '...'},
            'param2': {'type': 'int', 'required': False, 'default': 42}
        }

        JSON Schema format:
        {
            'type': 'object',
            'properties': {
                'param1': {'type': 'string', 'description': '...'},
                'param2': {'type': 'integer', 'default': 42}
            },
            'required': ['param1']
        }

        This method also handles complex types like BaseModel, TypedDict, Literal,
        Optional[Literal[...]], etc. by expanding them to their full JSON schema representation.

        Args:
            agentle_params: Parameters in Agentle's flat format or JSON Schema format.
            tool: The tool object (for resolving complex type annotations).

        Returns:
            Parameters in JSON Schema format.
        """
        # Check if this is already in JSON Schema format
        if "type" in agentle_params and "properties" in agentle_params:
            return agentle_params

        # Check if it's a $schema format (also JSON Schema)
        if "$schema" in agentle_params or "properties" in agentle_params:
            if "type" not in agentle_params:
                result = {"type": "object"}
                result.update(agentle_params)
                return result
            return agentle_params

        # Convert from Agentle flat format to JSON Schema format
        properties: dict[str, Any] = {}
        required: list[str] = []

        # Type mapping from Python types to JSON Schema types
        type_mapping = {
            "str": "string",
            "string": "string",
            "int": "integer",
            "integer": "integer",
            "float": "number",
            "number": "number",
            "bool": "boolean",
            "boolean": "boolean",
            "list": "array",
            "array": "array",
            "dict": "object",
            "object": "object",
            "none": "null",
            "nonetype": "null",
            "null": "null",
        }

        for param_name, param_info in agentle_params.items():
            if not isinstance(param_info, dict):
                continue

            # Extract the parameter info
            param_type_str: str = cast(str, param_info.get("type", "string"))
            is_required = param_info.get("required", False)

            # Create the property schema
            prop_schema: dict[str, Any] = {}

            # Resolve the type annotation
            type_annotation = self._resolve_type_annotation(param_type_str, tool)

            # Check if this is an Optional type
            is_optional = False
            inner_type = type_annotation
            
            if type_annotation and self._is_optional_type(type_annotation):
                is_optional = True
                inner_type = self._extract_optional_inner_type(type_annotation)
                logger.debug(
                    f"Detected Optional type for parameter '{param_name}', inner type: {inner_type}"
                )

            # Now check the inner type (which might be the original type if not Optional)
            if inner_type and self._is_literal_type(inner_type):
                # Handle Literal types by extracting enum values
                enum_values = self._extract_literal_values(inner_type)
                if enum_values:
                    # Infer JSON type from first value
                    first_value = enum_values[0]
                    if isinstance(first_value, bool):
                        json_type = "boolean"
                    elif isinstance(first_value, int):
                        json_type = "integer"
                    elif isinstance(first_value, float):
                        json_type = "number"
                    else:
                        json_type = "string"
                    
                    if is_optional:
                        # For Optional[Literal[...]], use anyOf with enum and null
                        prop_schema = {
                            "anyOf": [
                                {"type": json_type, "enum": enum_values},
                                {"type": "null"}
                            ]
                        }
                        logger.debug(
                            f"Converted Optional[Literal] type for parameter '{param_name}' to anyOf with enum: {enum_values}"
                        )
                    else:
                        # Regular Literal
                        prop_schema = {
                            "type": json_type,
                            "enum": enum_values
                        }
                        logger.debug(
                            f"Converted Literal type for parameter '{param_name}' to enum: {enum_values}"
                        )
                else:
                    # Empty Literal (shouldn't happen, but handle gracefully)
                    prop_schema = {"type": "string"}
            elif inner_type and (
                self._is_complex_type(inner_type)
                or get_origin(inner_type) in (list, tuple, set, typing.List, typing.Tuple, typing.Set, dict, typing.Dict, frozenset, typing.FrozenSet, Union)
                or inner_type in (list, tuple, set, dict, frozenset)
            ):
                # Expand the complex type to its full JSON schema
                logger.debug(
                    f"Expanding generic or complex type for parameter '{param_name}': {param_type_str}"
                )
                prop_schema = self._expand_complex_type(inner_type)
                
                # If it was Optional, add null to the type
                if is_optional:
                    if "type" in prop_schema:
                        current_type = prop_schema["type"]
                        if isinstance(current_type, list):
                            if "null" not in current_type:
                                prop_schema["type"] = current_type + ["null"]
                        else:
                            prop_schema["type"] = [current_type, "null"]
            else:
                # Map the type to JSON Schema type
                if "|" in param_type_str:
                    # Handle union types like "str | None"
                    parts = [p.strip() for p in param_type_str.split("|")]
                    json_types: list[str] = []
                    
                    for part in parts:
                        mapped = type_mapping.get(part.lower(), part)
                        if mapped not in json_types:
                            json_types.append(mapped)
                    
                    if len(json_types) == 1:
                        prop_schema["type"] = json_types[0]
                    else:
                        prop_schema["type"] = json_types
                else:
                    json_type = type_mapping.get(param_type_str.lower(), param_type_str)
                    prop_schema["type"] = json_type

            # Copy over other attributes (excluding 'required' and 'type')
            for key, value in param_info.items():
                if key not in ("required", "type"):
                    # Don't overwrite if already set by Literal or complex type expansion
                    if key not in prop_schema:
                        prop_schema[key] = value

            properties[param_name] = prop_schema

            if is_required:
                required.append(param_name)

        result: dict[str, Any] = {"type": "object", "properties": properties}

        if required:
            result["required"] = required

        return result

    @override
    def adapt(self, tool: Tool) -> OpenRouterTool:
        """
        Convert an Agentle Tool to OpenRouter format.

        Args:
            tool: The Agentle Tool to convert.

        Returns:
            The corresponding OpenRouter tool definition.
        """
        # Convert parameters to JSON Schema format
        json_schema_params = self._convert_to_json_schema(tool.parameters, tool)

        return OpenRouterTool(
            type="function",
            function=OpenRouterToolFunction(
                name=tool.name,
                description=tool.description or "",
                parameters=OpenRouterToolFunctionParameters(
                    type="object",
                    properties=json_schema_params.get("properties", {}),
                    required=json_schema_params.get("required", []),
                ),
            ),
        )