"""
Reasoning configuration for generation requests.

This module defines the GenerationReasoning model, which encapsulates provider-
agnostic controls for reasoning or thinking features in model requests.
"""

from __future__ import annotations

from typing import Literal

from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class GenerationReasoning(BaseModel):
    """
    Controls reasoning or thinking behavior for providers that support it.

    Attributes:
        effort: Abstract effort level for reasoning-intensive models.
        max_tokens: Explicit reasoning token budget for providers that support it.
        exclude: Whether reasoning should be hidden from the returned response.
        enabled: Whether reasoning should be enabled with provider defaults.
    """

    effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] | None = (
        Field(
            default=None,
            description="Reasoning effort level for providers that support effort-based reasoning controls.",
        )
    )

    max_tokens: int | None = Field(
        default=None,
        gt=0,
        description="Explicit budget for reasoning tokens on providers that support token-based reasoning controls.",
    )

    exclude: bool | None = Field(
        default=None,
        description="Whether reasoning should be used internally but excluded from the response payload.",
    )

    enabled: bool | None = Field(
        default=None,
        description="Whether reasoning should be enabled with provider defaults when no explicit effort or token budget is supplied.",
    )
