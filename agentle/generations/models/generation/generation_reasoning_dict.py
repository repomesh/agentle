"""
TypedDict counterpart for generation reasoning configuration.
"""

from typing import Literal, NotRequired, TypedDict


class GenerationReasoningDict(TypedDict):
    """
    Dictionary form of reasoning configuration for generation requests.
    """

    effort: NotRequired[
        Literal["none", "minimal", "low", "medium", "high", "xhigh"] | None
    ]
    max_tokens: NotRequired[int | None]
    exclude: NotRequired[bool | None]
    enabled: NotRequired[bool | None]
