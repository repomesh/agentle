import pytest

from agentle.generations.models.generation.generation_config import GenerationConfig


def test_generation_config_accepts_reasoning_dict():
    config = GenerationConfig(reasoning={"effort": "high", "exclude": True})

    assert config.reasoning is not None
    assert config.reasoning.effort == "high"
    assert config.reasoning.exclude is True


def test_generation_config_rejects_effort_and_max_tokens_together():
    with pytest.raises(
        ValueError,
        match="Only one of reasoning.effort or reasoning.max_tokens should be set.",
    ):
        GenerationConfig(reasoning={"effort": "high", "max_tokens": 2048})


def test_generation_config_clone_preserves_reasoning():
    config = GenerationConfig(
        reasoning={"effort": "medium"},
        trace_params={"name": "reasoning-test"},
    )

    cloned = config.clone(new_trace_params={"session_id": "session-1"})

    assert cloned.reasoning is not None
    assert cloned.reasoning.effort == "medium"
    assert cloned.trace_params["name"] == "reasoning-test"
    assert cloned.trace_params["session_id"] == "session-1"
