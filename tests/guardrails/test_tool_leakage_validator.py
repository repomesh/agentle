import pytest

from agentle.guardrails.core.guardrail_result import GuardrailAction
from agentle.guardrails.validators.tool_leakage_validator import ToolLeakageValidator


@pytest.mark.asyncio
async def test_tool_leakage_validator_sanitizes_rendered_tool_call() -> None:
    validator = ToolLeakageValidator(tool_names=["registrar_agendamento"])

    result = await validator.validate_async(
        "Tool: registrar_agendamento\n"
        "Args: {'data_hora_inicio': '2026-05-19 11:15', 'agenda_titulo': 'RX 2'}",
        {},
    )

    assert result.action == GuardrailAction.MODIFY
    assert result.modified_content == ""


@pytest.mark.asyncio
async def test_tool_leakage_validator_sanitizes_textual_tool_payload() -> None:
    validator = ToolLeakageValidator(tool_names=["registrar_agendamento"])

    result = await validator.validate_async(
        "Pronto.\nTool: registrar_agendamento\nArguments: {'nome': 'Maria'}",
        {},
    )

    assert result.action == GuardrailAction.MODIFY
    assert result.modified_content == "Pronto."
