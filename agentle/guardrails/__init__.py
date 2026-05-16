# agentle/guardrails/__init__.py
"""
Sistema de Guardrails nativo para Agentle Framework.

Este módulo fornece um sistema completo de validação e segurança para agentes de IA,
incluindo validação de entrada, saída, e controles de segurança em tempo real.
"""

from .core.guardrail_result import GuardrailResult, GuardrailAction
from .core.guardrail_validator import GuardrailValidator
from .core.input_guardrail_validator import InputGuardrailValidator
from .core.output_guardrail_validator import OutputGuardrailValidator
from .core.guardrail_manager import GuardrailManager
from .core.guardrail_error import GuardrailError, GuardrailViolationError
from .core.guardrail_metrics import GuardrailMetrics

# Validadores built-in
from .validators.content_safety_validator import ContentSafetyValidator
from .validators.pii_detection_validator import PIIDetectionValidator
from .validators.toxicity_validator import ToxicityValidator
from .validators.prompt_injection_validator import PromptInjectionValidator
from .validators.response_quality_validator import ResponseQualityValidator
from .validators.tool_leakage_validator import ToolLeakageValidator

__all__ = [
    # Core
    "GuardrailResult",
    "GuardrailAction",
    "GuardrailValidator",
    "InputGuardrailValidator",
    "OutputGuardrailValidator",
    "GuardrailManager",
    "GuardrailError",
    "GuardrailViolationError",
    "GuardrailMetrics",
    # Built-in Validators
    "ContentSafetyValidator",
    "PIIDetectionValidator",
    "ToxicityValidator",
    "PromptInjectionValidator",
    "ResponseQualityValidator",
    "ToolLeakageValidator",
]
