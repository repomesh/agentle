"""
Gerenciador principal do sistema de guardrails.
"""

import asyncio
from collections.abc import Sequence
from typing import Any, cast

from agentle.guardrails.core.guardrail_validator import GuardrailValidator
from agentle.guardrails.core.input_guardrail_validator import InputGuardrailValidator
from agentle.guardrails.core.output_guardrail_validator import OutputGuardrailValidator
from agentle.guardrails.core.guardrail_result import GuardrailResult, GuardrailAction
from agentle.guardrails.core.guardrail_error import GuardrailViolationError
from agentle.guardrails.core.guardrail_metrics import GuardrailMetrics


class GuardrailManager:
    """
    Gerenciador central para todos os guardrails.

    Coordena a execução de múltiplos validadores, gerencia cache,
    coleta métricas e toma decisões baseadas nos resultados.
    """

    def __init__(
        self,
        fail_fast: bool = True,
        parallel_execution: bool = True,
        cache_enabled: bool = True,
        max_cache_size: int = 1000,
    ):
        self.input_validators: list[InputGuardrailValidator] = []
        self.output_validators: list[OutputGuardrailValidator] = []

        self.fail_fast = fail_fast
        self.parallel_execution = parallel_execution
        self.cache_enabled = cache_enabled
        self.max_cache_size = max_cache_size

        self.metrics = GuardrailMetrics()
        self._cache: dict[str, GuardrailResult] = {}

    def add_input_validator(self, validator: InputGuardrailValidator) -> None:
        """Adiciona um validador de entrada."""
        self.input_validators.append(validator)
        self.input_validators.sort(key=lambda x: x.priority)

    def add_output_validator(self, validator: OutputGuardrailValidator) -> None:
        """Adiciona um validador de saída."""
        self.output_validators.append(validator)
        self.output_validators.sort(key=lambda x: x.priority)

    def remove_validator(self, name: str) -> bool:
        """Remove um validador pelo nome."""
        # Remover dos validadores de entrada
        for i, validator in enumerate(self.input_validators):
            if validator.name == name:
                del self.input_validators[i]
                return True

        # Remover dos validadores de saída
        for i, validator in enumerate(self.output_validators):
            if validator.name == name:
                del self.output_validators[i]
                return True

        return False

    def get_validator(self, name: str) -> GuardrailValidator | None:
        """Obtém um validador pelo nome."""
        for validator in self.input_validators + self.output_validators:
            if validator.name == name:
                return validator
        return None

    async def validate_input_async(
        self,
        content: str,
        context: dict[str, Any] | None = None,
        raise_on_violation: bool = True,
    ) -> str | GuardrailResult:
        """
        Valida entrada do usuário usando todos os validadores de entrada.

        Args:
            content: Conteúdo a ser validado
            context: Contexto adicional
            raise_on_violation: Se deve lançar exceção em violações

        Returns:
            String processada ou GuardrailResult em caso de violação

        Raises:
            GuardrailViolationError: Se uma violação for detectada e raise_on_violation=True
        """
        self.metrics.input_validations += 1
        return await self._validate_with_validators(
            content, self.input_validators, context, raise_on_violation
        )

    async def validate_output_async(
        self,
        content: str,
        context: dict[str, Any] | None = None,
        raise_on_violation: bool = True,
    ) -> str | GuardrailResult:
        """
        Valida saída do modelo usando todos os validadores de saída.
        """
        self.metrics.output_validations += 1
        return await self._validate_with_validators(
            content, self.output_validators, context, raise_on_violation
        )

    async def _validate_with_validators(
        self,
        content: str,
        validators: Sequence[GuardrailValidator],
        context: dict[str, Any] | None = None,
        raise_on_violation: bool = True,
    ) -> str | GuardrailResult:
        """
        Executa validação com uma lista de validadores.
        """
        if not validators:
            return content

        # Filtrar validadores habilitados
        active_validators = [v for v in validators if v.enabled]
        if not active_validators:
            return content

        processed_content = content
        all_results: list[GuardrailResult] = []

        if self.parallel_execution:
            # Execução paralela
            tasks = [
                validator.validate_async(processed_content, context)
                for validator in active_validators
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    # Tratar erro como bloqueio
                    error_result = GuardrailResult(
                        action=GuardrailAction.BLOCK,
                        confidence=1.0,
                        reason=f"Validation error: {str(result)}",
                        validator_name=active_validators[i].name,
                        metadata={"error": str(result)},
                    )
                    all_results.append(error_result)
                else:
                    all_results.append(cast(GuardrailResult, result))
        else:
            # Execução sequencial
            for validator in active_validators:
                try:
                    result = await validator.validate_async(processed_content, context)
                    all_results.append(result)

                    # Atualizar métricas
                    self.metrics.update_validation_metrics(validator.name, result)

                    if result.should_modify and result.modified_content is not None:
                        processed_content = result.modified_content

                    if result.should_block and self.fail_fast:
                        if raise_on_violation:
                            raise GuardrailViolationError(result)
                        return result

                except Exception as e:
                    error_result = GuardrailResult(
                        action=GuardrailAction.BLOCK,
                        confidence=1.0,
                        reason=f"Validation error in {validator.name}: {str(e)}",
                        validator_name=validator.name,
                        metadata={"error": str(e)},
                    )
                    all_results.append(error_result)

                    if self.fail_fast:
                        if raise_on_violation:
                            raise GuardrailViolationError(error_result)
                        return error_result

        # Processar todos os resultados
        final_result = self._process_results(all_results, processed_content)

        if final_result.should_block and raise_on_violation:
            raise GuardrailViolationError(final_result)

        if final_result.should_block or final_result.action == GuardrailAction.WARN:
            return final_result

        if final_result.modified_content is not None:
            return final_result.modified_content

        return processed_content

    def _process_results(
        self, results: list[GuardrailResult], content: str
    ) -> GuardrailResult:
        """
        Processa múltiplos resultados de validação em uma decisão final.
        """
        if not results:
            return GuardrailResult(
                action=GuardrailAction.ALLOW,
                confidence=1.0,
                reason="No validators executed",
                validator_name="system",
            )

        # Verificar se algum resultado requer bloqueio
        blocking_results = [r for r in results if r.should_block]
        if blocking_results:
            # Retornar o resultado de bloqueio com maior confiança
            worst_result = max(blocking_results, key=lambda x: x.confidence)
            return worst_result

        # Verificar modificações
        modified_content = content
        modifications: list[str] = []

        for result in results:
            if result.should_modify and result.modified_content is not None:
                modified_content = result.modified_content
                modifications.append(result.validator_name)

        # Verificar avisos
        warnings = [r for r in results if r.action == GuardrailAction.WARN]

        if modifications:
            return GuardrailResult(
                action=GuardrailAction.MODIFY,
                confidence=0.8,
                reason=f"Content modified by: {', '.join(modifications)}",
                validator_name="system",
                modified_content=modified_content,
                metadata={"modifications": modifications},
            )

        if warnings:
            worst_warning = max(warnings, key=lambda x: x.confidence)
            return worst_warning

        # Tudo passou - permitir
        return GuardrailResult(
            action=GuardrailAction.ALLOW,
            confidence=1.0,
            reason="All validations passed",
            validator_name="system",
            modified_content=modified_content,
        )

    def _generate_cache_key(self, content: str, validator_name: str) -> str:
        """Gera chave de cache para conteúdo e validador."""
        import hashlib

        content_hash = hashlib.md5(content.encode()).hexdigest()
        return f"{validator_name}:{content_hash}"

    def clear_cache(self) -> None:
        """Limpa o cache de validações."""
        self._cache.clear()

    def get_metrics_summary(self) -> dict[str, Any]:
        """Retorna resumo das métricas."""
        return {
            "total_validations": self.metrics.total_validations,
            "input_validations": self.metrics.input_validations,
            "output_validations": self.metrics.output_validations,
            "total_blocks": self.metrics.total_blocks,
            "total_modifications": self.metrics.total_modifications,
            "total_warnings": self.metrics.total_warnings,
            "block_rate": self.metrics.block_rate,
            "average_processing_time_ms": self.metrics.average_processing_time_ms,
            "cache_hit_rate": self.metrics.cache_hit_rate,
            "validator_metrics": self.metrics.validator_metrics,
            "top_block_reasons": dict(
                list(
                    sorted(
                        self.metrics.block_reasons.items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )
                )[:5]
            ),
        }
