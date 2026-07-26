"""
Final Score — композитная метрика (Уровень 1+2+3).

Формула: final_score = α·norm_coverage + β·step_correctness + γ·answer_correctness
Веса по умолчанию: α = β = γ = 1/3

В отчётах все три компонента всегда выводятся отдельно.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict


# ---------------------------------------------------------------------------
# Веса
# ---------------------------------------------------------------------------

@dataclass
class EvalWeights:
    """Веса для трёх компонент итоговой метрики. Должны суммироваться в 1.0."""

    norm_coverage: float = 1 / 3
    step_correctness: float = 1 / 3
    answer_correctness: float = 1 / 3

    def total(self) -> float:
        return self.norm_coverage + self.step_correctness + self.answer_correctness

    def is_valid(self, tol: float = 1e-6) -> bool:
        return abs(self.total() - 1.0) <= tol


DEFAULT_WEIGHTS = EvalWeights()


# ---------------------------------------------------------------------------
# Результат
# ---------------------------------------------------------------------------

class FinalScoreResult(TypedDict):
    norm_coverage: float        # компонента 1
    step_correctness: float     # компонента 2
    answer_correctness: float   # компонента 3
    final_score: float          # взвешенная сумма
    weights: dict[str, float]   # применённые веса


# ---------------------------------------------------------------------------
# Основная функция
# ---------------------------------------------------------------------------

def compute_final_score(
    norm_coverage: float,
    step_correctness: float,
    answer_correctness: float,
    weights: EvalWeights | None = None,
) -> FinalScoreResult:
    """
    Агрегирует три метрики в единый Final Score.

    norm_coverage     — F1 из compute_list_coverage или compute_reasoning_coverage
    step_correctness  — из StepCorrectnessResult['step_correctness']
    answer_correctness — из AnswerCorrectnessResult['score']
    weights           — веса компонент (по умолчанию 1/3 каждая)

    Предупреждение: при weights.total() != 1.0 результат выходит за [0, 1].
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    final = (
        weights.norm_coverage     * norm_coverage
        + weights.step_correctness  * step_correctness
        + weights.answer_correctness * answer_correctness
    )

    return {
        "norm_coverage":      norm_coverage,
        "step_correctness":   step_correctness,
        "answer_correctness": answer_correctness,
        "final_score":        final,
        "weights": {
            "norm_coverage":      weights.norm_coverage,
            "step_correctness":   weights.step_correctness,
            "answer_correctness": weights.answer_correctness,
        },
    }
