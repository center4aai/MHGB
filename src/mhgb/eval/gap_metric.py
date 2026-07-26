"""
GAP-метрика — разрыв между closed-book и open-book результатами.

compute_gap             — простая разница open - closed
classify_gap_quadrant   — классификация по 4 квадрантам
compute_all_gaps        — GAP по всем 4 компонентам из FinalScoreResult
aggregate_gaps_by_slice — агрегация батча результатов по разрезам
                          (модель × тип задачи × hop-группа × отрасль права)
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Literal, TypedDict


# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

GAP_FIELDS = (
    "norm_coverage_gap",
    "step_correctness_gap",
    "answer_correctness_gap",
    "final_score_gap",
)

Quadrant = Literal["knows", "reasons", "hallucinates", "incompetent"]


# ---------------------------------------------------------------------------
# Типы
# ---------------------------------------------------------------------------

class GapResult(TypedDict):
    norm_coverage_gap: float
    step_correctness_gap: float
    answer_correctness_gap: float
    final_score_gap: float
    quadrant: str


class SliceStats(TypedDict):
    count: int
    norm_coverage_gap_mean: float
    norm_coverage_gap_std: float
    step_correctness_gap_mean: float
    step_correctness_gap_std: float
    answer_correctness_gap_mean: float
    answer_correctness_gap_std: float
    final_score_gap_mean: float
    final_score_gap_std: float
    quadrants: dict[str, int]


# ---------------------------------------------------------------------------
# Базовые функции
# ---------------------------------------------------------------------------

def compute_gap(closed_score: float, open_score: float) -> float:
    """Разница open − closed. Положительная → open лучше."""
    return open_score - closed_score


def classify_gap_quadrant(
    closed: float,
    open: float,
    threshold: float = 0.5,
) -> Quadrant:
    """
    Квадрант интерпретации GAP-метрики.

    knows        — оба выше threshold: модель и так знает закон
    reasons      — только open выше: умеет рассуждать, но не знает закон
    hallucinates — только closed выше: знает закон, плохо рассуждает по тексту
    incompetent  — оба ниже threshold: не умеет правоприменительное рассуждение
    """
    closed_high = closed >= threshold
    open_high = open >= threshold

    if closed_high and open_high:
        return "knows"
    if not closed_high and open_high:
        return "reasons"
    if closed_high and not open_high:
        return "hallucinates"
    return "incompetent"


# ---------------------------------------------------------------------------
# Агрегация по FinalScoreResult
# ---------------------------------------------------------------------------

def compute_all_gaps(
    closed_result: dict,
    open_result: dict,
    threshold: float = 0.5,
) -> GapResult:
    """
    Вычисляет GAP по всем 4 компонентам из пары FinalScoreResult.

    closed_result / open_result должны содержать ключи:
        norm_coverage, step_correctness, answer_correctness, final_score
    """
    return {
        "norm_coverage_gap": compute_gap(
            closed_result["norm_coverage"], open_result["norm_coverage"]
        ),
        "step_correctness_gap": compute_gap(
            closed_result["step_correctness"], open_result["step_correctness"]
        ),
        "answer_correctness_gap": compute_gap(
            closed_result["answer_correctness"], open_result["answer_correctness"]
        ),
        "final_score_gap": compute_gap(
            closed_result["final_score"], open_result["final_score"]
        ),
        "quadrant": classify_gap_quadrant(
            closed_result["final_score"], open_result["final_score"], threshold
        ),
    }


# ---------------------------------------------------------------------------
# Аналитика по разрезам
# ---------------------------------------------------------------------------

def aggregate_gaps_by_slice(
    records: list[dict],
    by: list[str],
) -> dict[tuple, SliceStats]:
    """
    Агрегирует батч GapResult-записей по произвольным разрезам.

    records — список dict, каждый содержит поля GapResult + метаданные
              (например: model, task_type, hop_group, branch_of_law)
    by      — список ключей для группировки, например ["model", "task_type"]

    Возвращает dict: ключ-кортеж значений → SliceStats со средним, std и
    распределением квадрантов для каждой группы.

    Пример:
        aggregate_gaps_by_slice(records, by=["model", "hop_group"])
        → {("gpt-4", "shallow"): {"count": 10, "final_score_gap_mean": 0.15, ...}}
    """
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        key = tuple(str(record.get(k, "")) for k in by)
        groups[key].append(record)

    result: dict[tuple, SliceStats] = {}
    for key, group in groups.items():
        stats: dict = {"count": len(group)}

        for field in GAP_FIELDS:
            values = [r[field] for r in group if field in r]
            if values:
                stats[f"{field}_mean"] = statistics.mean(values)
                stats[f"{field}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            else:
                stats[f"{field}_mean"] = 0.0
                stats[f"{field}_std"] = 0.0

        quadrant_counts: dict[str, int] = {}
        for r in group:
            q = r.get("quadrant", "")
            quadrant_counts[q] = quadrant_counts.get(q, 0) + 1
        stats["quadrants"] = quadrant_counts

        result[key] = stats

    return result
