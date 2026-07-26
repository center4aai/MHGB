"""
Инфраструктура экспертной валидации задач MHGB (Шаг 6.2).

Компоненты:
- select_stratified_sample() — выборка по матрице 4 типа × 3 hop-группы
- compute_cohens_kappa()     — Cohen's κ между двумя рецензентами
- compute_agreement()        — простой процент согласия
- compare_llm_vs_expert()    — сравнение LLM-валидатора с экспертом
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any


TASK_TYPES = [
    "issue_spotting",
    "rule_selection",
    "conflict_resolution",
    "temporal_validity",
]

HOP_GROUPS = ["shallow", "medium", "deep"]


# ---------------------------------------------------------------------------
# Стратифицированная выборка
# ---------------------------------------------------------------------------

def select_stratified_sample(
    tasks: list[dict[str, Any]],
    n_per_cell: int = 3,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Стратифицированная выборка задач по матрице 4 типа × 3 hop-группы.

    Из каждой ячейки (type, hop_group) берётся до n_per_cell задач (режим closed-book).
    Если задач меньше — берутся все. Порядок фиксирован через seed.

    tasks      — список задач (dicts с полями mode, type, hop_group)
    n_per_cell — максимум задач на ячейку матрицы
    seed       — для воспроизводимости
    """
    rng = random.Random(seed)

    cells: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        if task.get("mode") == "closed":
            key = (task.get("type", ""), task.get("hop_group", ""))
            cells[key].append(task)

    sample: list[dict[str, Any]] = []
    for task_type in TASK_TYPES:
        for hop_group in HOP_GROUPS:
            cell = cells.get((task_type, hop_group), [])
            n = min(n_per_cell, len(cell))
            if n > 0:
                sample.extend(rng.sample(cell, n))

    return sample


# ---------------------------------------------------------------------------
# Метрики согласия
# ---------------------------------------------------------------------------

def compute_cohens_kappa(
    judgments_a: list[bool],
    judgments_b: list[bool],
) -> float:
    """
    Cohen's κ между двумя рецензентами.

    judgments_a, judgments_b — списки булевых оценок (одинаковой длины).
    Возвращает κ ∈ [-1, 1]. Возвращает 0.0 если n=0 или p_e=1.
    """
    if len(judgments_a) != len(judgments_b):
        raise ValueError(
            f"Длины списков оценок не совпадают: "
            f"{len(judgments_a)} vs {len(judgments_b)}"
        )
    n = len(judgments_a)
    if n == 0:
        return 0.0

    p_o = sum(a == b for a, b in zip(judgments_a, judgments_b)) / n

    p_yes_a = sum(judgments_a) / n
    p_yes_b = sum(judgments_b) / n
    p_e = p_yes_a * p_yes_b + (1 - p_yes_a) * (1 - p_yes_b)

    if p_e >= 1.0:
        return 0.0

    return (p_o - p_e) / (1.0 - p_e)


def compute_agreement(
    judgments_a: list[bool],
    judgments_b: list[bool],
) -> float:
    """
    Простой процент согласия двух рецензентов.
    Возвращает значение в [0, 1].
    """
    if len(judgments_a) != len(judgments_b):
        raise ValueError(
            f"Длины списков оценок не совпадают: "
            f"{len(judgments_a)} vs {len(judgments_b)}"
        )
    n = len(judgments_a)
    if n == 0:
        return 0.0
    return sum(a == b for a, b in zip(judgments_a, judgments_b)) / n


# ---------------------------------------------------------------------------
# Сравнение LLM-валидатора с экспертом
# ---------------------------------------------------------------------------

def compare_llm_vs_expert(
    llm_results: list[dict[str, Any]],
    expert_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Сравнивает результаты LLM-валидатора с экспертными оценками.

    llm_results, expert_results — списки dicts с полями task_id и is_valid.
    Используется пересечение task_id из обоих источников.

    Возвращает dict с полями:
      total, agreement, cohens_kappa, counts
    """
    llm_map = {r["task_id"]: bool(r["is_valid"]) for r in llm_results}
    expert_map = {r["task_id"]: bool(r["is_valid"]) for r in expert_results}

    common_ids = sorted(set(llm_map) & set(expert_map))
    llm_j   = [llm_map[tid] for tid in common_ids]
    expert_j = [expert_map[tid] for tid in common_ids]

    return {
        "total": len(common_ids),
        "agreement": compute_agreement(llm_j, expert_j),
        "cohens_kappa": compute_cohens_kappa(llm_j, expert_j),
        "counts": {
            "both_valid":               sum(a and b      for a, b in zip(llm_j, expert_j)),
            "both_rejected":            sum(not a and not b for a, b in zip(llm_j, expert_j)),
            "llm_valid_expert_rejected": sum(a and not b  for a, b in zip(llm_j, expert_j)),
            "llm_rejected_expert_valid": sum(not a and b  for a, b in zip(llm_j, expert_j)),
        },
    }
