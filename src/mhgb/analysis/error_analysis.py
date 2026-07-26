"""
Таксономия ошибок: классификация по доминирующему компоненту провала.

Категории (в порядке проверки):
  hallucination  — norm_coverage_reasoning >> norm_coverage_list (gap > 0.3)
  multi_fail     — все компоненты провалились одновременно
  norm_miss      — модель не нашла нужные нормы (norm_coverage_list < 0.3)
  reasoning_fail — нормы найдены, но цепочка рассуждений неверна (step_correctness < 0.4)
  answer_fail    — рассуждение есть, но итоговый ответ неправильный (answer_correctness < 0.33)
  ok             — final_score >= threshold

CLI:
  uv run python src/mhgb/analysis/error_analysis.py \\
    --results reports/main_v1/results.jsonl \\
    --tasks data/tasks_public_120.jsonl \\
    --output-dir reports/main_v1 \\
    --threshold 0.5
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal

from mhgb.logging_setup import logger

ErrorCategory = Literal[
    "hallucination", "multi_fail", "norm_miss", "reasoning_fail", "answer_fail", "ok"
]


def classify_error(record: dict, threshold: float = 0.5) -> ErrorCategory:
    """
    Классифицирует запись по доминирующему типу ошибки.
    Возвращает 'ok' если final_score >= threshold.
    """
    final    = float(record.get("final_score",              0.0) or 0.0)
    if final >= threshold:
        return "ok"

    norm_cov = float(record.get("norm_coverage_list",       0.0) or 0.0)
    step_cor = float(record.get("step_correctness",         0.0) or 0.0)
    ans_cor  = float(record.get("answer_correctness",       0.0) or 0.0)
    norm_rea = float(record.get("norm_coverage_reasoning",  0.0) or 0.0)

    # Галлюцинация: рассуждение содержит нормы, которых нет в списке ответа
    if norm_rea - norm_cov > 0.3:
        return "hallucination"

    # Полный провал всех компонентов
    if norm_cov < 0.3 and step_cor < 0.4 and ans_cor < 0.33:
        return "multi_fail"

    if norm_cov < 0.3:
        return "norm_miss"
    if step_cor < 0.4:
        return "reasoning_fail"
    return "answer_fail"


def analyze_errors(records: list[dict], threshold: float = 0.5) -> dict:
    """Агрегированная статистика ошибок по категориям и разрезам."""
    valid = [r for r in records if "error" not in r]
    categories = [classify_error(r, threshold) for r in valid]
    total = len(categories)

    category_counts = Counter(categories)
    by_model: dict[str, Counter] = defaultdict(Counter)
    by_type:  dict[str, Counter] = defaultdict(Counter)
    by_mode:  dict[str, Counter] = defaultdict(Counter)

    for r, cat in zip(valid, categories):
        by_model[r.get("model_name", "unknown")][cat] += 1
        by_type[r.get("task_type",  "unknown")][cat]  += 1
        by_mode[r.get("mode",       "unknown")][cat]   += 1

    return {
        "total":       total,
        "threshold":   threshold,
        "overall":     dict(category_counts),
        "overall_pct": {
            k: round(v / total, 3) if total else 0.0
            for k, v in category_counts.items()
        },
        "by_model":     {m: dict(c) for m, c in sorted(by_model.items())},
        "by_task_type": {t: dict(c) for t, c in sorted(by_type.items())},
        "by_mode":      {m: dict(c) for m, c in sorted(by_mode.items())},
    }


def main() -> None:
    import argparse

    from mhgb.analysis.aggregate_results import (
        enrich_with_task_meta,
        load_results_from_jsonl,
        load_tasks_index,
    )

    parser = argparse.ArgumentParser(description="Анализ ошибок моделей")
    parser.add_argument("--results",    required=True)
    parser.add_argument("--tasks",      default="data/tasks_public_120.jsonl")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--threshold",  type=float, default=0.5)
    args = parser.parse_args()

    records = load_results_from_jsonl(args.results)
    tasks_index = load_tasks_index(args.tasks) if Path(args.tasks).exists() else {}
    if tasks_index:
        records = enrich_with_task_meta(records, tasks_index)

    result = analyze_errors(records, threshold=args.threshold)

    output_dir = Path(args.output_dir or "reports/error_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "error_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    n_errors = result["total"] - result["overall"].get("ok", 0)
    logger.info(f"Ошибок (threshold={args.threshold}): {n_errors} / {result['total']}")
    for cat, n in sorted(result["overall"].items(), key=lambda x: -x[1]):
        pct = result["overall_pct"].get(cat, 0.0)
        logger.info(f"  {cat:20s}: {n:4d}  ({pct:.1%})")
    logger.info(f"Сохранено: {out_path}")


if __name__ == "__main__":
    main()
