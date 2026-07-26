"""
GAP-анализ: разрыв между closed-book и open-book результатами.

Загружает пары (closed, open) для каждой (model_name, task_id),
вычисляет GAP-метрики, агрегирует по разрезам.

CLI:
  uv run python src/mhgb/analysis/compute_gap_analysis.py \\
    --experiment-name main_v1 \\
    --tasks data/tasks_public_120.jsonl \\
    --results reports/main_v1/results.jsonl \\
    --output-dir reports/main_v1
"""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from mhgb.eval.gap_metric import classify_gap_quadrant, compute_gap
from mhgb.logging_setup import logger

_GAP_FIELDS = (
    "norm_coverage_gap",
    "step_correctness_gap",
    "answer_correctness_gap",
    "final_score_gap",
)

# norm_coverage_list → gap field name: norm_coverage (matches compute_all_gaps convention)
_METRIC_MAP: dict[str, str] = {
    "norm_coverage":    "norm_coverage_list",
    "step_correctness": "step_correctness",
    "answer_correctness": "answer_correctness",
    "final_score":      "final_score",
}


def pair_closed_open(records: list[dict]) -> list[dict]:
    """
    Сопоставляет closed и open_full_graph записи по (model_name, task_id).
    Возвращает список gap-записей с метаданными задачи.
    """
    by_key: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for r in records:
        if "error" in r:
            continue
        raw_id = r.get("task_id", "")
        base_id = raw_id.removesuffix("_closed").removesuffix("_open")
        key = (r.get("model_name", ""), base_id)
        mode = r.get("mode", "")
        by_key[key][mode] = r

    pairs: list[dict] = []
    for (model_name, task_id), modes in by_key.items():
        open_ = modes.get("open_full_graph") or modes.get("open_full_graph_edges")
        if "closed" not in modes or open_ is None:
            continue
        closed = modes["closed"]

        gap_record: dict[str, Any] = {
            "model_name":    model_name,
            "task_id":       task_id,
            "task_type":     closed.get("task_type")    or open_.get("task_type",    "unknown"),
            "hop_group":     closed.get("hop_group")    or open_.get("hop_group",    "unknown"),
            "branch_of_law": closed.get("branch_of_law") or open_.get("branch_of_law", "unknown"),
            "closed_final_score": closed.get("final_score"),
            "open_final_score":   open_.get("final_score"),
        }

        for gap_name, field in _METRIC_MAP.items():
            c = float(closed.get(field) or 0.0)
            o = float(open_.get(field)  or 0.0)
            gap_record[f"{gap_name}_gap"] = round(compute_gap(c, o), 4)

        gap_record["quadrant"] = classify_gap_quadrant(
            float(closed.get("final_score") or 0.0),
            float(open_.get("final_score")  or 0.0),
        )
        pairs.append(gap_record)
    return pairs


def summarize_gaps(gap_records: list[dict], by: list[str]) -> list[dict]:
    """Агрегирует gap_records по разрезам, возвращает список dict."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in gap_records:
        key = tuple(str(r.get(k, "")) for k in by)
        groups[key].append(r)

    rows: list[dict] = []
    for key, group in sorted(groups.items()):
        row: dict = dict(zip(by, key))
        row["n_pairs"] = len(group)

        for field in _GAP_FIELDS:
            values = [r[field] for r in group if field in r]
            if values:
                row[f"{field}_mean"] = round(statistics.mean(values), 4)
                row[f"{field}_std"]  = round(
                    statistics.stdev(values) if len(values) > 1 else 0.0, 4
                )

        quadrant_counts: dict[str, int] = defaultdict(int)
        for r in group:
            quadrant_counts[r.get("quadrant", "")] += 1
        for q in ("knows", "reasons", "hallucinates", "incompetent"):
            row[f"q_{q}"] = quadrant_counts.get(q, 0)
        rows.append(row)
    return rows


def _save_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Сохранено: {path} ({len(rows)} строк)")


def main() -> None:
    import argparse

    from mhgb.analysis.aggregate_results import (
        enrich_with_task_meta,
        load_evaluations_from_mongo,
        load_results_from_jsonl,
        load_tasks_index,
    )

    parser = argparse.ArgumentParser(description="GAP-анализ closed vs open_full_graph")
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--tasks",      default="data/tasks_public_120.jsonl")
    parser.add_argument("--results",    default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--use-mongo",  action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir or f"reports/{args.experiment_name}")

    if args.results:
        records = load_results_from_jsonl(args.results)
    elif args.use_mongo:
        from mhgb.storage.mongo_client import MongoStorage
        storage = MongoStorage()
        records = load_evaluations_from_mongo(storage, args.experiment_name)
    else:
        default_path = Path(f"reports/{args.experiment_name}/results.jsonl")
        if default_path.exists():
            logger.info(f"Авто-обнаружение результатов: {default_path}")
            records = load_results_from_jsonl(str(default_path))
        else:
            parser.error("Укажите --results <path> или --use-mongo")

    tasks_index = load_tasks_index(args.tasks) if Path(args.tasks).exists() else {}
    if tasks_index:
        records = enrich_with_task_meta(records, tasks_index)

    gap_records = pair_closed_open(records)
    logger.info(f"Пар (closed, open_full_graph): {len(gap_records)}")

    out_jsonl = output_dir / "gap_records.jsonl"
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in gap_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"Сохранено: {out_jsonl}")

    slice_configs = [
        ["model_name"],
        ["model_name", "task_type"],
        ["model_name", "hop_group"],
        ["task_type"],
        ["hop_group"],
    ]
    for slices in slice_configs:
        slug = "_".join(slices)
        rows = summarize_gaps(gap_records, by=slices)
        _save_csv(rows, output_dir / f"gap_{slug}.csv")

    logger.info(f"GAP-анализ завершён → {output_dir}")


if __name__ == "__main__":
    main()
