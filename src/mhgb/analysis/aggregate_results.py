"""
Агрегация результатов эксперимента по разрезам.

Читает из MongoDB (evaluations) или JSONL-файла результатов,
обогащает метаданными задач (task_type, hop_group, branch_of_law),
агрегирует mean/std/count по метрикам.

CLI:
  uv run python src/mhgb/analysis/aggregate_results.py \\
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

from mhgb.logging_setup import logger

METRICS = (
    "norm_coverage_list",
    "norm_coverage_reasoning",
    "step_correctness",
    "answer_correctness",
    "final_score",
)


def load_task_ids_filter(path: str) -> set[str]:
    """Загружает base_task_ids из JSONL (стрипает суффиксы _closed/_open)."""
    ids: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                task = json.loads(line)
                task_id = task.get("id", "")
                base_id = task_id.removesuffix("_closed").removesuffix("_open")
                ids.add(base_id)
    return ids


def filter_by_task_ids(records: list[dict], task_ids: set[str]) -> list[dict]:
    """Фильтрует записи по base_task_id (стрипает суффиксы _closed/_open)."""
    filtered = []
    for r in records:
        task_id = r.get("task_id", "")
        base_id = task_id.removesuffix("_closed").removesuffix("_open")
        if base_id in task_ids:
            filtered.append(r)
    return filtered


def load_tasks_index(tasks_path: str) -> dict[str, dict]:
    """Загружает задачи из JSONL → dict task_id → task."""
    index: dict[str, dict] = {}
    with open(tasks_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                task = json.loads(line)
                index[task["id"]] = task
    return index


def load_evaluations_from_mongo(storage, experiment_name: str) -> list[dict]:
    """Загружает evaluations из MongoDB."""
    return list(
        storage.db["evaluations"].find({"experiment_name": experiment_name}, {"_id": 0})
    )


def load_results_from_jsonl(path: str) -> list[dict]:
    """Загружает результаты из JSONL-файла, пропуская записи с ошибками."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return [r for r in records if "error" not in r]


def enrich_with_task_meta(records: list[dict], tasks_index: dict[str, dict]) -> list[dict]:
    """Добавляет task_type, hop_group, branch_of_law из tasks_index (если ещё нет)."""
    enriched = []
    for r in records:
        task = tasks_index.get(r.get("task_id", ""), {})
        enriched.append({
            **r,
            "task_type":    r.get("task_type")    or task.get("type",          "unknown"),
            "hop_group":    r.get("hop_group")    or task.get("hop_group",     "unknown"),
            "branch_of_law": r.get("branch_of_law") or task.get("branch_of_law", "unknown"),
        })
    return enriched


def aggregate_by_slice(records: list[dict], by: list[str]) -> list[dict]:
    """
    Агрегирует метрики по произвольным разрезам.
    Возвращает список dict: ключи разреза + {metric}_mean / {metric}_std + count.
    """
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        key = tuple(str(r.get(k, "")) for k in by)
        groups[key].append(r)

    rows = []
    for key, group in sorted(groups.items()):
        row: dict = dict(zip(by, key))
        row["count"] = len(group)
        for metric in METRICS:
            values = [r[metric] for r in group if metric in r and r[metric] is not None]
            if values:
                row[f"{metric}_mean"] = round(statistics.mean(values), 4)
                row[f"{metric}_std"]  = round(
                    statistics.stdev(values) if len(values) > 1 else 0.0, 4
                )
            else:
                row[f"{metric}_mean"] = None
                row[f"{metric}_std"]  = None
        rows.append(row)
    return rows


def save_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        logger.warning(f"Нет данных для {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Сохранено: {path} ({len(rows)} строк)")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Агрегация результатов эксперимента")
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--tasks",      default="data/tasks_public_120.jsonl")
    parser.add_argument("--results",    default=None,
                        help="JSONL-файл результатов (вместо MongoDB)")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--use-mongo",  action="store_true")
    parser.add_argument("--task-ids-filter", default=None,
                        help="JSONL с задачами: фильтрует результаты по base_task_id")
    args = parser.parse_args()

    output_dir = Path(args.output_dir or f"reports/{args.experiment_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.results:
        logger.info(f"Загрузка из {args.results}")
        records = load_results_from_jsonl(args.results)
    elif args.use_mongo:
        from mhgb.storage.mongo_client import MongoStorage
        storage = MongoStorage()
        logger.info(f"Загрузка из MongoDB: {args.experiment_name}")
        records = load_evaluations_from_mongo(storage, args.experiment_name)
    else:
        default_path = Path(f"reports/{args.experiment_name}/results.jsonl")
        if default_path.exists():
            logger.info(f"Авто-обнаружение результатов: {default_path}")
            records = load_results_from_jsonl(str(default_path))
        else:
            parser.error("Укажите --results <path> или --use-mongo")

    logger.info(f"Загружено {len(records)} записей")

    if args.task_ids_filter:
        task_ids = load_task_ids_filter(args.task_ids_filter)
        before = len(records)
        records = filter_by_task_ids(records, task_ids)
        logger.info(f"Фильтр по task_ids: {before} → {len(records)} записей ({len(task_ids)} id в фильтре)")

    tasks_index = load_tasks_index(args.tasks) if Path(args.tasks).exists() else {}
    if tasks_index:
        records = enrich_with_task_meta(records, tasks_index)

    slice_configs = [
        ["model_name", "mode"],
        ["model_name", "task_type", "mode"],
        ["model_name", "hop_group", "mode"],
        ["model_name", "branch_of_law", "mode"],
        ["task_type", "hop_group"],
    ]
    for slices in slice_configs:
        slug = "_".join(slices)
        rows = aggregate_by_slice(records, by=slices)
        save_csv(rows, output_dir / f"agg_{slug}.csv")

    logger.info(f"Агрегация завершена → {output_dir}")


if __name__ == "__main__":
    main()
