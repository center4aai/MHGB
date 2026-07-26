"""
Шаг 4.7 — RtA-анализ.

Читает results_rta.jsonl (один или все эксперименты) и вычисляет:
  - общую статистику отказов
  - разбивку по срезам (mode, task_type, hop_group, rta_topic, experiment)

CLI-запуск:
  uv run python -m mhgb.analysis.compute_rta_analysis \\
      --experiment yandexgpt_full600
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parents[3]
REPORTS_DIR = ROOT / "reports"


# ---------------------------------------------------------------------------
# Загрузка данных
# ---------------------------------------------------------------------------

def load_rta_data(exp_name: str, reports_dir: Path = REPORTS_DIR) -> list[dict]:
    """
    Загружает записи одного эксперимента для RtA-анализа.

    База — results.jsonl (ВСЕ ответы модели, дедуплицированный keep-last
    per task_id: retry-дубли error→ok разрешаются в пользу успешной версии).
    Каждая запись обогащается RtA-полями (is_rta/rta_type/rta_topic/
    rta_confidence) из results_rta.jsonl по task_id.

    Почему база = results.jsonl, а не results_rta.jsonl:
      - знаменатель RtA = все содержательные ответы (не покрытие детектора);
      - retry-дубли (error+ok) корректно разрешаются дедупликацией, а не
        помечаются error по факту наличия упавшей попытки;
      - канон-фильтр (is_answered / is_genuine_rta) нуждается в raw_response
        и tokens_output, которых нет в results_rta.jsonl.
    """
    results_path = reports_dir / exp_name / "results.jsonl"
    rta_path = reports_dir / exp_name / "results_rta.jsonl"
    if not rta_path.exists():
        raise FileNotFoundError(f"results_rta.jsonl не найден: {rta_path}")
    if not results_path.exists():
        raise FileNotFoundError(f"results.jsonl не найден: {results_path}")

    # RtA-индекс по task_id
    rta_idx: dict[str, dict] = {}
    with rta_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                rta_idx[r.get("task_id")] = r

    # База: results.jsonl, дедуп keep-last per task_id
    seen: dict[str, dict] = {}
    with results_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                seen[r.get("task_id")] = r

    records: list[dict] = []
    for tid, r in seen.items():
        ri = rta_idx.get(tid, {})
        # error-запись не может быть отказом (нет ответа)
        r["is_rta"] = bool(ri.get("is_rta")) and not r.get("error")
        r["rta_type"] = ri.get("rta_type")
        r["rta_topic"] = ri.get("rta_topic")
        r["rta_confidence"] = ri.get("rta_confidence")
        records.append(r)
    return records


def load_all_rta_data(reports_dir: Path = REPORTS_DIR) -> list[dict]:
    """Загружает results_rta.jsonl из всех экспериментов, где он есть.

    ⚠️ reports_dir по умолчанию = reports/ (MVP). Для Phase-2 передавай
    """
    all_records: list[dict] = []
    for exp_dir in sorted(reports_dir.iterdir()):
        rta_path = exp_dir / "results_rta.jsonl"
        if rta_path.exists():
            recs = []
            with rta_path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        recs.append(json.loads(line))
            # добавляем поле experiment_name если отсутствует
            for r in recs:
                r.setdefault("experiment_name", exp_dir.name)
            all_records.extend(recs)
    return all_records


# ---------------------------------------------------------------------------
# Вычисление статистики
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# КАНОН RtA-ставки — ЕДИНСТВЕННАЯ точка расчёта (UI + postprocess зовут ЭТО)
# ---------------------------------------------------------------------------
# Детектор RtA ложно помечает is_rta=True на тех-артефактах (пустой ответ,
# дегенеративная петля/reasoning-обрезка при cap, overflow). Настоящий отказ =
# is_rta И СОДЕРЖАТЕЛЬНЫЙ refusal-текст. Критерий — по ТЕКСТУ (raw_response),
# НЕ по topic: topic=None легитимен у generic-отказов слабых моделей
# (gigachat-2-max 50 таких, qwen3 36). Обоснование: DESIGN_LOG Урок 15.

def is_answered(record: dict) -> bool:
    """True если модель дала СОДЕРЖАТЕЛЬНЫЙ ответ (знаменатель RtA).

    Исключаются ТОЛЬКО состояния «нет ответа»: error/overflow и пустой
    raw_response. Дегенеративные петли и cap-обрезки ОСТАЮТСЯ (модель
    ответила плохо = failure mode, консистентно с SC/AC, где петли держатся
    в наборе с заслуженным низким баллом, а не исключаются как тех-сбой).
    """
    if record.get("error"):
        return False
    return bool((record.get("raw_response") or "").strip())


def is_genuine_rta(record: dict, max_tokens: int | None = None) -> bool:
    """True если запись — НАСТОЯЩИЙ отказ (числитель RtA), не тех-артефакт.

    is_rta И содержательный raw_response (не пустой, не обрезан по cap).
    Артефакты, ложно помеченные детектором, исключаются:
      - error/overflow: нет raw_response;
      - пустой raw_response (''): модель ничего не выдала;
      - cap (tokens_output==max_tokens): петля/reasoning-обрыв → вердикт
        «отказ» на неполном тексте ненадёжен (Урок 8). Проверено: реальные
        отказы короткие, до cap не доходят → «не cap» не выкидывает настоящих.
    """
    if not record.get("is_rta"):
        return False
    if not is_answered(record):
        return False
    if max_tokens is not None and record.get("tokens_output", 0) == max_tokens:
        return False
    return True


def compute_rta_rate(records: list[dict], max_tokens: int | None = None) -> dict:
    """КАНОН RtA-ставки: настоящие отказы / содержательные ответы.

    Единственная точка расчёта — вызывается из postprocess (compute_rta_summary)
    и UI (graph_explorer). НЕ дублировать формулу в другом месте.
    """
    n_answered = sum(1 for r in records if is_answered(r))
    n_rta = sum(1 for r in records if is_genuine_rta(r, max_tokens))
    return {
        "n_answered": n_answered,
        "n_rta": n_rta,
        "rta_rate": n_rta / n_answered if n_answered else 0.0,
    }


def compute_rta_summary(records: list[dict], max_tokens: int | None = None) -> dict:
    """
    Общая RtA-статистика по списку записей.

    RtA-ставка считается канон-функцией compute_rta_rate (настоящие отказы /
    содержательные ответы). by_type/by_topic — по настоящим отказам.

    Возвращает dict:
      n_total        — содержательных ответов (знаменатель, = n_answered)
      n_rta          — настоящих отказов (числитель)
      rta_rate       — доля отказов (0..1)
      by_type        — Counter refusal_type среди настоящих RtA
      by_topic       — Counter refusal_topic среди настоящих RtA
      by_mode        — Counter mode → (n_total, n_rta) по настоящим
      mean_confidence — среднее confidence среди настоящих RtA (None если 0)
    """
    rate = compute_rta_rate(records, max_tokens)
    n_total = rate["n_answered"]
    rta_records = [r for r in records if is_genuine_rta(r, max_tokens)]
    n_rta = rate["n_rta"]

    by_type: Counter = Counter(r.get("rta_type") for r in rta_records)
    by_topic: Counter = Counter(r.get("rta_topic") for r in rta_records)

    by_mode: dict[str, dict] = {}
    for r in records:
        if not is_answered(r):
            continue  # знаменатель — только содержательные ответы
        mode = r.get("mode", "unknown")
        if mode not in by_mode:
            by_mode[mode] = {"n_total": 0, "n_rta": 0}
        by_mode[mode]["n_total"] += 1
        if is_genuine_rta(r, max_tokens):
            by_mode[mode]["n_rta"] += 1

    confidences = [
        r["rta_confidence"] for r in rta_records
        if r.get("rta_confidence") is not None
    ]
    mean_conf = sum(confidences) / len(confidences) if confidences else None

    return {
        "n_total": n_total,
        "n_rta": n_rta,
        "rta_rate": n_rta / n_total if n_total else 0.0,
        "by_type": dict(by_type),
        "by_topic": dict(by_topic),
        "by_mode": by_mode,
        "mean_confidence": mean_conf,
    }


def compute_rta_by_slice(records: list[dict], by: str,
                         max_tokens: int | None = None) -> list[dict]:
    """
    Разбивка RtA-статистики по срезу (канон-фильтр is_genuine_rta / is_answered).

    by — имя поля: "mode", "task_type", "hop_group", "rta_topic",
                   "experiment_name", "model_name"

    Возвращает список dict (по одному на группу):
      {by: value, n_total, n_rta, rta_rate, top_type, top_topic}
    """
    groups: dict[str, list[dict]] = {}
    for r in records:
        key = str(r.get(by, "unknown"))
        groups.setdefault(key, []).append(r)

    result = []
    for key, recs in sorted(groups.items()):
        rta_recs = [r for r in recs if is_genuine_rta(r, max_tokens)]
        n_total = sum(1 for r in recs if is_answered(r))
        n_rta = len(rta_recs)
        top_type = Counter(r.get("rta_type") for r in rta_recs).most_common(1)
        top_topic = Counter(r.get("rta_topic") for r in rta_recs).most_common(1)
        result.append({
            by: key,
            "n_total": n_total,
            "n_rta": n_rta,
            "rta_rate": n_rta / n_total if n_total else 0.0,
            "top_type": top_type[0][0] if top_type else None,
            "top_topic": top_topic[0][0] if top_topic else None,
        })
    return result


def compute_rta_cross(records: list[dict],
                      by_x: str,
                      by_y: str,
                      max_tokens: int | None = None) -> dict[tuple[str, str], dict]:
    """
    Матрица RtA-ставок по двум срезам (например, model × task_type).
    Канон-фильтр is_genuine_rta / is_answered.

    Возвращает dict[(x_val, y_val)] → {n_total, n_rta, rta_rate}.
    """
    cells: dict[tuple[str, str], dict] = {}
    for r in records:
        if not is_answered(r):
            continue
        x = str(r.get(by_x, "unknown"))
        y = str(r.get(by_y, "unknown"))
        key = (x, y)
        if key not in cells:
            cells[key] = {"n_total": 0, "n_rta": 0}
        cells[key]["n_total"] += 1
        if is_genuine_rta(r, max_tokens):
            cells[key]["n_rta"] += 1
    for cell in cells.values():
        cell["rta_rate"] = (
            cell["n_rta"] / cell["n_total"] if cell["n_total"] else 0.0
        )
    return cells


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_summary(summary: dict, label: str = "") -> None:
    if label:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
    print(f"Всего записей : {summary['n_total']}")
    print(f"RtA=True      : {summary['n_rta']}  ({100*summary['rta_rate']:.1f}%)")
    if summary["mean_confidence"] is not None:
        print(f"Mean confidence (RtA): {summary['mean_confidence']:.3f}")
    print(f"\nПо типу отказа:")
    for k, v in sorted(summary["by_type"].items(), key=lambda x: -x[1]):
        print(f"  {k or 'null':30s}: {v}")
    print(f"\nПо теме отказа:")
    for k, v in sorted(summary["by_topic"].items(), key=lambda x: -x[1]):
        print(f"  {k or 'null':30s}: {v}")
    print(f"\nПо режиму:")
    for mode, d in sorted(summary["by_mode"].items()):
        rate = d["n_rta"] / d["n_total"] if d["n_total"] else 0
        print(f"  {mode:20s}: {d['n_rta']:3d}/{d['n_total']}  ({100*rate:.1f}%)")


def load_max_tokens_map(config_path: str = "configs/models.yaml") -> dict[str, int]:
    """model_name → max_tokens из yaml (для cap-детекции в канон-фильтре RtA)."""
    import yaml
    p = Path(config_path)
    if not p.exists():
        return {}
    cfg = yaml.safe_load(p.open(encoding="utf-8"))
    return {m["name"]: m.get("max_tokens") for m in cfg.get("models", [])
            if m.get("max_tokens") is not None}


def _max_tokens_for_records(records: list[dict], mt_map: dict[str, int]) -> int | None:
    """max_tokens единой модели в records (None если моделей несколько/неизвестна)."""
    names = {r.get("model_name") for r in records if r.get("model_name")}
    if len(names) == 1:
        return mt_map.get(next(iter(names)))
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="RtA-анализ.")
    parser.add_argument("--experiment", default=None,
                        help="Имя эксперимента (папка в reports/). Если не задан — все эксперименты.")
    parser.add_argument("--by", default=None,
                        help="Срез для разбивки: mode|task_type|hop_group|rta_topic|model_name")
    args = parser.parse_args()

    reports_dir = REPORTS_DIR

    if args.experiment:
        records = load_rta_data(args.experiment, reports_dir)
        label = args.experiment
    else:
        records = load_all_rta_data(reports_dir)
        label = "Все эксперименты"

    if not records:
        print("Нет данных (results_rta.jsonl пуст или не найден).")
        sys.exit(0)

    mt_map = load_max_tokens_map()
    max_tokens = _max_tokens_for_records(records, mt_map)
    if max_tokens is None and args.experiment:
        print("⚠️  max_tokens не определён — cap-фильтр отключён (RtA может быть завышен на cap-петлях)")

    summary = compute_rta_summary(records, max_tokens)
    _print_summary(summary, label)

    if args.by:
        print(f"\n{'─'*60}")
        print(f"Разбивка по: {args.by}")
        print(f"{'─'*60}")
        slices = compute_rta_by_slice(records, args.by, max_tokens)
        for s in slices:
            key_val = s[args.by]
            rate = s["rta_rate"]
            print(
                f"  {str(key_val):25s}  "
                f"RtA {s['n_rta']:3d}/{s['n_total']:3d} ({100*rate:5.1f}%)  "
                f"top_type={s['top_type'] or '-':25s}  "
                f"top_topic={s['top_topic'] or '-'}"
            )


if __name__ == "__main__":
    main()
