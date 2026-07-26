"""
Генерация Markdown и LaTeX таблиц для статьи.

Читает CSV-файлы из reports/{experiment_name}/ (созданные
aggregate_results.py и compute_gap_analysis.py).

CLI:
  uv run python src/mhgb/analysis/generate_tables.py \\
    --experiment-name main_v1 \\
    --output-dir reports/main_v1/tables
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from mhgb.logging_setup import logger


# ---------------------------------------------------------------------------
# Форматирование
# ---------------------------------------------------------------------------

def fmt(val: Any, decimals: int = 3) -> str:
    if val is None or val == "":
        return "—"
    try:
        return f"{float(val):.{decimals}f}"
    except (ValueError, TypeError):
        return str(val)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(str(cell)))

    def _row(cells: list[str]) -> str:
        return "| " + " | ".join(str(c).ljust(col_widths[i]) for i, c in enumerate(cells)) + " |"

    sep = "|" + "|".join("-" * (w + 2) for w in col_widths) + "|"
    return "\n".join([_row(headers), sep] + [_row(r) for r in rows])


def latex_table(
    headers: list[str],
    rows: list[list[str]],
    caption: str = "",
    label: str = "",
) -> str:
    n = len(headers)
    col_spec = "l" + "r" * (n - 1)
    lines = [
        "\\begin{table}[ht]",
        "  \\centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        "    \\toprule",
        "    " + " & ".join(headers) + " \\\\",
        "    \\midrule",
    ]
    for row in rows:
        lines.append("    " + " & ".join(str(c) for c in row) + " \\\\")
    lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Загрузка CSV
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Таблица 1: главные результаты (model × mode → final_score)
# ---------------------------------------------------------------------------

def generate_main_results_table(agg_path: Path) -> tuple[str, str]:
    """
    Pivot: model_name × mode (closed / open_full_graph).
    Колонки: Model | Closed | Open | GAP | Norm (closed) | Step (closed) | Ans (closed)
    """
    rows_data = load_csv(agg_path)

    pivot: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rows_data:
        pivot[r.get("model_name", "")][r.get("mode", "")] = r

    headers = ["Model", "Closed FS", "Open FS", "GAP",
               "NormCov↓", "StepCorr↓", "AnsCorr↓"]
    table_rows: list[list[str]] = []

    for model in sorted(pivot):
        c = pivot[model].get("closed", {})
        o = pivot[model].get("open_full_graph", {})
        c_fs = float(c.get("final_score_mean") or 0)
        o_fs = float(o.get("final_score_mean") or 0)
        gap  = round(o_fs - c_fs, 3)
        row = [
            model,
            fmt(c.get("final_score_mean")),
            fmt(o.get("final_score_mean")),
            f"{gap:+.3f}",
            fmt(c.get("norm_coverage_list_mean")),
            fmt(c.get("step_correctness_mean")),
            fmt(c.get("answer_correctness_mean")),
        ]
        table_rows.append(row)

    md    = markdown_table(headers, table_rows)
    latex = latex_table(headers, table_rows,
                        caption="Main results: Final Score by model and mode",
                        label="tab:main_results")
    return md, latex


# ---------------------------------------------------------------------------
# Таблица 2: GAP по моделям с квадрантами
# ---------------------------------------------------------------------------

def generate_gap_by_model_table(gap_path: Path) -> tuple[str, str]:
    rows_data = load_csv(gap_path)

    headers = ["Model", "N", "GAP mean", "GAP std",
               "knows%", "reasons%", "halluc%", "incomp%"]
    table_rows: list[list[str]] = []

    for r in rows_data:
        counts = {q: int(r.get(f"q_{q}", 0) or 0)
                  for q in ("knows", "reasons", "hallucinates", "incompetent")}
        total = sum(counts.values()) or 1
        row = [
            r.get("model_name", ""),
            r.get("n_pairs", ""),
            fmt(r.get("final_score_gap_mean")),
            fmt(r.get("final_score_gap_std")),
            f"{counts['knows']/total:.0%}",
            f"{counts['reasons']/total:.0%}",
            f"{counts['hallucinates']/total:.0%}",
            f"{counts['incompetent']/total:.0%}",
        ]
        table_rows.append(row)

    md    = markdown_table(headers, table_rows)
    latex = latex_table(headers, table_rows,
                        caption="GAP analysis by model",
                        label="tab:gap_model")
    return md, latex


# ---------------------------------------------------------------------------
# Таблица 3: результаты по типу задачи
# ---------------------------------------------------------------------------

def generate_task_type_table(agg_path: Path) -> tuple[str, str]:
    rows_data = load_csv(agg_path)

    pivot: dict[tuple, dict] = {}
    for r in rows_data:
        key = (r.get("task_type", ""), r.get("mode", ""))
        pivot[key] = r

    task_types = sorted({k[0] for k in pivot})
    headers = ["Task type", "Closed FS", "Open FS", "GAP"]
    table_rows: list[list[str]] = []

    for tt in task_types:
        c = pivot.get((tt, "closed"), {})
        o = pivot.get((tt, "open_full_graph"), {})
        c_fs = float(c.get("final_score_mean") or 0)
        o_fs = float(o.get("final_score_mean") or 0)
        gap  = round(o_fs - c_fs, 3)
        table_rows.append([tt, fmt(c.get("final_score_mean")),
                            fmt(o.get("final_score_mean")), f"{gap:+.3f}"])

    md    = markdown_table(headers, table_rows)
    latex = latex_table(headers, table_rows,
                        caption="Results by task type",
                        label="tab:task_type")
    return md, latex


# ---------------------------------------------------------------------------
# Сохранение
# ---------------------------------------------------------------------------

def save_table(md: str, latex: str, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{name}.md").write_text(md, encoding="utf-8")
    (output_dir / f"{name}.tex").write_text(latex, encoding="utf-8")
    logger.info(f"Таблица: {output_dir / name}.(md|tex)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Генерация таблиц для статьи")
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--output-dir",      default=None)
    args = parser.parse_args()

    report_dir = Path(f"reports/{args.experiment_name}")
    output_dir = Path(args.output_dir or report_dir / "tables")

    # Таблица 1 — главные результаты
    p = report_dir / "agg_model_name_mode.csv"
    if p.exists():
        md, latex = generate_main_results_table(p)
        save_table(md, latex, output_dir, "main_results")
    else:
        logger.warning(f"Не найдено {p} — сначала запустите aggregate_results.py")

    # Таблица 2 — GAP по моделям
    p = report_dir / "gap_model_name.csv"
    if p.exists():
        md, latex = generate_gap_by_model_table(p)
        save_table(md, latex, output_dir, "gap_by_model")
    else:
        logger.warning(f"Не найдено {p} — сначала запустите compute_gap_analysis.py")

    # Таблица 3 — по типу задачи
    p = report_dir / "agg_task_type_hop_group.csv"
    if p.exists():
        md, latex = generate_task_type_table(p)
        save_table(md, latex, output_dir, "task_type_results")
    else:
        logger.warning(f"Не найдено {p}")

    logger.info("Готово")


if __name__ == "__main__":
    main()
