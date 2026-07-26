"""Post-hoc мердж SC/AC судьи (из results_crossjudge_<judge>.jsonl) в results.jsonl
+ пересчёт final_score.

Для моделей, прогнанных с --skip-judge (генерация + NC без судьи; судья отдельно
post-hoc, т.к. не влезает на карту вместе с моделью — напр. deepseek-r1-32b 62.5ГБ +
llama 49.6ГБ > 96ГБ .64). После мерджа results.jsonl ИДЕНТИЧЕН обычным прогонам
(NC + SC/AC + final_score), downstream-анализ (gap/aggregate/лидерборд) читает без правок.

Схема deepseek: run_main --skip-judge --skip-rta → run_cross_judge --judge llama →
ЭТОТ мердж → (run_cross_judge --judge qwen для κ) → run_rta_detection --judge llama.

CLI:
  uv run python scripts/merge_crossjudge_into_results.py \\
    --experiment deepseek-r1-32b-ollama_full600 --judge llama3.3-70b-judge
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mhgb.eval.final_score import compute_final_score


def main() -> None:
    ap = argparse.ArgumentParser(description="Мердж SC/AC судьи в results.jsonl (для --skip-judge прогонов)")
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--judge", default="llama3.3-70b-judge",
                    help="Судья, чьи оценки заливаем как ОСНОВНЫЕ (канон = llama)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base = ROOT / "reports"
    exp_dir = base / args.experiment
    results_path = exp_dir / "results.jsonl"
    judge_slug = args.judge.replace("/", "_").replace(":", "-")
    cj_path = exp_dir / f"results_crossjudge_{judge_slug}.jsonl"

    if not results_path.exists():
        raise SystemExit(f"Нет {results_path}")
    if not cj_path.exists():
        raise SystemExit(f"Нет {cj_path} (сначала run_cross_judge --judge {args.judge})")

    # SC/AC судьи по task_id
    cj: dict[str, dict] = {}
    with cj_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                cj[r["task_id"]] = r

    rows: list[dict] = []
    merged = missing = errskip = 0
    with results_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "error" in r:
                rows.append(r); errskip += 1; continue
            c = cj.get(r["task_id"])
            if c is None:
                rows.append(r); missing += 1; continue
            sc = c.get("sc_crossjudge")
            ac = c.get("ac_crossjudge")
            # Только поля, которые есть в обычном results.jsonl (step_scores/judge_logs
            # исключаются _writeable в run_main → НЕ добавляем, чтобы формат был идентичен).
            r["step_correctness"]  = sc
            r["answer_correctness"] = ac
            nc = r.get("norm_coverage_list") or 0.0
            r["final_score"] = compute_final_score(
                norm_coverage=nc,
                step_correctness=sc if sc is not None else 0.0,
                answer_correctness=ac if ac is not None else 0.0,
            )["final_score"]
            rows.append(r); merged += 1

    print(f"results: {len(rows)} | merged {merged} | missing-in-crossjudge {missing} | error-skip {errskip}")
    if args.dry_run:
        print("(dry-run, файл не тронут)")
        return
    shutil.copy(results_path, str(results_path) + ".bak_premerge")
    with results_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"✅ results.jsonl обновлён (бэкап .bak_premerge). Канон-судья: {args.judge}")


if __name__ == "__main__":
    main()
