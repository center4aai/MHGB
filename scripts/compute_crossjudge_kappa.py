"""
Вычислить κ (Cohen's kappa) между основным судьёй (llama, results.jsonl)
и вторым судьёй (qwen, results_crossjudge_*.jsonl).

Использование:
    uv run python scripts/compute_crossjudge_kappa.py \\
        --experiment tpro_full600

    uv run python scripts/compute_crossjudge_kappa.py \\
        --experiment gpt4o_full600 \\
        --judge qwen3-27b-server
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mhgb.analysis.statistics import _cohens_kappa, _quantize  # noqa: WPS450


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _exact_pct(a: list, b: list) -> float:
    if not a:
        return 0.0
    return sum(x == y for x, y in zip(a, b)) / len(a)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--judge", default="qwen3-27b-server")
    args = parser.parse_args()

    root = ROOT / "reports"
    exp_dir = root / args.experiment
    main_path = exp_dir / "results.jsonl"
    cj_name = f"results_crossjudge_{args.judge}.jsonl"
    cj_path = exp_dir / cj_name

    if not main_path.exists():
        print(f"❌ Не найден: {main_path}")
        sys.exit(1)
    if not cj_path.exists():
        print(f"❌ Не найден: {cj_path}")
        sys.exit(1)

    # Загрузка
    main_records = _load_jsonl(main_path)
    cj_records = _load_jsonl(cj_path)

    # Индекс cross-judge по task_id
    cj_idx: dict[str, dict] = {r["task_id"]: r for r in cj_records}

    # Выравнивание
    j1_step: list[int] = []
    j2_step: list[int] = []
    j1_ans: list[int] = []
    j2_ans: list[int] = []
    n_matched = 0
    n_skipped = 0

    for r in main_records:
        if r.get("error"):
            n_skipped += 1
            continue
        tid = r.get("task_id")
        sc1 = r.get("step_correctness")
        ac1 = r.get("answer_correctness")
        if sc1 is None or ac1 is None:
            n_skipped += 1
            continue
        cj = cj_idx.get(tid)
        if cj is None:
            continue
        sc2 = cj.get("sc_crossjudge")
        ac2 = cj.get("ac_crossjudge")
        if sc2 is None or ac2 is None:
            continue
        j1_step.append(_quantize(sc1))
        j2_step.append(_quantize(sc2))
        j1_ans.append(_quantize(ac1))
        j2_ans.append(_quantize(ac2))
        n_matched += 1

    if not j1_step:
        print("❌ Нет пар для сравнения")
        sys.exit(1)

    sc_kappa = _cohens_kappa(j1_step, j2_step)
    ac_kappa = _cohens_kappa(j1_ans, j2_ans)
    sc_exact = _exact_pct(j1_step, j2_step)
    ac_exact = _exact_pct(j1_ans, j2_ans)

    # Дельта средних (чтобы понять, кто строже)
    sc1_mean = sum(r.get("step_correctness", 0) for r in main_records if not r.get("error") and r.get("step_correctness") is not None) / max(n_matched, 1)
    sc2_mean = sum(r.get("sc_crossjudge", 0) for r in cj_records) / max(len(cj_records), 1)
    ac1_mean = sum(r.get("answer_correctness", 0) for r in main_records if not r.get("error") and r.get("answer_correctness") is not None) / max(n_matched, 1)
    ac2_mean = sum(r.get("ac_crossjudge", 0) for r in cj_records) / max(len(cj_records), 1)

    judge_slug = args.judge
    print(f"\n=== Cross-judge κ: {args.experiment} ===")
    print(f"  Пар:        {n_matched}  (пропущено: {n_skipped})")
    print(f"  Судья 1:    llama3.3-70b-judge (основной)")
    print(f"  Судья 2:    {judge_slug}")
    print()
    print(f"  SC  κ = {sc_kappa:.3f}   (точн. {sc_exact:.1%})")
    print(f"  AC  κ = {ac_kappa:.3f}   (точн. {ac_exact:.1%})")
    print()
    print(f"  ΔSC (qwen−llama) = {sc2_mean - sc1_mean:+.3f}   "
          f"(llama {sc1_mean:.3f}, qwen {sc2_mean:.3f})")
    print(f"  ΔAC (qwen−llama) = {ac2_mean - ac1_mean:+.3f}   "
          f"(llama {ac1_mean:.3f}, qwen {ac2_mean:.3f})")
    print()
    print(f"  → SC κ {'✅ существенное (≥0.6)' if sc_kappa >= 0.6 else '⚠️ умеренное (<0.6)'}")
    print(f"  → AC κ {'✅ существенное (≥0.6)' if ac_kappa >= 0.6 else '⚠️ умеренное (<0.6)'}")


if __name__ == "__main__":
    main()
