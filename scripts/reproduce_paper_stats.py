"""
Recompute the paper's headline statistics from the released per-model results.

Reads results/<experiment>/gap_records.jsonl — one record per fabula, holding the
closed-book and open-book Final Score and the GAP — and recomputes bootstrap
confidence intervals and paired t-tests. No LLM calls, no API keys, seconds to run.

    uv run python scripts/reproduce_paper_stats.py [--n-boot 2000] [--alpha 0.05] [--seed 42]

Outputs (results/statistics/):
    bootstrap_gap_ci.csv      GAP mean + 95% CI per model
    paired_test_results.csv   paired t-test on closed/open pairs per model
    closed_ci.csv             closed-book Final Score + 95% CI
    open_ci.csv               open-book Final Score + 95% CI

Sections that need raw per-response records (cross-judge kappa, norm-coverage CI,
weight sensitivity) are skipped automatically: those files are not part of the
public release.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

# Add src to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mhgb.analysis.statistics import bootstrap_ci, paired_gap_test

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RESULTS_DIR = ROOT / "results"
OUT_DIR = RESULTS_DIR / "statistics"

# Display order (closed desc → gemini first)
MODEL_ORDER = [
    ("gemini_full600",               "Gemini 3.1 Pro"),
    ("claude_full600",               "Claude Sonnet 4.6"),
    ("deepseek_api_full600",         "DeepSeek-R1 671B"),
    ("o3_full600",                   "o3"),
    ("gpt4o_full600",                "GPT-4o"),
    ("gigachat2max_full600",         "GigaChat-2 Max"),
    ("yandexgpt_full600",            "YandexGPT 5.1 Pro"),
    ("gemma4_full600",               "Gemma-4 26B"),
    ("gigachat_lite_full600",        "GigaChat-2 Lite"),
    ("mistral_full600",              "Mistral Small 3.2"),
    ("tpro_full600",                 "T-pro-it-2.1"),
    ("deepseek-r1-32b-ollama_full600", "DeepSeek-R1 32B"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_gap_records(exp_dir: Path) -> list[dict]:
    path = exp_dir / "gap_records.jsonl"
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_crossjudge(exp_dir: Path) -> list[dict]:
    """Load results_crossjudge_qwen.jsonl for κ CI."""
    for fname in ("results_crossjudge_qwen3-27b-server.jsonl",
                  "results_crossjudge_qwen.jsonl",
                  "results_crossjudge_qwen3-27b.jsonl"):
        p = exp_dir / fname
        if p.exists():
            records = []
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            return records
    return []


def _quantize(v: float, bins: int = 5) -> int:
    return min(bins - 1, int(v * bins))


def kappa_bootstrap_ci(
    llama_records: list[dict],
    qwen_records: list[dict],
    field: str,       # "step_correctness" or "answer_correctness" (from results.jsonl)
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float, int]:
    """
    Bootstrap CI for Cohen's κ (5-bin) between llama and qwen on `field`.
    llama_records: from results.jsonl (fields: step_correctness, answer_correctness)
    qwen_records: from results_crossjudge_qwen*.jsonl (fields: sc_crossjudge, ac_crossjudge)
    Returns (kappa_point, ci_lower, ci_upper, n_matched).
    """
    import random
    import math

    # Map main results field → crossjudge field name
    CROSSJUDGE_FIELD = {
        "step_correctness": "sc_crossjudge",
        "answer_correctness": "ac_crossjudge",
    }
    qwen_field = CROSSJUDGE_FIELD.get(field, field)

    qwen_idx = {r["task_id"]: r for r in qwen_records}

    pairs = []
    for r in llama_records:
        tid = r.get("task_id")
        r2 = qwen_idx.get(tid)
        if r2 is None:
            continue
        v1 = r.get(field)
        v2 = r2.get(qwen_field)
        if v1 is None or v2 is None:
            continue
        pairs.append((_quantize(float(v1)), _quantize(float(v2))))

    if len(pairs) < 2:
        return (float("nan"), float("nan"), float("nan"), 0)

    def _kappa(ps: list[tuple[int,int]]) -> float:
        n = len(ps)
        cats = list(range(5))
        counts = {(a, b): 0 for a in cats for b in cats}
        for a, b in ps:
            counts[(a, b)] += 1
        p_o = sum(counts[(i, i)] for i in cats) / n
        p_e = sum(
            (sum(counts[(i, j)] for j in cats) / n) *
            (sum(counts[(j, i)] for j in cats) / n)
            for i in cats
        )
        return (p_o - p_e) / (1.0 - p_e) if p_e < 1.0 else 1.0

    # Point estimate
    kappa_point = _kappa(pairs)

    # Bootstrap
    rng = random.Random(seed)
    n = len(pairs)
    boot_kappas = sorted(
        _kappa([rng.choice(pairs) for _ in range(n)])
        for _ in range(n_boot)
    )
    lo_idx = int(math.floor(alpha / 2.0 * n_boot))
    hi_idx = min(int(math.ceil((1.0 - alpha / 2.0) * n_boot)) - 1, n_boot - 1)
    return (kappa_point, boot_kappas[lo_idx], boot_kappas[hi_idx], n)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(n_boot: int = 2000, alpha: float = 0.05, seed: int = 42) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows_gap_ci = []
    rows_paired = []
    rows_closed_ci = []
    rows_open_ci = []
    rows_kappa_ci = []

    print(f"Bootstrap CI: n_boot={n_boot}, alpha={alpha}, seed={seed}\n")

    for exp_name, display_name in MODEL_ORDER:
        exp_dir = RESULTS_DIR / exp_name
        if not exp_dir.exists():
            print(f"  SKIP {exp_name} (нет директории)")
            continue

        records = load_gap_records(exp_dir)
        if not records:
            print(f"  SKIP {exp_name} (нет gap_records.jsonl)")
            continue

        n_pairs = len(records)
        closed = [r["closed_final_score"] for r in records]
        open_  = [r["open_final_score"]   for r in records]
        gaps   = [r["final_score_gap"]    for r in records]

        # 1) Bootstrap CI on GAP
        ci_gap = bootstrap_ci(gaps, n_boot=n_boot, alpha=alpha, seed=seed)
        rows_gap_ci.append({
            "model": display_name,
            "exp": exp_name,
            "n_pairs": n_pairs,
            "mean_gap": round(ci_gap.mean, 4),
            "ci_low": round(ci_gap.lower, 4),
            "ci_high": round(ci_gap.upper, 4),
        })

        # 2) Paired test closed ↔ open
        pt = paired_gap_test(closed, open_, n_boot=n_boot, alpha=alpha, seed=seed)
        sig = "***" if pt.p_value < 0.001 else ("**" if pt.p_value < 0.01 else ("*" if pt.p_value < 0.05 else "n.s."))
        rows_paired.append({
            "model": display_name,
            "n_pairs": n_pairs,
            "mean_gap": round(pt.mean_gap, 4),
            "t_stat": round(pt.t_stat, 3),
            "p_value": round(pt.p_value, 6),
            "sig": sig,
            "gap_ci_low": round(pt.gap_ci[0], 4),
            "gap_ci_high": round(pt.gap_ci[1], 4),
        })

        # 3) CI on closed and open scores
        ci_c = bootstrap_ci(closed, n_boot=n_boot, alpha=alpha, seed=seed)
        ci_o = bootstrap_ci(open_,  n_boot=n_boot, alpha=alpha, seed=seed)
        rows_closed_ci.append({
            "model": display_name,
            "mean_closed": round(ci_c.mean, 4),
            "ci_low": round(ci_c.lower, 4),
            "ci_high": round(ci_c.upper, 4),
        })
        rows_open_ci.append({
            "model": display_name,
            "mean_open": round(ci_o.mean, 4),
            "ci_low": round(ci_o.lower, 4),
            "ci_high": round(ci_o.upper, 4),
        })

        print(f"  {display_name:30s}  n={n_pairs}  GAP={ci_gap.mean:+.4f}  "
              f"95%CI[{ci_gap.lower:+.4f}, {ci_gap.upper:+.4f}]  p={pt.p_value:.4f}{sig}")

    print()

    # 4) κ CI from cross-judge files
    print("=== κ Bootstrap CI (llama↔qwen, 5-bin) ===\n")
    for exp_name, display_name in MODEL_ORDER:
        exp_dir = RESULTS_DIR / exp_name
        if not exp_dir.exists():
            continue

        # llama scores come from main results.jsonl
        results_path = exp_dir / "results.jsonl"
        if not results_path.exists():
            continue
        qwen_records = load_crossjudge(exp_dir)
        if not qwen_records:
            print(f"  SKIP {display_name} — нет crossjudge файла")
            continue

        llama_records = []
        with open(results_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("error"):
                    continue
                sc = r.get("step_correctness")
                ac = r.get("answer_correctness")
                if sc is None and ac is None:
                    continue
                llama_records.append(r)

        sc_k, sc_lo, sc_hi, n_sc = kappa_bootstrap_ci(
            llama_records, qwen_records, "step_correctness",
            n_boot=n_boot, alpha=alpha, seed=seed,
        )
        ac_k, ac_lo, ac_hi, n_ac = kappa_bootstrap_ci(
            llama_records, qwen_records, "answer_correctness",
            n_boot=n_boot, alpha=alpha, seed=seed,
        )
        rows_kappa_ci.append({
            "model": display_name,
            "n_sc": n_sc,
            "sc_kappa": round(sc_k, 4) if sc_k == sc_k else "nan",
            "sc_ci_low": round(sc_lo, 4) if sc_lo == sc_lo else "nan",
            "sc_ci_high": round(sc_hi, 4) if sc_hi == sc_hi else "nan",
            "n_ac": n_ac,
            "ac_kappa": round(ac_k, 4) if ac_k == ac_k else "nan",
            "ac_ci_low": round(ac_lo, 4) if ac_lo == ac_lo else "nan",
            "ac_ci_high": round(ac_hi, 4) if ac_hi == ac_hi else "nan",
        })
        print(f"  {display_name:30s}  "
              f"SC κ={sc_k:.4f} [{sc_lo:.4f},{sc_hi:.4f}] n={n_sc}  |  "
              f"AC κ={ac_k:.4f} [{ac_lo:.4f},{ac_hi:.4f}] n={n_ac}")

    # --- Save CSVs ---
    def _write(path: Path, rows: list[dict]) -> None:
        if not rows:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n  → {path}")

    print()
    _write(OUT_DIR / "bootstrap_gap_ci.csv", rows_gap_ci)
    _write(OUT_DIR / "paired_test_results.csv", rows_paired)
    _write(OUT_DIR / "closed_ci.csv", rows_closed_ci)
    _write(OUT_DIR / "open_ci.csv", rows_open_ci)
    _write(OUT_DIR / "kappa_ci.csv", rows_kappa_ci)

    # --- Print summary tables ---
    print("\n" + "=" * 80)
    print("TABLE 1: Bootstrap CI на GAP (95%, n_boot=2000, seed=42)")
    print("=" * 80)
    print(f"{'Модель':<32} {'N':>5} {'GAP':>7} {'CI low':>8} {'CI high':>8}")
    print("-" * 65)
    for r in rows_gap_ci:
        print(f"{r['model']:<32} {r['n_pairs']:>5} {r['mean_gap']:>+7.4f} {r['ci_low']:>+8.4f} {r['ci_high']:>+8.4f}")

    print("\n" + "=" * 80)
    print("TABLE 2: Paired t-test closed ↔ open (значим ли GAP?)")
    print("=" * 80)
    print(f"{'Модель':<32} {'N':>5} {'GAP':>7} {'t':>8} {'p':>10} {'sig':>5}")
    print("-" * 65)
    for r in rows_paired:
        print(f"{r['model']:<32} {r['n_pairs']:>5} {r['mean_gap']:>+7.4f} {r['t_stat']:>8.3f} {r['p_value']:>10.6f} {r['sig']:>5}")

    print("\n" + "=" * 80)
    print("TABLE 3: Bootstrap CI на closed_score — пересечение «Знающих» vs «Рассуждающих»")
    print("=" * 80)
    print(f"{'Модель':<32} {'closed':>7} {'CI low':>8} {'CI high':>8}")
    print("-" * 60)
    for r in rows_closed_ci:
        print(f"{r['model']:<32} {r['mean_closed']:>7.4f} {r['ci_low']:>+8.4f} {r['ci_high']:>+8.4f}")

    if rows_kappa_ci:
        print("\n" + "=" * 80)
        print("TABLE 4: Bootstrap CI на κ судей (llama↔qwen, 5-bin)")
        print("=" * 80)
        print(f"{'Модель':<32} {'κ SC':>7} {'SC low':>8} {'SC high':>8} {'κ AC':>7} {'AC low':>8} {'AC high':>8}")
        print("-" * 82)
        for r in rows_kappa_ci:
            print(f"{r['model']:<32} {r['sc_kappa']:>7} {r['sc_ci_low']:>8} {r['sc_ci_high']:>8} "
                  f"{r['ac_kappa']:>7} {r['ac_ci_low']:>8} {r['ac_ci_high']:>8}")


def compute_nc_and_sensitivity(n_boot: int = 2000, alpha: float = 0.05, seed: int = 42) -> None:
    """
    P2-5 пункты 3 + 4:
      3) Sensitivity analysis: rank stability across weight variations.
      4) Bootstrap CI on NC per model.
    """
    from mhgb.analysis.statistics import bootstrap_ci

    # Weight grid: равные + 3 варианта с усиленным компонентом
    WEIGHT_GRID = [
        (1/3, 1/3, 1/3),
        (0.5, 0.25, 0.25),   # NC-heavy
        (0.25, 0.5, 0.25),   # Step-heavy
        (0.25, 0.25, 0.5),   # Answer-heavy
    ]
    WEIGHT_NAMES = ["equal (1/3 each)", "NC-heavy (0.5/0.25/0.25)",
                    "Step-heavy (0.25/0.5/0.25)", "Answer-heavy (0.25/0.25/0.5)"]

    # Load per-task NC, Step, Answer for each model, split by mode
    closed_scores: dict[str, list[tuple[float,float,float]]] = {}
    open_scores:   dict[str, list[tuple[float,float,float]]] = {}

    nc_closed: dict[str, list[float]] = {}
    nc_open:   dict[str, list[float]] = {}

    for exp_name, display_name in MODEL_ORDER:
        exp_dir = RESULTS_DIR / exp_name
        results_path = exp_dir / "results.jsonl"
        if not results_path.exists():
            continue

        c_triples, o_triples = [], []
        c_nc, o_nc = [], []

        with open(results_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("error"):
                    continue
                nc_l = r.get("norm_coverage_list")
                nc_r = r.get("norm_coverage_reasoning")
                sc   = r.get("step_correctness")
                ac   = r.get("answer_correctness")
                if any(v is None for v in [nc_l, nc_r, sc, ac]):
                    continue
                nc = (float(nc_l) + float(nc_r)) / 2.0
                triple = (nc, float(sc), float(ac))
                mode = r.get("mode", "")
                if "closed" in mode:
                    c_triples.append(triple)
                    c_nc.append(nc)
                else:
                    o_triples.append(triple)
                    o_nc.append(nc)

        if c_triples:
            closed_scores[display_name] = c_triples
            nc_closed[display_name] = c_nc
        if o_triples:
            open_scores[display_name] = o_triples
            nc_open[display_name] = o_nc

    # -----------------------------------------------------------------------
    # 4) NC CI
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TABLE 5: Bootstrap CI на NC (Norm Coverage) — closed и open")
    print(f"         95%, n_boot={n_boot}, seed={seed}")
    print("=" * 80)
    print(f"{'Модель':<32} {'NC_cl':>7} {'cl_lo':>7} {'cl_hi':>7}  {'NC_op':>7} {'op_lo':>7} {'op_hi':>7}")
    print("-" * 78)

    nc_ci_rows = []
    for _, display_name in MODEL_ORDER:
        if display_name not in nc_closed:
            continue
        ci_c = bootstrap_ci(nc_closed[display_name], n_boot=n_boot, alpha=alpha, seed=seed)
        ci_o_data = nc_open.get(display_name, [])
        if ci_o_data:
            ci_o = bootstrap_ci(ci_o_data, n_boot=n_boot, alpha=alpha, seed=seed)
            print(f"{display_name:<32} {ci_c.mean:>7.4f} {ci_c.lower:>7.4f} {ci_c.upper:>7.4f}  "
                  f"{ci_o.mean:>7.4f} {ci_o.lower:>7.4f} {ci_o.upper:>7.4f}")
            nc_ci_rows.append({
                "model": display_name,
                "nc_closed": round(ci_c.mean, 4),
                "nc_closed_lo": round(ci_c.lower, 4),
                "nc_closed_hi": round(ci_c.upper, 4),
                "nc_open": round(ci_o.mean, 4),
                "nc_open_lo": round(ci_o.lower, 4),
                "nc_open_hi": round(ci_o.upper, 4),
            })
        else:
            print(f"{display_name:<32} {ci_c.mean:>7.4f} {ci_c.lower:>7.4f} {ci_c.upper:>7.4f}  {'—':>7}")

    if nc_ci_rows:
        p = OUT_DIR / "nc_ci.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(nc_ci_rows[0].keys()))
            w.writeheader()
            w.writerows(nc_ci_rows)
        print(f"\n  → {p}")

    # -----------------------------------------------------------------------
    # 3) Sensitivity analysis
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TABLE 6: Sensitivity Analysis — стабильность рейтинга при вариациях весов")
    print("         NC / Step / Answer → final_score; Kendall τ vs equal-weights baseline")
    print("=" * 80)

    def _mean_fs_local(triples, w):
        a, b, g = w
        return sum(a*nc + b*st + g*an for nc, st, an in triples) / len(triples)

    def _kendall_tau(r1, r2):
        """Local Kendall τ between two rankings."""
        n = len(r1)
        if n <= 1:
            return 1.0
        pos2 = {item: i for i, item in enumerate(r2)}
        concordant = discordant = 0
        for i in range(n):
            for j in range(i + 1, n):
                pi = pos2.get(r1[i], n)
                pj = pos2.get(r1[j], n)
                if pi < pj:
                    concordant += 1
                elif pi > pj:
                    discordant += 1
        total = n * (n - 1) // 2
        return (concordant - discordant) / total if total > 0 else 1.0

    sensitivity_rows = []

    for label, scores_dict, mode_label in [
        ("CLOSED-book scores", closed_scores, "closed"),
        ("OPEN-book scores",   open_scores,   "open"),
    ]:
        print(f"\n  [{label}]")
        base_w = (1/3, 1/3, 1/3)
        base_ranking = sorted(scores_dict,
                              key=lambda m: _mean_fs_local(scores_dict[m], base_w),
                              reverse=True)
        base_means = {m: round(_mean_fs_local(scores_dict[m], base_w), 4) for m in scores_dict}

        print(f"\n  Base ranking (equal weights):")
        prev_above = True
        for i, m in enumerate(base_ranking, 1):
            marker = ""
            curr_above = base_means[m] >= 0.5
            if prev_above and not curr_above:
                marker = "  ← τ=0.5 split here"
            prev_above = curr_above
            print(f"    {i:2}. {m:<32} FS={base_means[m]:.4f}{marker}")

        print(f"\n  Rank stability (Kendall τ vs equal-weights baseline):")
        for w, wname in zip(WEIGHT_GRID[1:], WEIGHT_NAMES[1:]):
            w_fixed = w
            alt = sorted(scores_dict,
                         key=lambda m: _mean_fs_local(scores_dict[m], w_fixed),
                         reverse=True)
            tau = _kendall_tau(base_ranking, alt)
            # Check profile split stability (for closed only)
            if mode_label == "closed":
                alt_means = {m: round(_mean_fs_local(scores_dict[m], w), 4) for m in scores_dict}
                base_knows = {m for m in scores_dict if base_means[m] >= 0.5}
                alt_knows  = {m for m in scores_dict if alt_means[m] >= 0.5}
                changed = (base_knows - alt_knows) | (alt_knows - base_knows)
                split_status = "✅ split стабилен" if not changed else f"⚠️  ИЗМЕНИЛСЯ: {changed}"
            else:
                split_status = ""
            print(f"    {wname:<35}  τ={tau:.4f}  {split_status}")
            sensitivity_rows.append({
                "mode": mode_label,
                "weights": wname,
                "alpha": w[0], "beta": w[1], "gamma": w[2],
                "kendall_tau": round(tau, 4),
            })

    if sensitivity_rows:
        p = OUT_DIR / "sensitivity.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w_writer = csv.DictWriter(f, fieldnames=list(sensitivity_rows[0].keys()))
            w_writer.writeheader()
            w_writer.writerows(sensitivity_rows)
        print(f"\n  → {p}")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all", action="store_true", help="Run all analyses incl. sensitivity + NC CI")
    args = parser.parse_args()
    main(n_boot=args.n_boot, alpha=args.alpha, seed=args.seed)

    # Norm-coverage CI and weight sensitivity need per-response records
    # (results.jsonl), which are not part of the public release: they would be
    # computed on an empty base ranking and yield meaningless numbers. The paper
    # reports them in Appendix B; they are reproducible from the full run outputs.
    if any((RESULTS_DIR / exp / "results.jsonl").exists() for exp, _ in MODEL_ORDER):
        compute_nc_and_sensitivity(n_boot=args.n_boot, alpha=args.alpha, seed=args.seed)
    else:
        print("\nSKIP: norm-coverage CI and weight sensitivity require per-response "
              "records (results.jsonl), which are not part of the public release.")
