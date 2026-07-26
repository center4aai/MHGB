"""
P2-4.5 — cross-judge κ по RtA (llama ↔ qwen) на пуле отказов.

Считает согласие двух судей по метке RtA. КАНОН (Урок 15): обе метки
прогоняются через is_genuine_rta (is_answered + не-cap) — иначе qwen даст
артефакты (петли/пустые/cap-обрезка как ложные отказы). κ — канон-функцией
_cohens_kappa из statistics.py (ручной κ запрещён, Урок 12).

Дизайн выборки (разрешение развилки, check 4):
  - κ считается ТОЛЬКО на 3 отказчивых (gigachat-lite/max, yandex): у них
    класс RtA ненулевой (79+79+32=190 реальных отказов) → κ на достаточной
    выборке, узкие CI.
  - 9 неотказчиков (RtA≈0): класс вырожден → κ неопределён, только
    descriptive-подтверждение (llama 0 → qwen подтверждает 0).

Матч по task_id (check 3): пары только где у ОБОИХ судей есть вердикт И
запись is_answered; непарные (overflow yandex 587) исключены из знаменателя,
НЕ считаются рассогласованием.


CLI:
  uv run python scripts/compute_rta_kappa.py \\
      --qwen-judge qwen3-27b-server
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mhgb.analysis.compute_rta_analysis import (
    is_answered,
    is_genuine_rta,
    load_max_tokens_map,
)
from mhgb.analysis.statistics import _cohens_kappa

# 3 отказчивых модели Phase-2 (ненулевой RtA) → валидный κ.
# (эксперимент, model_name-в-yaml для max_tokens cap-детекции)
REFUSERS = [
    ("gigachat_lite_full600", "gigachat-lite"),
    ("gigachat2max_full600", "gigachat-2-max"),
    ("yandexgpt_full600", "yandexgpt-5-pro"),
]

# 9 неотказчиков — только descriptive (κ вырожден).
NON_REFUSERS = [
    "tpro_full600", "gemma4_full600", "mistral_full600",
    "deepseek-r1-32b-ollama_full600", "gpt4o_full600", "claude_full600",
    "deepseek_api_full600", "o3_full600", "gemini_full600",
]


# ---------------------------------------------------------------------------
# Загрузка
# ---------------------------------------------------------------------------

def _load_base(exp_dir: Path) -> dict[str, dict]:
    """results.jsonl → task_id → запись (дедуп keep-last, как compute_rta_analysis)."""
    seen: dict[str, dict] = {}
    with (exp_dir / "results.jsonl").open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                seen[r.get("task_id")] = r
    return seen


def _load_judge(path: Path) -> dict[str, bool | None]:
    """results_rta_*.jsonl → task_id → сырой is_rta судьи (None если нет вердикта)."""
    idx: dict[str, bool | None] = {}
    if not path.exists():
        return idx
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                idx[r.get("task_id")] = r.get("is_rta")
    return idx


# ---------------------------------------------------------------------------
# κ по паре судей на одном эксперименте
# ---------------------------------------------------------------------------

def _canonical_label(base_rec: dict, judge_is_rta: bool | None,
                     max_tokens: int | None) -> int:
    """Метка судьи через КАНОН is_genuine_rta (Урок 15). 1=настоящий отказ."""
    merged = {**base_rec, "is_rta": bool(judge_is_rta)}
    return 1 if is_genuine_rta(merged, max_tokens) else 0


def compute_pairs(exp_dir: Path, mt_key: str, qwen_slug: str,
                  mt_map: dict[str, int]) -> dict:
    """Строит пары (llama, qwen) по task_id и считает κ + descriptive.

    Пул = is_answered И у ОБОИХ судей есть вердикт. Непарные исключены.
    """
    base = _load_base(exp_dir)
    llama = _load_judge(exp_dir / "results_rta.jsonl")
    qwen = _load_judge(exp_dir / f"results_rta_{qwen_slug}.jsonl")
    max_tokens = mt_map.get(mt_key)

    llama_labels: list[int] = []
    qwen_labels: list[int] = []
    n_no_verdict = 0
    n_not_answered = 0

    for tid, rec in base.items():
        if not is_answered(rec):
            n_not_answered += 1
            continue
        # check 3: пары только где у ОБОИХ есть вердикт
        if tid not in llama or tid not in qwen or llama[tid] is None or qwen[tid] is None:
            n_no_verdict += 1
            continue
        llama_labels.append(_canonical_label(rec, llama[tid], max_tokens))
        qwen_labels.append(_canonical_label(rec, qwen[tid], max_tokens))

    # 2×2 confusion (llama × qwen)
    both = sum(1 for a, b in zip(llama_labels, qwen_labels) if a == 1 and b == 1)
    llama_only = sum(1 for a, b in zip(llama_labels, qwen_labels) if a == 1 and b == 0)
    qwen_only = sum(1 for a, b in zip(llama_labels, qwen_labels) if a == 0 and b == 1)
    neither = sum(1 for a, b in zip(llama_labels, qwen_labels) if a == 0 and b == 0)

    n_llama_pos = both + llama_only
    n_qwen_pos = both + qwen_only
    kappa = _cohens_kappa(llama_labels, qwen_labels) if llama_labels else 0.0

    return {
        "experiment": exp_dir.name,
        "max_tokens": max_tokens,
        "n_pairs": len(llama_labels),
        "n_not_answered": n_not_answered,
        "n_no_verdict": n_no_verdict,
        "kappa": kappa,
        "confusion": {
            "both_rta": both, "llama_only": llama_only,
            "qwen_only": qwen_only, "neither": neither,
        },
        "n_llama_pos": n_llama_pos,
        "n_qwen_pos": n_qwen_pos,
        # descriptive: из N llama-отказов сколько qwen подтвердил
        "qwen_confirmed_of_llama": both / n_llama_pos if n_llama_pos else None,
        "_labels": (llama_labels, qwen_labels),  # для bootstrap/pooled
    }


def _bootstrap_kappa_ci(llama: list[int], qwen: list[int],
                        n_boot: int = 2000, alpha: float = 0.05,
                        seed: int = 42) -> tuple[float, float]:
    """Bootstrap CI для κ: ресэмпл пар с возвращением, пересчёт κ."""
    n = len(llama)
    if n < 2:
        return (0.0, 0.0)
    rng = random.Random(seed)
    kappas: list[float] = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        l = [llama[i] for i in idx]
        q = [qwen[i] for i in idx]
        kappas.append(_cohens_kappa(l, q))
    kappas.sort()
    lo = kappas[int((alpha / 2) * n_boot)]
    hi = kappas[int((1 - alpha / 2) * n_boot)]
    return (lo, hi)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Cross-judge κ по RtA (llama↔qwen).")
    ap.add_argument("--qwen-judge", default="qwen3-27b-server",
                    help="Имя qwen-судьи (слаг файла results_rta_<slug>.jsonl)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--json-out", default=None, help="Сохранить отчёт в JSON")
    args = ap.parse_args()

    reports_dir = ROOT / "reports"
    qwen_slug = args.qwen_judge.replace("/", "_").replace(":", "-")
    mt_map = load_max_tokens_map()

    print(f"\n{'='*70}")
    print(f"  P2-4.5 — Cross-judge κ (llama ↔ qwen) на пуле отказов")
    print(f"  Источник: {reports_dir}  |  qwen-файл: results_rta_{qwen_slug}.jsonl")
    print(f"{'='*70}")

    report: dict = {"refusers": [], "non_refusers": []}
    pooled_llama: list[int] = []
    pooled_qwen: list[int] = []

    print(f"\n── 3 ОТКАЗЧИВЫХ (валидный κ) ──\n")
    for exp, mt_key in REFUSERS:
        exp_dir = reports_dir / exp
        qwen_path = exp_dir / f"results_rta_{qwen_slug}.jsonl"
        if not qwen_path.exists():
            print(f"  ⚠️  {exp}: qwen-файл не найден ({qwen_path.name}) — пропуск")
            continue
        res = compute_pairs(exp_dir, mt_key, qwen_slug, mt_map)
        l, q = res.pop("_labels")
        pooled_llama.extend(l)
        pooled_qwen.extend(q)
        lo, hi = _bootstrap_kappa_ci(l, q, n_boot=args.n_boot)
        res["kappa_ci95"] = [lo, hi]
        report["refusers"].append(res)

        c = res["confusion"]
        conf = res["qwen_confirmed_of_llama"]
        print(f"  {exp}")
        print(f"    пар (оба вердикта, is_answered) : {res['n_pairs']}"
              f"  (непарных: {res['n_no_verdict']}, без ответа: {res['n_not_answered']})")
        print(f"    κ llama↔qwen                    : {res['kappa']:.3f}  "
              f"CI95 [{lo:.3f}, {hi:.3f}]")
        print(f"    confusion (llama×qwen)          : "
              f"оба={c['both_rta']}  llama-only={c['llama_only']}  "
              f"qwen-only={c['qwen_only']}  никто={c['neither']}")
        print(f"    descriptive                     : llama-отказов={res['n_llama_pos']}, "
              f"qwen подтвердил={c['both_rta']}"
              f"{f' ({100*conf:.0f}%)' if conf is not None else ''}, "
              f"qwen-доп={c['qwen_only']}")
        print()

    # Pooled κ по 3 отказчивым
    if pooled_llama:
        pooled_k = _cohens_kappa(pooled_llama, pooled_qwen)
        plo, phi = _bootstrap_kappa_ci(pooled_llama, pooled_qwen, n_boot=args.n_boot)
        both = sum(1 for a, b in zip(pooled_llama, pooled_qwen) if a == 1 and b == 1)
        n_lp = sum(pooled_llama)
        report["pooled"] = {
            "n_pairs": len(pooled_llama), "kappa": pooled_k,
            "kappa_ci95": [plo, phi], "n_llama_pos": n_lp, "both_rta": both,
        }
        print(f"  ▶ POOLED (3 отказчивых вместе)")
        print(f"    пар={len(pooled_llama)}  κ={pooled_k:.3f}  CI95 [{plo:.3f}, {phi:.3f}]")
        print(f"    llama-отказов={n_lp}, qwen подтвердил={both} "
              f"({100*both/n_lp:.0f}%)" if n_lp else "")

    # 9 неотказчиков — одна строка (κ вырожден)
    print(f"\n── 9 НЕОТКАЗЧИКОВ (κ вырожден, только descriptive) ──\n")
    for exp in NON_REFUSERS:
        exp_dir = reports_dir / exp
        qwen_path = exp_dir / f"results_rta_{qwen_slug}.jsonl"
        if not qwen_path.exists():
            report["non_refusers"].append({"experiment": exp, "qwen_present": False})
            continue
        res = compute_pairs(exp_dir, "", qwen_slug, mt_map)
        res.pop("_labels", None)
        c = res["confusion"]
        report["non_refusers"].append({
            "experiment": exp, "qwen_present": True,
            "n_llama_pos": res["n_llama_pos"], "n_qwen_pos": res["n_qwen_pos"],
            "both_rta": c["both_rta"],
        })
        print(f"  {exp:32s} llama-отказов={res['n_llama_pos']}, "
              f"qwen-отказов={res['n_qwen_pos']}, оба={c['both_rta']} "
              f"→ κ вырожден (класс≈0)")

    if not any(r.get("qwen_present") for r in report["non_refusers"]):
        print("  (qwen-файлов нет — неотказчиков не гоняли, ожидаемо: κ там вырожден)")

    if args.json_out:
        out = ROOT / args.json_out
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n💾 JSON → {out}")


if __name__ == "__main__":
    main()
