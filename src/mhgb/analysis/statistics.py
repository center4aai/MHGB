"""
Statistical utilities for MHGB analysis (P2-0.5).

Public API:
    bootstrap_ci(scores, n_boot, alpha, seed)  -> BootstrapResult
    paired_gap_test(closed, open_, ...)        -> PairedTestResult
    wilson_ci(successes, n, alpha)             -> (lower, upper)
    compute_fleiss_kappa(annotations)          -> float
    sensitivity_analysis(model_scores, ...)    -> SensitivityResult
    compute_cross_judge_agreement(j1, j2)      -> CrossJudgeResult
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _z_from_alpha(alpha: float) -> float:
    """Normal quantile at (1 - alpha/2) — Abramowitz & Stegun 26.2.17."""
    lookup = {0.05: 1.959964, 0.10: 1.644854, 0.01: 2.575829, 0.001: 3.290527}
    if alpha in lookup:
        return lookup[alpha]
    p = 1.0 - alpha / 2.0
    if p >= 1.0:
        return 8.0
    t = math.sqrt(-2.0 * math.log(1.0 - p))
    c = (2.515517, 0.802853, 0.010328)
    d = (1.432788, 0.189269, 0.001308)
    return t - (c[0] + c[1] * t + c[2] * t**2) / (1.0 + d[0] * t + d[1] * t**2 + d[2] * t**3)


def _t_p_two_tailed(t: float, df: int) -> float:
    """Two-tailed p-value using normal approximation (accurate for df >= 30)."""
    if df <= 0 or t == 0.0:
        return 1.0
    return min(1.0, math.erfc(abs(t) / math.sqrt(2)))


def _cohens_kappa(labels1: list, labels2: list) -> float:
    """Cohen's κ for two raters on the same N items."""
    n = len(labels1)
    if n == 0:
        return 0.0
    cats = sorted(set(labels1) | set(labels2), key=str)
    k = len(cats)
    idx = {c: i for i, c in enumerate(cats)}
    mat = [[0] * k for _ in range(k)]
    for a, b in zip(labels1, labels2):
        mat[idx[a]][idx[b]] += 1
    p_o = sum(mat[i][i] for i in range(k)) / n
    p_e = sum(
        (sum(mat[i][j] for j in range(k)) / n)
        * (sum(mat[j][i] for j in range(k)) / n)
        for i in range(k)
    )
    return (p_o - p_e) / (1.0 - p_e) if p_e < 1.0 else 1.0


def _spearman(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation."""
    n = len(x)
    if n < 2:
        return 0.0

    def _ranks(vals: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: vals[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i + 1
            while j < n and vals[order[j]] == vals[order[i]]:
                j += 1
            avg = (i + j - 1) / 2.0 + 1.0
            for m in range(i, j):
                ranks[order[m]] = avg
            i = j
        return ranks

    rx, ry = _ranks(x), _ranks(y)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    den = math.sqrt(
        sum((rx[i] - mx) ** 2 for i in range(n))
        * sum((ry[i] - my) ** 2 for i in range(n))
    )
    return num / den if den > 0.0 else 0.0


def _kendall_tau_rankings(r1: list, r2: list) -> float:
    """Kendall's τ between two complete rankings of the same items."""
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


def _quantize(value: float, bins: int = 5) -> int:
    """Quantize [0, 1] float to integer bins for κ calculation."""
    return min(bins - 1, int(value * bins))


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

@dataclass
class BootstrapResult:
    mean: float
    lower: float
    upper: float
    n_boot: int


def bootstrap_ci(
    scores: Sequence[float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int | None = 42,
) -> BootstrapResult:
    """Percentile bootstrap CI for the mean of scores."""
    if not scores:
        raise ValueError("scores must not be empty")
    rng = random.Random(seed)
    s = list(scores)
    n = len(s)
    boot_means = sorted(
        sum(rng.choice(s) for _ in range(n)) / n
        for _ in range(n_boot)
    )
    lo = int(math.floor(alpha / 2.0 * n_boot))
    hi = min(int(math.ceil((1.0 - alpha / 2.0) * n_boot)) - 1, n_boot - 1)
    return BootstrapResult(
        mean=sum(s) / n,
        lower=boot_means[lo],
        upper=boot_means[hi],
        n_boot=n_boot,
    )


# ---------------------------------------------------------------------------
# Paired GAP test
# ---------------------------------------------------------------------------

@dataclass
class PairedTestResult:
    mean_gap: float
    gap_ci: tuple[float, float]   # bootstrap CI for mean GAP
    t_stat: float
    p_value: float


def paired_gap_test(
    closed_scores: Sequence[float],
    open_scores: Sequence[float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int | None = 42,
) -> PairedTestResult:
    """Paired t-test on GAP = open − closed + bootstrap CI for mean GAP."""
    c = list(closed_scores)
    o = list(open_scores)
    if len(c) != len(o):
        raise ValueError("closed_scores and open_scores must have the same length")
    n = len(c)
    if n < 2:
        raise ValueError("At least 2 pairs required")
    gaps = [oi - ci for ci, oi in zip(c, o)]
    mean_gap = sum(gaps) / n
    variance = sum((g - mean_gap) ** 2 for g in gaps) / (n - 1)
    se = math.sqrt(variance / n)
    if se > 0:
        t_stat = mean_gap / se
    elif mean_gap > 0:
        t_stat = float("inf")
    elif mean_gap < 0:
        t_stat = float("-inf")
    else:
        t_stat = 0.0
    p_value = _t_p_two_tailed(t_stat, df=n - 1)
    ci = bootstrap_ci(gaps, n_boot=n_boot, alpha=alpha, seed=seed)
    return PairedTestResult(
        mean_gap=mean_gap,
        gap_ci=(ci.lower, ci.upper),
        t_stat=t_stat,
        p_value=p_value,
    )


# ---------------------------------------------------------------------------
# Wilson CI for proportions
# ---------------------------------------------------------------------------

def wilson_ci(
    successes: int,
    n: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Wilson score interval for a proportion (successes / n)."""
    if n == 0:
        return (0.0, 1.0)
    z = _z_from_alpha(alpha)
    p_hat = successes / n
    z2n = z * z / n
    denom = 1.0 + z2n
    centre = (p_hat + z2n / 2.0) / denom
    margin = z * math.sqrt(p_hat * (1.0 - p_hat) / n + z2n / (4.0 * n)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


# ---------------------------------------------------------------------------
# Fleiss' κ
# ---------------------------------------------------------------------------

def compute_fleiss_kappa(annotations: list[list]) -> float:
    """
    Fleiss' κ for multi-rater agreement.

    annotations: N items × R raters — each row lists R labels assigned to that item.
    Returns κ ∈ [-1, 1]; 1 = perfect agreement, 0 = chance.
    """
    if not annotations:
        raise ValueError("annotations must not be empty")
    n_items = len(annotations)
    n_raters = len(annotations[0])
    if any(len(row) != n_raters for row in annotations):
        raise ValueError("All items must have the same number of raters")
    if n_raters < 2:
        raise ValueError("At least 2 raters required")

    cats = sorted({label for row in annotations for label in row}, key=str)
    k = len(cats)
    if k < 2:
        return 1.0
    cat_idx = {c: i for i, c in enumerate(cats)}

    counts = [[0] * k for _ in range(n_items)]
    for i, row in enumerate(annotations):
        for label in row:
            counts[i][cat_idx[label]] += 1

    total = n_items * n_raters
    p_j = [sum(counts[i][j] for i in range(n_items)) / total for j in range(k)]
    P_i = [
        (sum(counts[i][j] ** 2 for j in range(k)) - n_raters)
        / (n_raters * (n_raters - 1))
        for i in range(n_items)
    ]
    P_bar = sum(P_i) / n_items
    P_e = sum(p ** 2 for p in p_j)
    return (P_bar - P_e) / (1.0 - P_e) if P_e < 1.0 else 1.0


# ---------------------------------------------------------------------------
# Sensitivity analysis (D5.2)
# ---------------------------------------------------------------------------

@dataclass
class SensitivityResult:
    """Rank stability of model leaderboard across weight / tau configurations."""
    base_ranking: list[str]
    kendall_taus: list[float]
    min_tau: float
    mean_tau: float
    configs: list[dict]          # each: {alpha, beta, gamma, tau, ranking}


def sensitivity_analysis(
    model_scores: dict[str, list[tuple[float, float, float]]],
    weight_grid: list[tuple[float, float, float]] | None = None,
    tau_values: list[float] | None = None,
) -> SensitivityResult:
    """
    Recompute FS rankings under weight / tau variations; measure rank stability.

    model_scores: {model_name: [(NC, Step, Answer), ...]} per-task 3-tuples.
    weight_grid:  list of (alpha, beta, gamma) triples (default: equal + 3 imbalanced).
    tau_values:   quadrant thresholds to record (0.4, 0.5, 0.6); do not affect FS.
    """
    if not model_scores:
        return SensitivityResult(
            base_ranking=[], kendall_taus=[], min_tau=1.0, mean_tau=1.0, configs=[]
        )
    if weight_grid is None:
        weight_grid = [
            (1 / 3, 1 / 3, 1 / 3),
            (0.5, 0.25, 0.25),
            (0.25, 0.5, 0.25),
            (0.25, 0.25, 0.5),
        ]
    if tau_values is None:
        tau_values = [0.4, 0.5, 0.6]

    def _mean_fs(scores: list[tuple[float, float, float]], w: tuple) -> float:
        a, b, g = w
        return sum(a * nc + b * st + g * an for nc, st, an in scores) / len(scores)

    base_w = (1 / 3, 1 / 3, 1 / 3)
    base_ranking = sorted(model_scores, key=lambda m: _mean_fs(model_scores[m], base_w), reverse=True)

    taus: list[float] = []
    configs: list[dict] = []
    for w in weight_grid:
        w_fixed = w   # bind for lambda
        alt = sorted(model_scores, key=lambda m, ww=w_fixed: _mean_fs(model_scores[m], ww), reverse=True)
        for tau in tau_values:
            tau_k = _kendall_tau_rankings(base_ranking, alt)
            taus.append(tau_k)
            configs.append({
                "alpha": w[0], "beta": w[1], "gamma": w[2],
                "tau": tau, "ranking": list(alt),
            })

    return SensitivityResult(
        base_ranking=base_ranking,
        kendall_taus=taus,
        min_tau=min(taus) if taus else 1.0,
        mean_tau=sum(taus) / len(taus) if taus else 1.0,
        configs=configs,
    )


# ---------------------------------------------------------------------------
# Cross-judge agreement (D1.7)
# ---------------------------------------------------------------------------

@dataclass
class CrossJudgeResult:
    """Agreement between two judges on NC, Step, Answer scores + rank stability."""
    nc_kappa: float
    step_kappa: float
    answer_kappa: float
    kendall_tau: float    # rank stability by model mean FS
    spearman: float       # Spearman correlation of model-level mean FS


def compute_cross_judge_agreement(
    j1_records: list[dict],
    j2_records: list[dict],
) -> CrossJudgeResult:
    """
    Cross-judge agreement on per-task scores, matched by (task_id, model_name).

    Each record must have: task_id, model_name, nc, step, answer, final_score.
    """
    j2_idx = {(r["task_id"], r["model_name"]): r for r in j2_records}
    j1_nc: list[int] = []
    j2_nc: list[int] = []
    j1_step: list[int] = []
    j2_step: list[int] = []
    j1_ans: list[int] = []
    j2_ans: list[int] = []
    j1_mfs: dict[str, list[float]] = {}
    j2_mfs: dict[str, list[float]] = {}

    for r1 in j1_records:
        key = (r1["task_id"], r1["model_name"])
        r2 = j2_idx.get(key)
        if r2 is None:
            continue
        j1_nc.append(_quantize(r1["nc"]))
        j2_nc.append(_quantize(r2["nc"]))
        j1_step.append(_quantize(r1["step"]))
        j2_step.append(_quantize(r2["step"]))
        j1_ans.append(_quantize(r1["answer"]))
        j2_ans.append(_quantize(r2["answer"]))
        j1_mfs.setdefault(r1["model_name"], []).append(r1["final_score"])
        j2_mfs.setdefault(r1["model_name"], []).append(r2["final_score"])

    models = sorted(set(j1_mfs) & set(j2_mfs))
    fs1_dict = {m: sum(j1_mfs[m]) / len(j1_mfs[m]) for m in models}
    fs2_dict = {m: sum(j2_mfs[m]) / len(j2_mfs[m]) for m in models}
    fs1 = [fs1_dict[m] for m in models]
    fs2 = [fs2_dict[m] for m in models]

    r1_sorted = sorted(models, key=lambda m: fs1_dict[m], reverse=True)
    r2_sorted = sorted(models, key=lambda m: fs2_dict[m], reverse=True)

    return CrossJudgeResult(
        nc_kappa=_cohens_kappa(j1_nc, j2_nc) if j1_nc else 0.0,
        step_kappa=_cohens_kappa(j1_step, j2_step) if j1_step else 0.0,
        answer_kappa=_cohens_kappa(j1_ans, j2_ans) if j1_ans else 0.0,
        kendall_tau=_kendall_tau_rankings(r1_sorted, r2_sorted) if len(models) > 1 else 1.0,
        spearman=_spearman(fs1, fs2) if len(fs1) > 1 else 0.0,
    )
