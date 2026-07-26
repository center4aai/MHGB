"""Tests for src/mhgb/analysis/statistics.py (P2-0.5)."""
import math
import pytest

from mhgb.analysis.statistics import (
    BootstrapResult,
    CrossJudgeResult,
    PairedTestResult,
    SensitivityResult,
    bootstrap_ci,
    compute_cross_judge_agreement,
    compute_fleiss_kappa,
    paired_gap_test,
    sensitivity_analysis,
    wilson_ci,
)


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------

class TestBootstrapCI:
    def test_bounds_valid(self):
        result = bootstrap_ci([0.1, 0.3, 0.5, 0.7, 0.9], n_boot=500, seed=0)
        assert result.lower <= result.mean <= result.upper
        assert 0.0 <= result.lower
        assert result.upper <= 1.0

    def test_mean_correct(self):
        scores = [0.2, 0.4, 0.6, 0.8]
        result = bootstrap_ci(scores, n_boot=100, seed=1)
        assert result.mean == pytest.approx(0.5)

    def test_all_same_collapses_ci(self):
        result = bootstrap_ci([0.7] * 10, n_boot=200, seed=2)
        assert result.lower == pytest.approx(0.7, abs=1e-9)
        assert result.upper == pytest.approx(0.7, abs=1e-9)
        assert result.mean == pytest.approx(0.7)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            bootstrap_ci([])

    def test_seeded_reproducible(self):
        s = [0.1, 0.5, 0.9, 0.3, 0.2]
        r1 = bootstrap_ci(s, seed=7)
        r2 = bootstrap_ci(s, seed=7)
        assert r1.lower == r2.lower
        assert r1.upper == r2.upper

    def test_n_boot_stored(self):
        r = bootstrap_ci([0.5, 0.5], n_boot=333, seed=0)
        assert r.n_boot == 333

    def test_ci_width_decreases_with_sample_size(self):
        small = bootstrap_ci([0.5] * 5 + [0.0] * 5, n_boot=800, seed=3)
        large = bootstrap_ci([0.5] * 50 + [0.0] * 50, n_boot=800, seed=3)
        assert (small.upper - small.lower) >= (large.upper - large.lower) - 0.05


# ---------------------------------------------------------------------------
# paired_gap_test
# ---------------------------------------------------------------------------

class TestPairedGapTest:
    def test_large_gap_is_significant(self):
        closed = [0.2] * 30
        open_ = [0.8] * 30
        result = paired_gap_test(closed, open_)
        assert result.mean_gap == pytest.approx(0.6)
        assert result.p_value < 0.001
        assert result.t_stat > 0

    def test_zero_gap(self):
        scores = [0.5] * 20
        result = paired_gap_test(scores, scores)
        assert result.mean_gap == pytest.approx(0.0)
        assert result.t_stat == pytest.approx(0.0)
        # p-value close to 1 (perfectly zero difference)
        assert result.p_value > 0.9

    def test_ci_contains_mean_gap(self):
        closed = [0.3, 0.4, 0.5, 0.3, 0.4]
        open_ = [0.6, 0.7, 0.8, 0.6, 0.7]
        result = paired_gap_test(closed, open_)
        assert result.gap_ci[0] <= result.mean_gap <= result.gap_ci[1]

    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError):
            paired_gap_test([0.1, 0.2], [0.3])

    def test_too_few_pairs_raises(self):
        with pytest.raises(ValueError):
            paired_gap_test([0.5], [0.6])

    def test_negative_gap_detected(self):
        # open < closed (unusual but possible)
        result = paired_gap_test([0.9] * 20, [0.3] * 20)
        assert result.mean_gap == pytest.approx(-0.6)
        assert result.t_stat < 0
        assert result.p_value < 0.001


# ---------------------------------------------------------------------------
# wilson_ci
# ---------------------------------------------------------------------------

class TestWilsonCI:
    def test_n_zero(self):
        assert wilson_ci(0, 0) == (0.0, 1.0)

    def test_no_successes(self):
        lo, hi = wilson_ci(0, 100)
        assert lo == pytest.approx(0.0)
        assert 0.0 < hi < 0.05

    def test_all_successes(self):
        lo, hi = wilson_ci(100, 100)
        assert hi == pytest.approx(1.0)
        assert lo > 0.95

    def test_bounds_valid(self):
        lo, hi = wilson_ci(37, 100)
        assert 0.0 <= lo <= hi <= 1.0

    def test_symmetric_at_half(self):
        lo, hi = wilson_ci(500, 1000)
        assert abs(lo + hi - 1.0) < 0.001

    def test_tabular_value(self):
        # For successes=500, n=1000, alpha=0.05:
        # Wilson CI ≈ (0.469, 0.531)
        lo, hi = wilson_ci(500, 1000)
        assert abs(lo - 0.469) < 0.003
        assert abs(hi - 0.531) < 0.003

    def test_narrower_ci_with_larger_n(self):
        lo1, hi1 = wilson_ci(50, 100)
        lo2, hi2 = wilson_ci(500, 1000)
        assert (hi1 - lo1) > (hi2 - lo2)


# ---------------------------------------------------------------------------
# compute_fleiss_kappa
# ---------------------------------------------------------------------------

class TestFleissKappa:
    def test_perfect_agreement(self):
        annotations = [[0, 0, 0], [1, 1, 1], [0, 0, 0], [1, 1, 1]]
        assert compute_fleiss_kappa(annotations) == pytest.approx(1.0)

    def test_single_category_returns_one(self):
        # k < 2 → all agree → κ = 1
        annotations = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        assert compute_fleiss_kappa(annotations) == pytest.approx(1.0)

    def test_chance_agreement_near_zero(self):
        # Equal split, 2 items fully agree + 2 fully disagree → κ ≈ 0
        annotations = [[0, 0], [1, 1], [0, 1], [1, 0]]
        kappa = compute_fleiss_kappa(annotations)
        assert abs(kappa) < 0.05

    def test_output_range(self):
        annotations = [[0, 0, 1], [1, 1, 0], [0, 1, 0], [1, 0, 1]]
        assert -1.0 <= compute_fleiss_kappa(annotations) <= 1.0

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            compute_fleiss_kappa([])

    def test_mismatched_raters_raises(self):
        with pytest.raises(ValueError):
            compute_fleiss_kappa([[0, 1], [0]])

    def test_string_labels_perfect(self):
        annotations = [["yes", "yes"], ["no", "no"], ["yes", "yes"]]
        assert compute_fleiss_kappa(annotations) == pytest.approx(1.0)

    def test_three_raters_partial(self):
        # 2 items full agreement, 2 items 2/3 agreement → κ ≈ 1/3
        annotations = [[0, 0, 0], [1, 1, 0], [0, 0, 1], [1, 1, 1]]
        kappa = compute_fleiss_kappa(annotations)
        assert kappa == pytest.approx(1 / 3, abs=0.01)


# ---------------------------------------------------------------------------
# sensitivity_analysis
# ---------------------------------------------------------------------------

class TestSensitivityAnalysis:
    def test_empty_model_scores(self):
        result = sensitivity_analysis({})
        assert result.base_ranking == []
        assert result.kendall_taus == []
        assert result.min_tau == 1.0

    def test_single_model(self):
        result = sensitivity_analysis({"A": [(0.5, 0.5, 0.5)] * 10})
        assert result.base_ranking == ["A"]
        assert all(t == 1.0 for t in result.kendall_taus)

    def test_stable_ranking(self):
        # A always dominates on all metrics → rank never changes
        scores = {"A": [(0.9, 0.9, 0.9)] * 10, "B": [(0.1, 0.1, 0.1)] * 10}
        result = sensitivity_analysis(scores)
        assert result.base_ranking[0] == "A"
        assert all(t == pytest.approx(1.0) for t in result.kendall_taus)

    def test_unstable_ranking_flips(self):
        # A strong on NC, B strong on Step → ranking flips with weights
        scores = {
            "A": [(0.9, 0.1, 0.5)] * 10,
            "B": [(0.1, 0.9, 0.5)] * 10,
        }
        result = sensitivity_analysis(
            scores,
            weight_grid=[(1 / 3, 1 / 3, 1 / 3), (0.9, 0.05, 0.05), (0.05, 0.9, 0.05)],
            tau_values=[0.5],
        )
        assert result.min_tau < 1.0

    def test_configs_count(self):
        scores = {"A": [(0.5, 0.5, 0.5)], "B": [(0.3, 0.7, 0.4)]}
        wg = [(1 / 3, 1 / 3, 1 / 3), (0.5, 0.25, 0.25)]
        tv = [0.4, 0.5, 0.6]
        result = sensitivity_analysis(scores, weight_grid=wg, tau_values=tv)
        assert len(result.configs) == len(wg) * len(tv)

    def test_result_fields(self):
        result = sensitivity_analysis({"A": [(0.5, 0.5, 0.5)]})
        assert hasattr(result, "base_ranking")
        assert hasattr(result, "kendall_taus")
        assert hasattr(result, "min_tau")
        assert hasattr(result, "mean_tau")
        assert hasattr(result, "configs")

    def test_mean_tau_in_range(self):
        scores = {"A": [(0.8, 0.2, 0.5)] * 5, "B": [(0.2, 0.8, 0.5)] * 5, "C": [(0.5, 0.5, 0.5)] * 5}
        result = sensitivity_analysis(scores)
        assert -1.0 <= result.mean_tau <= 1.0
        assert -1.0 <= result.min_tau <= 1.0


# ---------------------------------------------------------------------------
# compute_cross_judge_agreement
# ---------------------------------------------------------------------------

class TestCrossJudgeAgreement:
    @staticmethod
    def _make_records(task_ids: list[str], model: str, nc: float, step: float, answer: float) -> list[dict]:
        return [
            {
                "task_id": tid,
                "model_name": model,
                "nc": nc,
                "step": step,
                "answer": answer,
                "final_score": (nc + step + answer) / 3.0,
            }
            for tid in task_ids
        ]

    def test_identical_judges(self):
        tasks = [f"t{i}" for i in range(20)]
        recs = self._make_records(tasks, "model_A", 0.8, 0.7, 0.9)
        result = compute_cross_judge_agreement(recs, recs)
        assert result.nc_kappa == pytest.approx(1.0)
        assert result.step_kappa == pytest.approx(1.0)
        assert result.answer_kappa == pytest.approx(1.0)

    def test_no_overlap_returns_zero_kappa(self):
        r1 = self._make_records(["t1"], "A", 0.5, 0.5, 0.5)
        r2 = self._make_records(["t2"], "A", 0.5, 0.5, 0.5)
        result = compute_cross_judge_agreement(r1, r2)
        assert result.nc_kappa == pytest.approx(0.0)

    def test_two_models_same_rank(self):
        tasks = [f"t{i}" for i in range(10)]
        r1 = (
            self._make_records(tasks, "A", 0.9, 0.9, 0.9)
            + self._make_records(tasks, "B", 0.3, 0.3, 0.3)
        )
        r2 = (
            self._make_records(tasks, "A", 0.8, 0.8, 0.8)
            + self._make_records(tasks, "B", 0.2, 0.2, 0.2)
        )
        result = compute_cross_judge_agreement(r1, r2)
        assert result.kendall_tau == pytest.approx(1.0)
        assert result.spearman == pytest.approx(1.0)

    def test_reversed_rank(self):
        tasks = [f"t{i}" for i in range(10)]
        r1 = (
            self._make_records(tasks, "A", 0.9, 0.9, 0.9)
            + self._make_records(tasks, "B", 0.3, 0.3, 0.3)
        )
        # J2 swaps A and B scores
        r2 = (
            self._make_records(tasks, "A", 0.3, 0.3, 0.3)
            + self._make_records(tasks, "B", 0.9, 0.9, 0.9)
        )
        result = compute_cross_judge_agreement(r1, r2)
        assert result.kendall_tau == pytest.approx(-1.0)
        assert result.spearman == pytest.approx(-1.0)

    def test_result_has_all_fields(self):
        tasks = ["t1", "t2"]
        r = self._make_records(tasks, "M", 0.5, 0.5, 0.5)
        result = compute_cross_judge_agreement(r, r)
        assert hasattr(result, "nc_kappa")
        assert hasattr(result, "step_kappa")
        assert hasattr(result, "answer_kappa")
        assert hasattr(result, "kendall_tau")
        assert hasattr(result, "spearman")

    def test_single_model_tau_is_one(self):
        tasks = [f"t{i}" for i in range(5)]
        r = self._make_records(tasks, "OnlyModel", 0.7, 0.6, 0.8)
        result = compute_cross_judge_agreement(r, r)
        assert result.kendall_tau == pytest.approx(1.0)
