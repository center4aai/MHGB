"""
Тесты для src/mhgb/eval/final_score.py.
"""

import pytest
from mhgb.eval.final_score import (
    DEFAULT_WEIGHTS,
    EvalWeights,
    FinalScoreResult,
    compute_final_score,
)


# ---------------------------------------------------------------------------
# EvalWeights
# ---------------------------------------------------------------------------

class TestEvalWeights:
    def test_default_weights_equal(self):
        w = EvalWeights()
        assert w.norm_coverage == pytest.approx(1 / 3)
        assert w.step_correctness == pytest.approx(1 / 3)
        assert w.answer_correctness == pytest.approx(1 / 3)

    def test_default_weights_sum_to_one(self):
        assert EvalWeights().is_valid()

    def test_custom_weights_valid(self):
        w = EvalWeights(norm_coverage=0.5, step_correctness=0.3, answer_correctness=0.2)
        assert w.is_valid()

    def test_custom_weights_not_summing_to_one(self):
        w = EvalWeights(norm_coverage=0.5, step_correctness=0.5, answer_correctness=0.5)
        assert not w.is_valid()

    def test_total(self):
        w = EvalWeights(0.2, 0.3, 0.5)
        assert w.total() == pytest.approx(1.0)

    def test_default_weights_singleton(self):
        assert DEFAULT_WEIGHTS.is_valid()


# ---------------------------------------------------------------------------
# compute_final_score — базовые сценарии
# ---------------------------------------------------------------------------

class TestComputeFinalScore:

    def test_all_perfect(self):
        r = compute_final_score(1.0, 1.0, 1.0)
        assert r["final_score"] == pytest.approx(1.0)

    def test_all_zero(self):
        r = compute_final_score(0.0, 0.0, 0.0)
        assert r["final_score"] == pytest.approx(0.0)

    def test_equal_weights_is_mean(self):
        r = compute_final_score(0.6, 0.9, 0.3)
        assert r["final_score"] == pytest.approx((0.6 + 0.9 + 0.3) / 3)

    def test_custom_weights_applied(self):
        w = EvalWeights(norm_coverage=0.5, step_correctness=0.3, answer_correctness=0.2)
        r = compute_final_score(1.0, 0.0, 0.0, weights=w)
        assert r["final_score"] == pytest.approx(0.5)

    def test_custom_weights_all_on_answer(self):
        w = EvalWeights(norm_coverage=0.0, step_correctness=0.0, answer_correctness=1.0)
        r = compute_final_score(0.0, 0.0, 0.75, weights=w)
        assert r["final_score"] == pytest.approx(0.75)

    def test_default_weights_used_when_none(self):
        r1 = compute_final_score(0.5, 0.7, 0.9)
        r2 = compute_final_score(0.5, 0.7, 0.9, weights=DEFAULT_WEIGHTS)
        assert r1["final_score"] == pytest.approx(r2["final_score"])


# ---------------------------------------------------------------------------
# Компоненты в результате
# ---------------------------------------------------------------------------

class TestFinalScoreResultComponents:

    def test_result_keys(self):
        r = compute_final_score(0.5, 0.5, 0.5)
        assert set(r.keys()) == {
            "norm_coverage", "step_correctness", "answer_correctness",
            "final_score", "weights",
        }

    def test_input_components_preserved(self):
        r = compute_final_score(0.3, 0.6, 0.9)
        assert r["norm_coverage"]      == pytest.approx(0.3)
        assert r["step_correctness"]   == pytest.approx(0.6)
        assert r["answer_correctness"] == pytest.approx(0.9)

    def test_weights_in_result(self):
        w = EvalWeights(0.5, 0.3, 0.2)
        r = compute_final_score(1.0, 1.0, 1.0, weights=w)
        assert r["weights"]["norm_coverage"]      == pytest.approx(0.5)
        assert r["weights"]["step_correctness"]   == pytest.approx(0.3)
        assert r["weights"]["answer_correctness"] == pytest.approx(0.2)

    def test_weights_keys_in_result(self):
        r = compute_final_score(0.5, 0.5, 0.5)
        assert set(r["weights"].keys()) == {
            "norm_coverage", "step_correctness", "answer_correctness"
        }

    def test_final_score_in_0_1_for_valid_weights(self):
        for nc, sc, ac in [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.4, 0.7, 0.2)]:
            r = compute_final_score(nc, sc, ac)
            assert 0.0 <= r["final_score"] <= 1.0


# ---------------------------------------------------------------------------
# Граничные значения
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_one_component_perfect_others_zero(self):
        w = EvalWeights(1.0, 0.0, 0.0)
        r = compute_final_score(0.8, 0.0, 0.0, weights=w)
        assert r["final_score"] == pytest.approx(0.8)

    def test_mixed_components(self):
        r = compute_final_score(0.0, 1.0, 0.5)
        expected = (0.0 + 1.0 + 0.5) / 3
        assert r["final_score"] == pytest.approx(expected)

    def test_returns_dict(self):
        r = compute_final_score(0.5, 0.5, 0.5)
        assert isinstance(r, dict)
