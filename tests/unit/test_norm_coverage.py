"""
Тесты для src/mhgb/eval/norm_coverage.py.
Проверяют:
  - compute_list_coverage: P/R/F1 по явному списку статей
  - compute_reasoning_coverage: P/R/F1 по упоминаниям в тексте
  - list_vs_reasoning_consistency: Jaccard-консистентность
"""

import pytest
from mhgb.eval.norm_coverage import (
    CoverageResult,
    compute_list_coverage,
    compute_reasoning_coverage,
    list_vs_reasoning_consistency,
)


def _corpus():
    return {
        "ТК_261": {"id": "ТК_261"},
        "ТК_256": {"id": "ТК_256"},
        "ТК_81":  {"id": "ТК_81"},
        "ТК_392": {"id": "ТК_392"},
        "ГК_15":  {"id": "ГК_15"},
    }


# ---------------------------------------------------------------------------
# compute_list_coverage
# ---------------------------------------------------------------------------

class TestComputeListCoverage:

    def test_perfect_match(self):
        r = compute_list_coverage({"ТК_261", "ТК_256"}, {"ТК_261", "ТК_256"})
        assert r["precision"] == pytest.approx(1.0)
        assert r["recall"] == pytest.approx(1.0)
        assert r["f1"] == pytest.approx(1.0)

    def test_full_recall_low_precision(self):
        # predicted contains all gold + one extra
        r = compute_list_coverage({"ТК_261", "ТК_256", "ТК_81"}, {"ТК_261", "ТК_256"})
        assert r["recall"] == pytest.approx(1.0)
        assert r["precision"] == pytest.approx(2 / 3)
        assert r["f1"] == pytest.approx(2 * (2/3) * 1.0 / (2/3 + 1.0))

    def test_full_precision_low_recall(self):
        # predicted is a subset of gold
        r = compute_list_coverage({"ТК_261"}, {"ТК_261", "ТК_256"})
        assert r["precision"] == pytest.approx(1.0)
        assert r["recall"] == pytest.approx(0.5)
        assert r["f1"] == pytest.approx(2 * 1.0 * 0.5 / (1.0 + 0.5))

    def test_no_overlap(self):
        r = compute_list_coverage({"ТК_81"}, {"ТК_261", "ТК_256"})
        assert r["precision"] == pytest.approx(0.0)
        assert r["recall"] == pytest.approx(0.0)
        assert r["f1"] == pytest.approx(0.0)

    def test_empty_predicted(self):
        r = compute_list_coverage(set(), {"ТК_261", "ТК_256"})
        assert r["precision"] == pytest.approx(0.0)
        assert r["recall"] == pytest.approx(0.0)
        assert r["f1"] == pytest.approx(0.0)

    def test_empty_gold_returns_zeros(self):
        r = compute_list_coverage({"ТК_261"}, set())
        assert r["precision"] == pytest.approx(0.0)
        assert r["recall"] == pytest.approx(0.0)
        assert r["f1"] == pytest.approx(0.0)

    def test_both_empty_returns_zeros(self):
        r = compute_list_coverage(set(), set())
        assert r == {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    def test_single_match(self):
        r = compute_list_coverage({"ТК_261"}, {"ТК_261"})
        assert r["f1"] == pytest.approx(1.0)

    def test_returns_typed_dict_keys(self):
        r = compute_list_coverage({"ТК_261"}, {"ТК_261"})
        assert set(r.keys()) == {"precision", "recall", "f1"}

    def test_values_in_0_1(self):
        r = compute_list_coverage({"ТК_261", "ТК_81"}, {"ТК_261", "ТК_256"})
        for v in r.values():
            assert 0.0 <= v <= 1.0

    def test_f1_is_harmonic_mean(self):
        r = compute_list_coverage({"ТК_261"}, {"ТК_261", "ТК_256"})
        expected_f1 = 2 * r["precision"] * r["recall"] / (r["precision"] + r["recall"])
        assert r["f1"] == pytest.approx(expected_f1)


# ---------------------------------------------------------------------------
# compute_reasoning_coverage
# ---------------------------------------------------------------------------

class TestComputeReasoningCoverage:

    def test_exact_mention_in_reasoning(self):
        text = "Применяется ст. 261 ТК РФ и ст. 256 ТК РФ."
        r = compute_reasoning_coverage(text, {"ТК_261", "ТК_256"}, _corpus())
        assert r["precision"] == pytest.approx(1.0)
        assert r["recall"] == pytest.approx(1.0)

    def test_partial_mention(self):
        text = "Применяется ст. 261 ТК РФ."
        r = compute_reasoning_coverage(text, {"ТК_261", "ТК_256"}, _corpus())
        assert r["recall"] == pytest.approx(0.5)
        assert r["precision"] == pytest.approx(1.0)

    def test_extra_hallucination_filtered_by_corpus(self):
        # ТК_999 is not in corpus, so precision stays perfect
        text = "ст. 261 ТК и ст. 999 ТК РФ"
        r = compute_reasoning_coverage(text, {"ТК_261"}, _corpus())
        assert r["precision"] == pytest.approx(1.0)
        assert r["recall"] == pytest.approx(1.0)

    def test_no_norms_in_text(self):
        r = compute_reasoning_coverage("Норм не упоминается.", {"ТК_261"}, _corpus())
        assert r["recall"] == pytest.approx(0.0)
        assert r["precision"] == pytest.approx(0.0)
        assert r["f1"] == pytest.approx(0.0)

    def test_empty_text(self):
        r = compute_reasoning_coverage("", {"ТК_261"}, _corpus())
        assert r == {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    def test_empty_corpus_returns_zeros(self):
        r = compute_reasoning_coverage("ст. 261 ТК РФ", {"ТК_261"}, {})
        assert r["recall"] == pytest.approx(0.0)

    def test_multi_law_in_reasoning(self):
        text = "ст. 261 ТК и ст. 15 ГК РФ применимы."
        r = compute_reasoning_coverage(text, {"ТК_261", "ГК_15"}, _corpus())
        assert r["f1"] == pytest.approx(1.0)

    def test_returns_typed_dict_keys(self):
        r = compute_reasoning_coverage("", {"ТК_261"}, _corpus())
        assert set(r.keys()) == {"precision", "recall", "f1"}


# ---------------------------------------------------------------------------
# list_vs_reasoning_consistency
# ---------------------------------------------------------------------------

class TestListVsReasoningConsistency:

    def test_identical_sets_returns_one(self):
        text = "ст. 261 ТК РФ и ст. 256 ТК"
        score = list_vs_reasoning_consistency({"ТК_261", "ТК_256"}, text, _corpus())
        assert score == pytest.approx(1.0)

    def test_no_overlap_returns_zero(self):
        text = "ст. 81 ТК РФ"
        score = list_vs_reasoning_consistency({"ТК_261"}, text, _corpus())
        assert score == pytest.approx(0.0)

    def test_partial_overlap(self):
        # list: {261, 256}, reasoning mentions: {261, 81}
        # intersection = {261}, union = {261, 256, 81}, jaccard = 1/3
        text = "ст. 261 ТК и ст. 81 ТК РФ"
        score = list_vs_reasoning_consistency({"ТК_261", "ТК_256"}, text, _corpus())
        assert score == pytest.approx(1 / 3)

    def test_both_empty_returns_one(self):
        score = list_vs_reasoning_consistency(set(), "текст без норм", _corpus())
        assert score == pytest.approx(1.0)

    def test_empty_list_reasoning_has_norms(self):
        # list empty, reasoning has ТК_261 → no intersection
        text = "ст. 261 ТК РФ"
        score = list_vs_reasoning_consistency(set(), text, _corpus())
        assert score == pytest.approx(0.0)

    def test_list_has_norms_reasoning_empty(self):
        score = list_vs_reasoning_consistency({"ТК_261"}, "нет норм", _corpus())
        assert score == pytest.approx(0.0)

    def test_returns_float(self):
        score = list_vs_reasoning_consistency({"ТК_261"}, "ст. 261 ТК", _corpus())
        assert isinstance(score, float)

    def test_value_in_0_1(self):
        text = "ст. 261 ТК и ст. 256 ТК"
        for list_set in [{"ТК_261"}, {"ТК_261", "ТК_256", "ТК_81"}, set()]:
            score = list_vs_reasoning_consistency(list_set, text, _corpus())
            assert 0.0 <= score <= 1.0
