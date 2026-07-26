"""Тесты для src/mhgb/eval/gap_metric.py"""

import pytest
from mhgb.eval.gap_metric import (
    GAP_FIELDS,
    GapResult,
    aggregate_gaps_by_slice,
    classify_gap_quadrant,
    compute_all_gaps,
    compute_gap,
)


# ---------------------------------------------------------------------------
# compute_gap
# ---------------------------------------------------------------------------

class TestComputeGap:
    def test_positive_gap(self):
        assert compute_gap(0.3, 0.7) == pytest.approx(0.4)

    def test_negative_gap(self):
        assert compute_gap(0.8, 0.5) == pytest.approx(-0.3)

    def test_zero_gap(self):
        assert compute_gap(0.5, 0.5) == pytest.approx(0.0)

    def test_perfect_open(self):
        assert compute_gap(0.0, 1.0) == pytest.approx(1.0)

    def test_perfect_closed(self):
        assert compute_gap(1.0, 0.0) == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# classify_gap_quadrant
# ---------------------------------------------------------------------------

class TestClassifyGapQuadrant:
    def test_knows_both_high(self):
        assert classify_gap_quadrant(0.8, 0.9) == "knows"

    def test_knows_both_at_threshold(self):
        assert classify_gap_quadrant(0.5, 0.5) == "knows"

    def test_reasons_open_high_closed_low(self):
        assert classify_gap_quadrant(0.2, 0.8) == "reasons"

    def test_hallucinates_closed_high_open_low(self):
        assert classify_gap_quadrant(0.9, 0.1) == "hallucinates"

    def test_incompetent_both_low(self):
        assert classify_gap_quadrant(0.1, 0.2) == "incompetent"

    def test_both_zero(self):
        assert classify_gap_quadrant(0.0, 0.0) == "incompetent"

    def test_both_one(self):
        assert classify_gap_quadrant(1.0, 1.0) == "knows"

    def test_custom_threshold_high(self):
        # с threshold=0.7: 0.6 считается низким
        assert classify_gap_quadrant(0.6, 0.6, threshold=0.7) == "incompetent"

    def test_custom_threshold_low(self):
        # с threshold=0.3: 0.4 уже высокое
        assert classify_gap_quadrant(0.4, 0.4, threshold=0.3) == "knows"

    def test_boundary_just_below_threshold(self):
        assert classify_gap_quadrant(0.49, 0.49) == "incompetent"

    def test_boundary_just_at_threshold(self):
        assert classify_gap_quadrant(0.5, 0.49) == "hallucinates"


# ---------------------------------------------------------------------------
# compute_all_gaps
# ---------------------------------------------------------------------------

def _make_result(nc=0.5, sc=0.5, ac=0.5, fs=0.5):
    return {
        "norm_coverage": nc,
        "step_correctness": sc,
        "answer_correctness": ac,
        "final_score": fs,
    }


class TestComputeAllGaps:
    def test_all_fields_present(self):
        result = compute_all_gaps(_make_result(), _make_result())
        for field in GAP_FIELDS:
            assert field in result
        assert "quadrant" in result

    def test_gap_values_correct(self):
        closed = _make_result(nc=0.3, sc=0.4, ac=0.5, fs=0.4)
        open_ = _make_result(nc=0.7, sc=0.6, ac=0.8, fs=0.7)
        result = compute_all_gaps(closed, open_)
        assert result["norm_coverage_gap"] == pytest.approx(0.4)
        assert result["step_correctness_gap"] == pytest.approx(0.2)
        assert result["answer_correctness_gap"] == pytest.approx(0.3)
        assert result["final_score_gap"] == pytest.approx(0.3)

    def test_quadrant_from_final_score(self):
        closed = _make_result(fs=0.2)
        open_ = _make_result(fs=0.8)
        result = compute_all_gaps(closed, open_)
        assert result["quadrant"] == "reasons"

    def test_custom_threshold_passed(self):
        closed = _make_result(fs=0.6)
        open_ = _make_result(fs=0.6)
        result = compute_all_gaps(closed, open_, threshold=0.7)
        assert result["quadrant"] == "incompetent"

    def test_negative_gap_hallucinates(self):
        closed = _make_result(fs=0.9)
        open_ = _make_result(fs=0.2)
        result = compute_all_gaps(closed, open_)
        assert result["quadrant"] == "hallucinates"
        assert result["final_score_gap"] < 0


# ---------------------------------------------------------------------------
# aggregate_gaps_by_slice
# ---------------------------------------------------------------------------

def _make_record(model="m1", task_type="rule_selection", hop_group="shallow",
                 branch="трудовое", nc_gap=0.1, sc_gap=0.1, ac_gap=0.1,
                 fs_gap=0.1, quadrant="reasons"):
    return {
        "model": model,
        "task_type": task_type,
        "hop_group": hop_group,
        "branch_of_law": branch,
        "norm_coverage_gap": nc_gap,
        "step_correctness_gap": sc_gap,
        "answer_correctness_gap": ac_gap,
        "final_score_gap": fs_gap,
        "quadrant": quadrant,
    }


class TestAggregateGapsBySlice:
    def test_single_record(self):
        records = [_make_record(fs_gap=0.2)]
        result = aggregate_gaps_by_slice(records, by=["model"])
        assert ("m1",) in result
        assert result[("m1",)]["count"] == 1
        assert result[("m1",)]["final_score_gap_mean"] == pytest.approx(0.2)
        assert result[("m1",)]["final_score_gap_std"] == pytest.approx(0.0)

    def test_two_groups(self):
        records = [
            _make_record(model="m1", fs_gap=0.1),
            _make_record(model="m2", fs_gap=0.3),
        ]
        result = aggregate_gaps_by_slice(records, by=["model"])
        assert ("m1",) in result
        assert ("m2",) in result
        assert result[("m2",)]["final_score_gap_mean"] == pytest.approx(0.3)

    def test_mean_computed_correctly(self):
        records = [
            _make_record(model="m1", fs_gap=0.2),
            _make_record(model="m1", fs_gap=0.4),
        ]
        result = aggregate_gaps_by_slice(records, by=["model"])
        assert result[("m1",)]["final_score_gap_mean"] == pytest.approx(0.3)

    def test_std_computed_correctly(self):
        import statistics
        records = [
            _make_record(model="m1", fs_gap=0.2),
            _make_record(model="m1", fs_gap=0.4),
        ]
        result = aggregate_gaps_by_slice(records, by=["model"])
        expected_std = statistics.stdev([0.2, 0.4])
        assert result[("m1",)]["final_score_gap_std"] == pytest.approx(expected_std)

    def test_multi_key_grouping(self):
        records = [
            _make_record(model="m1", hop_group="shallow", fs_gap=0.1),
            _make_record(model="m1", hop_group="deep", fs_gap=0.5),
            _make_record(model="m2", hop_group="shallow", fs_gap=0.2),
        ]
        result = aggregate_gaps_by_slice(records, by=["model", "hop_group"])
        assert ("m1", "shallow") in result
        assert ("m1", "deep") in result
        assert ("m2", "shallow") in result
        assert result[("m1", "deep")]["final_score_gap_mean"] == pytest.approx(0.5)

    def test_quadrant_distribution(self):
        records = [
            _make_record(model="m1", quadrant="reasons"),
            _make_record(model="m1", quadrant="reasons"),
            _make_record(model="m1", quadrant="knows"),
        ]
        result = aggregate_gaps_by_slice(records, by=["model"])
        qdist = result[("m1",)]["quadrants"]
        assert qdist["reasons"] == 2
        assert qdist["knows"] == 1

    def test_count_field(self):
        records = [_make_record() for _ in range(5)]
        result = aggregate_gaps_by_slice(records, by=["model"])
        assert result[("m1",)]["count"] == 5

    def test_empty_records(self):
        result = aggregate_gaps_by_slice([], by=["model"])
        assert result == {}

    def test_missing_metadata_key_uses_empty_string(self):
        record = _make_record()
        del record["hop_group"]
        result = aggregate_gaps_by_slice([record], by=["hop_group"])
        assert ("",) in result

    def test_all_gap_fields_in_stats(self):
        records = [_make_record()]
        result = aggregate_gaps_by_slice(records, by=["model"])
        stats = result[("m1",)]
        for field in GAP_FIELDS:
            assert f"{field}_mean" in stats
            assert f"{field}_std" in stats
