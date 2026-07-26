"""
Тесты для src/mhgb/analysis/*.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest

from mhgb.analysis.aggregate_results import (
    aggregate_by_slice,
    enrich_with_task_meta,
    filter_by_task_ids,
    load_results_from_jsonl,
    load_task_ids_filter,
    load_tasks_index,
)
from mhgb.analysis.compute_gap_analysis import pair_closed_open, summarize_gaps
from mhgb.analysis.error_analysis import analyze_errors, classify_error
from mhgb.analysis.generate_tables import fmt, latex_table, markdown_table


# ---------------------------------------------------------------------------
# Вспомогательные фабрики
# ---------------------------------------------------------------------------

def _rec(
    task_id="t1",
    model_name="m1",
    mode="closed",
    norm_coverage_list=0.8,
    norm_coverage_reasoning=0.7,
    step_correctness=0.9,
    answer_correctness=1.0,
    final_score=0.9,
    task_type="rule_selection",
    hop_group="shallow",
    branch_of_law="трудовое",
    **kwargs,
) -> dict:
    return {
        "task_id":               task_id,
        "model_name":            model_name,
        "mode":                  mode,
        "norm_coverage_list":    norm_coverage_list,
        "norm_coverage_reasoning": norm_coverage_reasoning,
        "step_correctness":      step_correctness,
        "answer_correctness":    answer_correctness,
        "final_score":           final_score,
        "task_type":             task_type,
        "hop_group":             hop_group,
        "branch_of_law":         branch_of_law,
        **kwargs,
    }


# ---------------------------------------------------------------------------
# aggregate_results
# ---------------------------------------------------------------------------

class TestEnrichWithTaskMeta:

    def test_adds_task_type_from_index(self):
        records = [{"task_id": "t1", "task_type": None}]
        index = {"t1": {"type": "rule_selection", "hop_group": "shallow", "branch_of_law": "трудовое"}}
        result = enrich_with_task_meta(records, index)
        assert result[0]["task_type"] == "rule_selection"

    def test_adds_hop_group_from_index(self):
        records = [{"task_id": "t1", "hop_group": None}]
        index = {"t1": {"type": "x", "hop_group": "deep", "branch_of_law": "y"}}
        result = enrich_with_task_meta(records, index)
        assert result[0]["hop_group"] == "deep"

    def test_existing_task_type_preserved(self):
        records = [{"task_id": "t1", "task_type": "conflict_resolution"}]
        index = {"t1": {"type": "rule_selection", "hop_group": "shallow", "branch_of_law": "x"}}
        result = enrich_with_task_meta(records, index)
        assert result[0]["task_type"] == "conflict_resolution"

    def test_missing_task_gives_unknown(self):
        records = [{"task_id": "no_such_id", "task_type": None, "hop_group": None, "branch_of_law": None}]
        result = enrich_with_task_meta(records, {})
        assert result[0]["task_type"] == "unknown"
        assert result[0]["hop_group"] == "unknown"

    def test_does_not_mutate_input(self):
        orig = {"task_id": "t1", "task_type": None}
        result = enrich_with_task_meta([orig], {"t1": {"type": "x", "hop_group": "y", "branch_of_law": "z"}})
        assert orig["task_type"] is None  # original unchanged
        assert result[0]["task_type"] == "x"

    def test_empty_records(self):
        assert enrich_with_task_meta([], {}) == []


class TestAggregateBySlice:

    def test_correct_count(self):
        records = [_rec(model_name="m1") for _ in range(3)]
        rows = aggregate_by_slice(records, by=["model_name"])
        assert rows[0]["count"] == 3

    def test_mean_correct(self):
        records = [_rec(model_name="m1", final_score=0.4), _rec(model_name="m1", final_score=0.6)]
        rows = aggregate_by_slice(records, by=["model_name"])
        assert rows[0]["final_score_mean"] == pytest.approx(0.5)

    def test_two_groups(self):
        records = [_rec(model_name="m1"), _rec(model_name="m2")]
        rows = aggregate_by_slice(records, by=["model_name"])
        assert len(rows) == 2

    def test_multi_key_slice(self):
        records = [_rec(model_name="m1", mode="closed"), _rec(model_name="m1", mode="open_full_graph")]
        rows = aggregate_by_slice(records, by=["model_name", "mode"])
        assert len(rows) == 2

    def test_std_zero_for_single_record(self):
        records = [_rec(model_name="m1", final_score=0.7)]
        rows = aggregate_by_slice(records, by=["model_name"])
        assert rows[0]["final_score_std"] == 0.0

    def test_empty_records_returns_empty(self):
        assert aggregate_by_slice([], by=["model_name"]) == []

    def test_none_metric_ignored(self):
        records = [_rec(model_name="m1", final_score=0.8), {"model_name": "m1", "final_score": None}]
        rows = aggregate_by_slice(records, by=["model_name"])
        assert rows[0]["final_score_mean"] == pytest.approx(0.8)


class TestLoadResultsFromJsonl:

    def test_skips_error_records(self, tmp_path):
        p = tmp_path / "r.jsonl"
        good = {"task_id": "t1", "model_name": "m1", "final_score": 0.9}
        bad  = {"task_id": "t2", "model_name": "m1", "error": "connection failed"}
        p.write_text(json.dumps(good) + "\n" + json.dumps(bad), encoding="utf-8")
        records = load_results_from_jsonl(str(p))
        assert len(records) == 1
        assert records[0]["task_id"] == "t1"

    def test_loads_all_valid(self, tmp_path):
        p = tmp_path / "r.jsonl"
        data = [{"task_id": f"t{i}", "model_name": "m", "final_score": 0.5} for i in range(5)]
        p.write_text("\n".join(json.dumps(r) for r in data), encoding="utf-8")
        assert len(load_results_from_jsonl(str(p))) == 5

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text("", encoding="utf-8")
        assert load_results_from_jsonl(str(p)) == []


class TestLoadTasksIndex:

    def test_index_by_id(self, tmp_path):
        p = tmp_path / "tasks.jsonl"
        task = {"id": "t1", "type": "rule_selection", "hop_group": "shallow"}
        p.write_text(json.dumps(task), encoding="utf-8")
        index = load_tasks_index(str(p))
        assert "t1" in index
        assert index["t1"]["type"] == "rule_selection"


class TestLoadTaskIdsFilter:

    def _write(self, tmp_path, tasks: list[dict]) -> Path:
        p = tmp_path / "filter.jsonl"
        p.write_text("\n".join(json.dumps(t) for t in tasks), encoding="utf-8")
        return p

    def test_strips_closed_suffix(self, tmp_path):
        p = self._write(tmp_path, [{"id": "abc123_closed"}])
        result = load_task_ids_filter(str(p))
        assert "abc123" in result
        assert "abc123_closed" not in result

    def test_strips_open_suffix(self, tmp_path):
        p = self._write(tmp_path, [{"id": "abc123_open"}])
        result = load_task_ids_filter(str(p))
        assert "abc123" in result

    def test_no_suffix_preserved(self, tmp_path):
        p = self._write(tmp_path, [{"id": "abc123"}])
        result = load_task_ids_filter(str(p))
        assert "abc123" in result

    def test_returns_set_deduplicates(self, tmp_path):
        p = self._write(tmp_path, [{"id": "abc_closed"}, {"id": "abc_open"}])
        result = load_task_ids_filter(str(p))
        assert result == {"abc"}

    def test_mixed_suffixes(self, tmp_path):
        tasks = [{"id": "t1_closed"}, {"id": "t2_open"}, {"id": "t3"}]
        p = self._write(tmp_path, tasks)
        result = load_task_ids_filter(str(p))
        assert result == {"t1", "t2", "t3"}

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text("", encoding="utf-8")
        assert load_task_ids_filter(str(p)) == set()

    def test_skips_lines_without_id(self, tmp_path):
        p = self._write(tmp_path, [{"other": "no_id"}, {"id": "valid_closed"}])
        result = load_task_ids_filter(str(p))
        assert "" in result or "valid" in result
        assert "valid" in result


class TestFilterByTaskIds:

    def test_keeps_record_with_closed_suffix(self):
        records = [{"task_id": "abc_closed", "final_score": 0.9}]
        result = filter_by_task_ids(records, {"abc"})
        assert len(result) == 1

    def test_keeps_record_with_open_suffix(self):
        records = [{"task_id": "abc_open", "final_score": 0.7}]
        result = filter_by_task_ids(records, {"abc"})
        assert len(result) == 1

    def test_excludes_record_not_in_filter(self):
        records = [{"task_id": "xyz_closed", "final_score": 0.5}]
        result = filter_by_task_ids(records, {"abc"})
        assert result == []

    def test_empty_filter_returns_empty(self):
        records = [{"task_id": "abc_closed"}]
        assert filter_by_task_ids(records, set()) == []

    def test_empty_records_returns_empty(self):
        assert filter_by_task_ids([], {"abc"}) == []

    def test_mixed_in_and_out(self):
        records = [
            {"task_id": "a_closed"},
            {"task_id": "b_closed"},
            {"task_id": "c_closed"},
        ]
        result = filter_by_task_ids(records, {"a", "c"})
        ids = {r["task_id"] for r in result}
        assert ids == {"a_closed", "c_closed"}

    def test_record_without_suffix_matches(self):
        records = [{"task_id": "abc"}]
        result = filter_by_task_ids(records, {"abc"})
        assert len(result) == 1

    def test_does_not_mutate_input(self):
        orig = [{"task_id": "abc_closed", "x": 1}]
        filter_by_task_ids(orig, {"abc"})
        assert orig[0]["task_id"] == "abc_closed"


# ---------------------------------------------------------------------------
# compute_gap_analysis
# ---------------------------------------------------------------------------

class TestPairClosedOpen:

    def _pair(self, task_id="t1", model="m1", c_score=0.4, o_score=0.7):
        return [
            _rec(task_id=task_id, model_name=model, mode="closed",          final_score=c_score),
            _rec(task_id=task_id, model_name=model, mode="open_full_graph",  final_score=o_score),
        ]

    def test_produces_one_pair(self):
        pairs = pair_closed_open(self._pair())
        assert len(pairs) == 1

    def test_gap_computed_correctly(self):
        pairs = pair_closed_open(self._pair(c_score=0.3, o_score=0.8))
        assert pairs[0]["final_score_gap"] == pytest.approx(0.5)

    def test_negative_gap(self):
        pairs = pair_closed_open(self._pair(c_score=0.9, o_score=0.4))
        assert pairs[0]["final_score_gap"] < 0

    def test_missing_one_mode_excluded(self):
        records = [_rec(mode="closed")]
        assert pair_closed_open(records) == []

    def test_error_records_excluded(self):
        records = [
            {"task_id": "t1", "model_name": "m1", "mode": "closed", "error": "fail"},
            _rec(task_id="t1", mode="open_full_graph"),
        ]
        assert pair_closed_open(records) == []

    def test_quadrant_field_set(self):
        pairs = pair_closed_open(self._pair(c_score=0.2, o_score=0.8))
        assert pairs[0]["quadrant"] in ("knows", "reasons", "hallucinates", "incompetent")

    def test_two_models_two_pairs(self):
        pairs = pair_closed_open(self._pair(model="m1") + self._pair(model="m2"))
        assert len(pairs) == 2

    def test_metadata_inherited_from_closed(self):
        records = [
            _rec(task_id="t1", model_name="m1", mode="closed",         task_type="conflict_resolution"),
            _rec(task_id="t1", model_name="m1", mode="open_full_graph", task_type="conflict_resolution"),
        ]
        pairs = pair_closed_open(records)
        assert pairs[0]["task_type"] == "conflict_resolution"


class TestSummarizeGaps:

    def _gap(self, model="m1", gap=0.2, quadrant="reasons"):
        return {
            "model_name":          model,
            "task_type":           "rule_selection",
            "hop_group":           "shallow",
            "norm_coverage_gap":   0.1,
            "step_correctness_gap": 0.1,
            "answer_correctness_gap": 0.1,
            "final_score_gap":     gap,
            "quadrant":            quadrant,
        }

    def test_correct_groups(self):
        rows = summarize_gaps([self._gap("m1"), self._gap("m2")], by=["model_name"])
        assert len(rows) == 2

    def test_mean_correct(self):
        rows = summarize_gaps([self._gap("m1", 0.2), self._gap("m1", 0.4)], by=["model_name"])
        assert rows[0]["final_score_gap_mean"] == pytest.approx(0.3)

    def test_quadrant_counts(self):
        rows = summarize_gaps(
            [self._gap("m1", quadrant="reasons"), self._gap("m1", quadrant="knows")],
            by=["model_name"],
        )
        assert rows[0]["q_reasons"] == 1
        assert rows[0]["q_knows"]   == 1

    def test_n_pairs(self):
        rows = summarize_gaps([self._gap("m1"), self._gap("m1")], by=["model_name"])
        assert rows[0]["n_pairs"] == 2

    def test_empty_returns_empty(self):
        assert summarize_gaps([], by=["model_name"]) == []


# ---------------------------------------------------------------------------
# error_analysis
# ---------------------------------------------------------------------------

class TestClassifyError:

    def test_ok_above_threshold(self):
        assert classify_error(_rec(final_score=0.8)) == "ok"

    def test_ok_exactly_threshold(self):
        assert classify_error(_rec(final_score=0.5)) == "ok"

    def test_norm_miss(self):
        r = _rec(final_score=0.2, norm_coverage_list=0.1, norm_coverage_reasoning=0.15,
                  step_correctness=0.6, answer_correctness=0.0)
        assert classify_error(r) == "norm_miss"

    def test_reasoning_fail(self):
        r = _rec(final_score=0.3, norm_coverage_list=0.8, norm_coverage_reasoning=0.8,
                  step_correctness=0.1, answer_correctness=0.0)
        assert classify_error(r) == "reasoning_fail"

    def test_answer_fail(self):
        r = _rec(final_score=0.3, norm_coverage_list=0.8, norm_coverage_reasoning=0.8,
                  step_correctness=0.8, answer_correctness=0.0)
        assert classify_error(r) == "answer_fail"

    def test_multi_fail_all_low(self):
        r = _rec(final_score=0.1, norm_coverage_list=0.1, norm_coverage_reasoning=0.15,
                  step_correctness=0.1, answer_correctness=0.1)
        assert classify_error(r) == "multi_fail"

    def test_hallucination_gap(self):
        r = _rec(final_score=0.2, norm_coverage_list=0.1, norm_coverage_reasoning=0.8,
                  step_correctness=0.3, answer_correctness=0.0)
        assert classify_error(r) == "hallucination"

    def test_custom_threshold(self):
        r = _rec(final_score=0.7)
        assert classify_error(r, threshold=0.8) != "ok"

    def test_none_values_treated_as_zero(self):
        r = {**_rec(final_score=0.1), "norm_coverage_list": None, "step_correctness": None,
             "answer_correctness": None, "norm_coverage_reasoning": None}
        result = classify_error(r)
        assert result in ("multi_fail", "norm_miss", "reasoning_fail", "answer_fail", "hallucination")


class TestAnalyzeErrors:

    def test_total_excludes_error_records(self):
        records = [_rec(final_score=0.9), {"task_id": "t2", "error": "fail"}]
        result = analyze_errors(records)
        assert result["total"] == 1

    def test_ok_counted(self):
        assert analyze_errors([_rec(final_score=0.9)])["overall"]["ok"] == 1

    def test_by_model_populated(self):
        result = analyze_errors([_rec(model_name="m1", final_score=0.9)])
        assert "m1" in result["by_model"]

    def test_overall_pct_sums_to_1(self):
        records = [
            _rec(final_score=0.9),
            _rec(task_id="t2", final_score=0.1, norm_coverage_list=0.1,
                 step_correctness=0.1, answer_correctness=0.1),
        ]
        result = analyze_errors(records, threshold=0.5)
        assert sum(result["overall_pct"].values()) == pytest.approx(1.0)

    def test_empty_records(self):
        result = analyze_errors([])
        assert result["total"] == 0

    def test_by_task_type_populated(self):
        result = analyze_errors([_rec(task_type="temporal_validity", final_score=0.1,
                                       norm_coverage_list=0.1, step_correctness=0.1,
                                       answer_correctness=0.1)])
        assert "temporal_validity" in result["by_task_type"]


# ---------------------------------------------------------------------------
# generate_tables
# ---------------------------------------------------------------------------

class TestFmt:

    def test_float(self):
        assert fmt(0.12345) == "0.123"

    def test_none_returns_dash(self):
        assert fmt(None) == "—"

    def test_empty_string_returns_dash(self):
        assert fmt("") == "—"

    def test_non_numeric_string(self):
        assert fmt("hello") == "hello"

    def test_custom_decimals(self):
        assert fmt(0.123456, decimals=2) == "0.12"


class TestMarkdownTable:

    def test_has_header(self):
        md = markdown_table(["A", "B"], [["1", "2"]])
        assert "A" in md and "B" in md

    def test_has_separator_row(self):
        md = markdown_table(["A"], [["x"]])
        assert "|-" in md or "|---" in md

    def test_data_in_output(self):
        md = markdown_table(["X"], [["hello"]])
        assert "hello" in md

    def test_correct_row_count(self):
        md = markdown_table(["A"], [["1"], ["2"], ["3"]])
        assert md.count("|") > 0
        lines = md.strip().splitlines()
        assert len(lines) == 5  # header + sep + 3 data rows


class TestLatexTable:

    def test_has_tabular(self):
        assert "\\begin{tabular}" in latex_table(["A"], [["x"]])

    def test_has_toprule(self):
        assert "\\toprule" in latex_table(["A"], [["x"]])

    def test_caption(self):
        assert "My caption" in latex_table(["A"], [["x"]], caption="My caption")

    def test_label(self):
        assert "tab:foo" in latex_table(["A"], [["x"]], label="tab:foo")

    def test_data_row(self):
        latex = latex_table(["A", "B"], [["val1", "val2"]])
        assert "val1" in latex and "val2" in latex


# ---------------------------------------------------------------------------
# Тесты фильтрации ошибочных RtA-записей (compute_rta_analysis)
# ---------------------------------------------------------------------------

from mhgb.analysis.compute_rta_analysis import (
    is_answered,
    is_genuine_rta,
    compute_rta_rate,
    load_rta_data,
    compute_rta_summary,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records),
        encoding="utf-8",
    )


class TestIsAnswered:
    """Знаменатель RtA: содержательный ответ (не error/overflow/пустой).
    Петли/cap ОСТАЮТСЯ (модель ответила плохо = failure mode)."""

    def test_error_not_answered(self):
        assert is_answered({"error": "overflow"}) is False

    def test_empty_not_answered(self):
        assert is_answered({"raw_response": ""}) is False
        assert is_answered({"raw_response": "   "}) is False
        assert is_answered({"raw_response": None}) is False

    def test_content_answered(self):
        assert is_answered({"raw_response": "ПРИМЕНИМЫЕ СТАТЬИ: ..."}) is True

    def test_cap_loop_still_answered(self):
        # дегенеративная петля упёрлась в cap — это ответ (плохой), не «нет ответа»
        assert is_answered({"raw_response": "ст. 1 ст. 2 ст. 3 ..."}) is True


class TestIsGenuineRta:
    """Числитель RtA: is_rta И содержательный (не пустой/cap). По ТЕКСТУ, не topic."""

    def test_real_refusal_genuine(self):
        r = {"is_rta": True, "raw_response": "Не могу дать оценку.", "tokens_output": 20}
        assert is_genuine_rta(r, max_tokens=2048) is True

    def test_generic_refusal_topic_none_still_genuine(self):
        # topic=None легитимен у generic-отказов слабых моделей — НЕ артефакт
        r = {"is_rta": True, "rta_topic": None, "raw_response": "Отказываюсь.", "tokens_output": 15}
        assert is_genuine_rta(r, max_tokens=2048) is True

    def test_empty_not_genuine(self):
        r = {"is_rta": True, "raw_response": "", "tokens_output": 9}
        assert is_genuine_rta(r, max_tokens=2048) is False

    def test_cap_loop_not_genuine(self):
        # cap-петля ложно помечена is_rta детектором → не отказ
        r = {"is_rta": True, "raw_response": "ст.1 ст.2 ...", "tokens_output": 2048}
        assert is_genuine_rta(r, max_tokens=2048) is False

    def test_not_rta_not_genuine(self):
        r = {"is_rta": False, "raw_response": "обычный ответ", "tokens_output": 100}
        assert is_genuine_rta(r, max_tokens=2048) is False

    def test_error_not_genuine(self):
        r = {"is_rta": True, "error": "overflow"}
        assert is_genuine_rta(r, max_tokens=2048) is False


class TestComputeRtaRate:
    def test_rate_excludes_artifacts_from_numerator_keeps_loops_in_denom(self):
        records = [
            {"task_id": "a", "is_rta": True,  "raw_response": "Отказ.",       "tokens_output": 20},   # genuine
            {"task_id": "b", "is_rta": True,  "raw_response": "ст.1 ст.2 ...", "tokens_output": 2048}, # cap-петля (артефакт)
            {"task_id": "c", "is_rta": True,  "raw_response": "",             "tokens_output": 9},    # пустой (артефакт)
            {"task_id": "d", "is_rta": False, "raw_response": "ответ",        "tokens_output": 100},  # ответ
            {"task_id": "e", "error": "overflow"},                                                    # нет ответа
        ]
        rate = compute_rta_rate(records, max_tokens=2048)
        # знаменатель: a,b,c,d (не error, непустой) = 4; петля b и пустой c... c пустой → вон
        assert rate["n_answered"] == 3   # a, b, d (c пустой исключён, e error исключён)
        assert rate["n_rta"] == 1        # только a (b cap-петля, c пустой — артефакты)
        assert rate["rta_rate"] == pytest.approx(1 / 3)


class TestLoadRtaData:
    def test_base_is_results_dedup_keeplast(self, tmp_path):
        # retry-дубль: t1_closed упал (error), затем успешно (ok) — keep-last → ok
        exp = tmp_path / "exp1"
        exp.mkdir()
        _write_jsonl(exp / "results.jsonl", [
            {"task_id": "t1_closed", "error": "timeout"},                       # упавшая попытка
            {"task_id": "t1_closed", "raw_response": "ответ", "tokens_output": 50},  # retry ok
            {"task_id": "t1_open",   "raw_response": "ответ2", "tokens_output": 60},
            {"task_id": "t2_open",   "error": "context overflow"},              # реальный overflow
        ])
        _write_jsonl(exp / "results_rta.jsonl", [
            {"task_id": "t1_closed", "is_rta": True,  "rta_type": "generic", "rta_topic": None},
            {"task_id": "t1_open",   "is_rta": False},
            {"task_id": "t2_open",   "is_rta": True},   # ложный (overflow)
        ])
        records = load_rta_data("exp1", tmp_path)
        # 3 уникальных task_id (дедуп); t1_closed = ok-версия (retry разрешён)
        assert len(records) == 3
        by_id = {r["task_id"]: r for r in records}
        assert "error" not in by_id["t1_closed"]              # keep-last → ok
        assert by_id["t1_closed"]["is_rta"] is True           # обогащён из rta
        assert by_id["t2_open"]["is_rta"] is False            # error → не отказ

    def test_summary_genuine_only(self, tmp_path):
        exp = tmp_path / "exp1"
        exp.mkdir()
        _write_jsonl(exp / "results.jsonl", [
            {"task_id": "t1_closed", "raw_response": "Отказ.",   "tokens_output": 20},
            {"task_id": "t1_open",   "raw_response": "ответ",    "tokens_output": 30},
            {"task_id": "t2_closed", "raw_response": "",         "tokens_output": 9},  # пустой
            {"task_id": "t2_open",   "error": "context overflow"},
        ])
        _write_jsonl(exp / "results_rta.jsonl", [
            {"task_id": "t1_closed", "is_rta": True,  "mode": "closed"},   # genuine
            {"task_id": "t1_open",   "is_rta": False, "mode": "open"},
            {"task_id": "t2_closed", "is_rta": True,  "mode": "closed"},   # пустой → артефакт
            {"task_id": "t2_open",   "is_rta": True,  "mode": "open"},     # overflow → артефакт
        ])
        records = load_rta_data("exp1", tmp_path)
        summary = compute_rta_summary(records)
        # знаменатель = содержательные (t1_closed, t1_open) = 2; t2_closed пустой, t2_open error
        assert summary["n_total"] == 2
        assert summary["n_rta"] == 1   # только t1_closed
        assert summary["rta_rate"] == pytest.approx(1 / 2)
