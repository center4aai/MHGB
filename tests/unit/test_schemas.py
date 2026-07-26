"""
Smoke-тесты для src/mhgb/storage/schemas.py.
Проверяют, что Pydantic-модели валидируют корректные документы
и отклоняют некорректные. MongoDB не нужна.
"""

import pytest
from datetime import datetime, timezone
from pydantic import ValidationError

from mhgb.storage.schemas import (
    CorpusVersion,
    GraphVersion,
    TaskBatch,
    Experiment,
    ModelResponse,
    Evaluation,
    ValidationLog,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# CorpusVersion
# ---------------------------------------------------------------------------

class TestCorpusVersion:
    def test_valid(self):
        obj = CorpusVersion(
            tag="v1_initial",
            article_count=3787,
            law_counts={"ТК": 531, "ГК": 1713},
            checksum="abc123",
        )
        assert obj.tag == "v1_initial"
        assert obj.article_count == 3787
        assert isinstance(obj.doc_id, str) and len(obj.doc_id) == 36  # UUID
        assert obj.notes is None

    def test_defaults_populated(self):
        obj = CorpusVersion(
            tag="v1", article_count=100,
            law_counts={}, checksum="x",
        )
        assert obj.doc_id != ""
        assert isinstance(obj.created_at, datetime)

    def test_two_instances_have_different_doc_ids(self):
        a = CorpusVersion(tag="v1", article_count=1, law_counts={}, checksum="x")
        b = CorpusVersion(tag="v2", article_count=2, law_counts={}, checksum="y")
        assert a.doc_id != b.doc_id

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            CorpusVersion(
                tag="v1", article_count=1, law_counts={}, checksum="x",
                unknown_field="bad",
            )

    def test_missing_required_field(self):
        with pytest.raises(ValidationError):
            CorpusVersion(tag="v1", article_count=1, law_counts={})  # нет checksum


# ---------------------------------------------------------------------------
# GraphVersion
# ---------------------------------------------------------------------------

class TestGraphVersion:
    def test_valid(self):
        obj = GraphVersion(
            tag="v1_tk_only",
            corpus_version_id="corpus-uuid-123",
            node_count=3757,
            edge_count=4136,
            edge_type_counts={"ссылается_на": 4024, "исключает": 28},
            params={"max_pairs": 200, "model_name": "qwen3-27b"},
        )
        assert obj.node_count == 3757
        assert obj.file_path is None

    def test_with_file_path(self):
        obj = GraphVersion(
            tag="v2", corpus_version_id="x",
            node_count=5000, edge_count=6000,
            edge_type_counts={}, params={},
            file_path="data/graph_v2.json",
        )
        assert obj.file_path == "data/graph_v2.json"


# ---------------------------------------------------------------------------
# TaskBatch
# ---------------------------------------------------------------------------

class TestTaskBatch:
    def test_valid(self):
        obj = TaskBatch(
            tag="v2_full_dataset",
            graph_version_id="graph-uuid",
            corpus_version_id="corpus-uuid",
            task_count=1000,
            task_type_counts={"issue_spotting": 250, "rule_selection": 250},
            generator_model="qwen3-27b",
            seed=42,
        )
        assert obj.task_count == 1000
        assert obj.seed == 42

    def test_without_optional_fields(self):
        obj = TaskBatch(
            tag="v1", graph_version_id="g", corpus_version_id="c",
            task_count=50, task_type_counts={}, generator_model="test",
        )
        assert obj.seed is None
        assert obj.notes is None


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

class TestExperiment:
    def test_valid(self):
        obj = Experiment(
            name="main_v1_qwen3-27b",
            task_batch_id="batch-uuid",
            model_name="qwen3-27b",
            model_size_b=27,
            modes=["closed", "open"],
            context_config="full_graph",
        )
        assert obj.status == "running"
        assert obj.adapted_to_ru is False

    def test_invalid_context_config(self):
        with pytest.raises(ValidationError):
            Experiment(
                name="x", task_batch_id="y", model_name="z",
                modes=["closed"],
                context_config="unknown_config",  # не из Literal
            )

    def test_invalid_mode(self):
        with pytest.raises(ValidationError):
            Experiment(
                name="x", task_batch_id="y", model_name="z",
                modes=["unknown_mode"],  # не closed/open
            )


# ---------------------------------------------------------------------------
# ModelResponse
# ---------------------------------------------------------------------------

class TestModelResponse:
    def test_valid(self):
        obj = ModelResponse(
            experiment_id="exp-uuid",
            task_id="task-abc_closed",
            model_name="qwen3-27b",
            mode="closed",
            raw_response="Увольнение незаконно согласно ст. 261 ТК РФ.",
        )
        assert obj.parsed_articles is None
        assert obj.latency_ms is None

    def test_with_parsed_fields(self):
        obj = ModelResponse(
            experiment_id="exp-uuid",
            task_id="task-abc_closed",
            model_name="qwen3-27b",
            mode="closed",
            raw_response="ответ",
            parsed_articles=["ТК_261", "ТК_256"],
            parsed_conclusion="Незаконно.",
            latency_ms=1500,
            tokens_input=512,
            tokens_output=256,
        )
        assert obj.parsed_articles == ["ТК_261", "ТК_256"]
        assert obj.latency_ms == 1500

    def test_invalid_mode(self):
        with pytest.raises(ValidationError):
            ModelResponse(
                experiment_id="x", task_id="y", model_name="z",
                mode="half-open",  # недопустимый режим
                raw_response="...",
            )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

class TestEvaluation:
    def test_valid_minimal(self):
        obj = Evaluation(
            experiment_id="exp-uuid",
            task_id="task-abc",
            model_response_id="resp-uuid",
        )
        assert obj.final_score is None
        assert obj.step_scores == []
        assert obj.judge_logs == []

    def test_with_scores(self):
        obj = Evaluation(
            experiment_id="exp",
            task_id="task",
            model_response_id="resp",
            norm_coverage_list=0.8,
            norm_coverage_reasoning=0.75,
            step_correctness=0.67,
            answer_correctness=1.0,
            final_score=0.807,
            gap_score=0.15,
            step_scores=[1.0, 0.5, 0.5],
        )
        assert obj.final_score == pytest.approx(0.807)
        assert len(obj.step_scores) == 3


# ---------------------------------------------------------------------------
# ValidationLog
# ---------------------------------------------------------------------------

class TestValidationLog:
    def test_valid_llm(self):
        obj = ValidationLog(
            task_id="task-abc",
            task_batch_id="batch-uuid",
            validator="llm",
            is_valid=True,
            checks={
                "gold_chain_matches_norm_ids": True,
                "task_solvable_from_context": True,
                "fabula_complete": True,
                "no_logical_gaps": True,
                "answer_consistent": True,
            },
        )
        assert obj.validator == "llm"
        assert obj.is_valid is True

    def test_valid_expert(self):
        obj = ValidationLog(
            task_id="task-abc",
            task_batch_id="batch-uuid",
            validator="expert",
            is_valid=False,
            issues=["Некорректная ссылка на ст. 261"],
            judge_explanation="Статья утратила силу на дату фабулы.",
        )
        assert obj.validator == "expert"
        assert len(obj.issues) == 1

    def test_invalid_validator_type(self):
        with pytest.raises(ValidationError):
            ValidationLog(
                task_id="x", task_batch_id="y",
                validator="robot",  # не llm/expert
                is_valid=True,
            )
