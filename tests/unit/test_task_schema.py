"""Тесты Pydantic-схемы Task (v2) — src/mhgb/schemas/task.py"""

import pytest
from pydantic import ValidationError

from mhgb.schemas.task import (
    EXPECTED_ANSWER_FORMAT,
    ContextChunk,
    GoldChainStep,
    Task,
    _hop_group,
)


# ---------------------------------------------------------------------------
# Фикстуры
# ---------------------------------------------------------------------------

def _step(n: int, norm_id: str = "ТК_261") -> dict:
    return {"step": n, "norm_id": norm_id, "reasoning": f"Шаг {n}", "is_critical": True}


def _base_task(**overrides) -> dict:
    base = {
        "id": "task_001_closed",
        "mode": "closed",
        "type": "rule_selection",
        "hop_count": 2,
        "branch_of_law": "трудовое",
        "norm_ids": ["ТК_261", "ТК_256"],
        "fabula": "Сотрудница в декрете уволена.",
        "question": "Законно ли увольнение?",
        "answer": "Незаконно.",
        "gold_chain": [_step(1, "ТК_261"), _step(2, "ТК_256")],
        "context_chunks": None,
        "graph_version_id": "gv_001",
        "corpus_version_id": "cv_001",
        "generator_model": "qwen3-6-27b",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Derived fields
# ---------------------------------------------------------------------------

class TestDerivedFields:
    def test_bloom_level_rule_selection(self):
        t = Task(**_base_task())
        assert t.bloom_level == "apply"

    def test_bloom_level_issue_spotting(self):
        t = Task(**_base_task(type="issue_spotting"))
        assert t.bloom_level == "analyze"

    def test_bloom_level_conflict_resolution(self):
        t = Task(**_base_task(type="conflict_resolution"))
        assert t.bloom_level == "evaluate"

    def test_bloom_level_temporal_validity(self):
        t = Task(**_base_task(type="temporal_validity"))
        assert t.bloom_level == "apply+analyze"

    def test_hop_group_shallow(self):
        t = Task(**_base_task(hop_count=1, gold_chain=[_step(1)]))
        assert t.hop_group == "shallow"

    def test_hop_group_shallow_two(self):
        t = Task(**_base_task())
        assert t.hop_group == "shallow"

    def test_hop_group_medium(self):
        chain = [_step(i) for i in range(1, 4)]
        t = Task(**_base_task(hop_count=3, gold_chain=chain))
        assert t.hop_group == "medium"

    def test_hop_group_deep(self):
        chain = [_step(i) for i in range(1, 6)]
        t = Task(**_base_task(hop_count=5, gold_chain=chain))
        assert t.hop_group == "deep"

    def test_bloom_explicit_not_overwritten(self):
        t = Task(**_base_task(bloom_level="evaluate"))
        assert t.bloom_level == "evaluate"


# ---------------------------------------------------------------------------
# hop_group helper
# ---------------------------------------------------------------------------

class TestHopGroup:
    def test_1(self):   assert _hop_group(1) == "shallow"
    def test_2(self):   assert _hop_group(2) == "shallow"
    def test_3(self):   assert _hop_group(3) == "medium"
    def test_4(self):   assert _hop_group(4) == "medium"
    def test_5(self):   assert _hop_group(5) == "deep"
    def test_10(self):  assert _hop_group(10) == "deep"


# ---------------------------------------------------------------------------
# mode / context_chunks validation
# ---------------------------------------------------------------------------

class TestModeContextChunks:
    def test_closed_no_chunks_ok(self):
        t = Task(**_base_task(mode="closed", context_chunks=None))
        assert t.context_chunks is None

    def test_open_with_chunks_ok(self):
        chunks = [{"norm_id": "ТК_261", "title": "Ст.261", "text": "Текст"}]
        t = Task(**_base_task(mode="open", context_chunks=chunks))
        assert len(t.context_chunks) == 1

    def test_open_without_chunks_fails(self):
        with pytest.raises(ValidationError, match="context_chunks"):
            Task(**_base_task(mode="open", context_chunks=None))

    def test_closed_with_chunks_fails(self):
        chunks = [{"norm_id": "ТК_261", "title": "Ст.261", "text": "Текст"}]
        with pytest.raises(ValidationError, match="context_chunks"):
            Task(**_base_task(mode="closed", context_chunks=chunks))


# ---------------------------------------------------------------------------
# hop_count vs gold_chain consistency
# ---------------------------------------------------------------------------

class TestHopCountConsistency:
    def test_mismatch_raises(self):
        with pytest.raises(ValidationError, match="hop_count"):
            Task(**_base_task(hop_count=3, gold_chain=[_step(1), _step(2)]))

    def test_match_ok(self):
        chain = [_step(i) for i in range(1, 4)]
        t = Task(**_base_task(hop_count=3, gold_chain=chain))
        assert t.hop_count == 3


# ---------------------------------------------------------------------------
# Validation status defaults
# ---------------------------------------------------------------------------

class TestValidationStatus:
    def test_default_pending(self):
        t = Task(**_base_task())
        assert t.validation_status == "pending"

    def test_explicit_status(self):
        t = Task(**_base_task(validation_status="llm_validated"))
        assert t.validation_status == "llm_validated"

    def test_invalid_status(self):
        with pytest.raises(ValidationError):
            Task(**_base_task(validation_status="approved"))


# ---------------------------------------------------------------------------
# GoldChainStep
# ---------------------------------------------------------------------------

class TestGoldChainStep:
    def test_valid(self):
        s = GoldChainStep(step=1, norm_id="ТК_261", reasoning="Запрет на увольнение", is_critical=True)
        assert s.step == 1

    def test_default_is_critical(self):
        s = GoldChainStep(step=1, norm_id="ТК_261", reasoning="r")
        assert s.is_critical is True

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            GoldChainStep(step=1, norm_id="ТК_261", reasoning="r", unknown="x")


# ---------------------------------------------------------------------------
# ContextChunk
# ---------------------------------------------------------------------------

class TestContextChunk:
    def test_valid_no_edge_type(self):
        c = ContextChunk(norm_id="ТК_261", title="Ст.261", text="Текст")
        assert c.edge_type is None

    def test_valid_with_edge_type(self):
        c = ContextChunk(norm_id="ТК_261", title="Ст.261", text="Текст", edge_type="исключает")
        assert c.edge_type == "исключает"

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            ContextChunk(norm_id="x", title="t", text="t", extra_field="y")


# ---------------------------------------------------------------------------
# expected_answer_format
# ---------------------------------------------------------------------------

class TestExpectedAnswerFormat:
    def test_default_contains_three_sections(self):
        t = Task(**_base_task())
        fmt = t.expected_answer_format
        assert "ПРИМЕНИМЫЕ СТАТЬИ" in fmt
        assert "ЦЕПОЧКА РАССУЖДЕНИЙ" in fmt
        assert "ВЫВОД" in fmt

    def test_default_matches_module_constant(self):
        t = Task(**_base_task())
        assert t.expected_answer_format == EXPECTED_ANSWER_FORMAT

    def test_custom_format_accepted(self):
        custom = "ANSWER: ..."
        t = Task(**_base_task(expected_answer_format=custom))
        assert t.expected_answer_format == custom
