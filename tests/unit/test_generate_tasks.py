"""
Smoke-тесты для src/mhgb/generate_tasks.py.
LLM не вызывается — тестируются только pure functions и схема задачи.
"""

import pytest
import networkx as nx
from mhgb.generate_tasks import (
    _hop_group_of,
    _parse_llm_json,
    _is_usable,
    _build_task,
    select_rule_selection,
    HOP_GROUP_DEPTHS,
    MAX_HOP_DEPTH,
)

REQUIRED_TASK_FIELDS = {
    "id", "mode", "type",
    "hop_count", "hop_group",
    "norm_ids", "fabula", "question", "answer",
    "gold_chain", "expected_answer_format",
}


class TestParseLlmJson:
    def test_clean_json(self):
        raw = '{"fabula": "тест", "question": "вопрос?"}'
        result = _parse_llm_json(raw)
        assert result == {"fabula": "тест", "question": "вопрос?"}

    def test_strips_think_block(self):
        raw = "<think>долгое рассуждение...</think>{\"fabula\": \"тест\"}"
        result = _parse_llm_json(raw)
        assert result == {"fabula": "тест"}

    def test_multiline_think_block(self):
        raw = "<think>\nстрока1\nстрока2\n</think>\n{\"answer\": \"да\"}"
        result = _parse_llm_json(raw)
        assert result == {"answer": "да"}

    def test_json_embedded_in_text(self):
        raw = 'Вот мой ответ: {"fabula": "тест"} — готово.'
        result = _parse_llm_json(raw)
        assert result == {"fabula": "тест"}

    def test_invalid_json_returns_none(self):
        assert _parse_llm_json("не JSON вообще") is None

    def test_empty_string_returns_none(self):
        assert _parse_llm_json("") is None

    def test_only_think_block_returns_none(self):
        assert _parse_llm_json("<think>только мысли</think>") is None

    def test_nested_json(self):
        raw = '{"gold_chain": [{"step": 1, "norm_id": "ТК_261"}]}'
        result = _parse_llm_json(raw)
        assert result is not None
        assert result["gold_chain"][0]["norm_id"] == "ТК_261"


class TestIsUsable:
    def test_normal_article_is_usable(self, tmp_corpus):
        assert _is_usable("ТК_261", tmp_corpus) is True

    def test_repealed_article_not_usable(self, tmp_corpus):
        assert _is_usable("ТК_999", tmp_corpus) is False

    def test_short_text_not_usable(self):
        corpus = {
            "ТК_1": {
                "id": "ТК_1",
                "text": "короткий текст",
                "is_repealed": False,
            }
        }
        assert _is_usable("ТК_1", corpus) is False

    def test_unknown_id_not_usable(self, tmp_corpus):
        assert _is_usable("UNKNOWN_999", tmp_corpus) is False

    def test_text_exactly_100_chars_is_usable(self):
        corpus = {
            "ТК_1": {
                "id": "ТК_1",
                "text": "а" * 100,
                "is_repealed": False,
            }
        }
        assert _is_usable("ТК_1", corpus) is True

    def test_text_99_chars_not_usable(self):
        corpus = {
            "ТК_1": {
                "id": "ТК_1",
                "text": "а" * 99,
                "is_repealed": False,
            }
        }
        assert _is_usable("ТК_1", corpus) is False


class TestBuildTask:
    def _make_llm_output(self):
        return {
            "fabula": "Сотрудница Иванова находится в декретном отпуске и получила приказ об увольнении.",
            "question": "Законно ли увольнение Ивановой?",
            "answer": "Нет, увольнение незаконно согласно ст. 261 ТК РФ.",
            "gold_chain": [
                {"step": 1, "norm_id": "ТК_261", "reasoning": "Запрещает увольнение."},
                {"step": 2, "norm_id": None, "conclusion": "Итог: незаконно."},
            ],
        }

    def test_returns_exactly_two_tasks(self, tmp_corpus):
        tasks = _build_task("rule_selection", ["ТК_261", "ТК_256"], tmp_corpus, self._make_llm_output())
        assert len(tasks) == 2

    def test_has_closed_and_open_modes(self, tmp_corpus):
        tasks = _build_task("issue_spotting", ["ТК_261"], tmp_corpus, self._make_llm_output())
        modes = {t["mode"] for t in tasks}
        assert modes == {"closed", "open"}

    def test_all_required_fields_present(self, tmp_corpus):
        tasks = _build_task("issue_spotting", ["ТК_261"], tmp_corpus, self._make_llm_output())
        for task in tasks:
            missing = REQUIRED_TASK_FIELDS - task.keys()
            assert not missing, f"Отсутствуют поля: {missing}"

    def test_closed_has_no_context_chunks(self, tmp_corpus):
        tasks = _build_task("rule_selection", ["ТК_261", "ТК_256"], tmp_corpus, self._make_llm_output())
        closed = next(t for t in tasks if t["mode"] == "closed")
        assert closed["context_chunks"] is None

    def test_open_has_context_chunks(self, tmp_corpus):
        tasks = _build_task("rule_selection", ["ТК_261", "ТК_256"], tmp_corpus, self._make_llm_output())
        open_task = next(t for t in tasks if t["mode"] == "open")
        assert open_task["context_chunks"] is not None
        assert len(open_task["context_chunks"]) == 2

    def test_context_chunks_have_norm_id_and_text(self, tmp_corpus):
        tasks = _build_task("rule_selection", ["ТК_261", "ТК_256"], tmp_corpus, self._make_llm_output())
        open_task = next(t for t in tasks if t["mode"] == "open")
        for chunk in open_task["context_chunks"]:
            assert "norm_id" in chunk
            assert "text" in chunk

    def test_norm_ids_preserved(self, tmp_corpus):
        norm_ids = ["ТК_261", "ТК_256", "ТК_81"]
        tasks = _build_task("conflict_resolution", norm_ids, tmp_corpus, self._make_llm_output())
        for task in tasks:
            assert task["norm_ids"] == norm_ids

    def test_task_type_preserved(self, tmp_corpus):
        tasks = _build_task("temporal_validity", ["ТК_261"], tmp_corpus, self._make_llm_output())
        for task in tasks:
            assert task["type"] == "temporal_validity"

    def test_ids_are_unique(self, tmp_corpus):
        tasks = _build_task("issue_spotting", ["ТК_261"], tmp_corpus, self._make_llm_output())
        assert tasks[0]["id"] != tasks[1]["id"]

    def test_hop_count_matches_norm_ids(self, tmp_corpus):
        tasks = _build_task("rule_selection", ["ТК_261", "ТК_256", "ТК_81"], tmp_corpus, self._make_llm_output())
        for task in tasks:
            assert task["hop_count"] == 3

    def test_hop_group_shallow_for_one_norm(self, tmp_corpus):
        tasks = _build_task("issue_spotting", ["ТК_261"], tmp_corpus, self._make_llm_output())
        for task in tasks:
            assert task["hop_group"] == "shallow"

    def test_hop_group_medium_for_three_norms(self, tmp_corpus):
        tasks = _build_task("rule_selection", ["ТК_261", "ТК_256", "ТК_81"], tmp_corpus, self._make_llm_output())
        for task in tasks:
            assert task["hop_group"] == "medium"

    def test_hop_group_deep_for_six_norms(self, tmp_corpus):
        norms = ["ТК_261"] * 6
        tasks = _build_task("rule_selection", norms, tmp_corpus, self._make_llm_output())
        for task in tasks:
            assert task["hop_group"] == "deep"


class TestHopGroupOf:
    def test_1_is_shallow(self):   assert _hop_group_of(1) == "shallow"
    def test_2_is_shallow(self):   assert _hop_group_of(2) == "shallow"
    def test_3_is_medium(self):    assert _hop_group_of(3) == "medium"
    def test_4_is_medium(self):    assert _hop_group_of(4) == "medium"
    def test_5_is_deep(self):      assert _hop_group_of(5) == "deep"
    def test_10_is_deep(self):     assert _hop_group_of(10) == "deep"


class TestHopGroupDepths:
    def test_shallow_range(self):
        assert HOP_GROUP_DEPTHS["shallow"] == (1, 2)

    def test_medium_range(self):
        assert HOP_GROUP_DEPTHS["medium"] == (3, 4)

    def test_deep_range(self):
        min_d, max_d = HOP_GROUP_DEPTHS["deep"]
        assert min_d == 5
        assert max_d == MAX_HOP_DEPTH

    def test_max_hop_depth_at_least_10(self):
        assert MAX_HOP_DEPTH >= 10


class TestSelectRuleSelection:
    def _make_graph(self) -> tuple[nx.DiGraph, dict]:
        G = nx.DiGraph()
        corpus = {}
        for i in range(1, 20):
            nid = f"ТК_{i}"
            G.add_node(nid)
            corpus[nid] = {"id": nid, "text": "а" * 200, "is_repealed": False}
        # Добавляем семантические рёбра: цепочка 1→2→3→...→15
        for i in range(1, 15):
            G.add_edge(f"ТК_{i}", f"ТК_{i+1}", edge_type="применяется_к")
        return G, corpus

    def test_returns_shallow_chains(self):
        G, corpus = self._make_graph()
        chains = select_rule_selection(G, corpus, n=5, target_hop_group="shallow")
        for c in chains:
            assert len(c["norm_ids"]) <= 2

    def test_returns_medium_chains(self):
        G, corpus = self._make_graph()
        chains = select_rule_selection(G, corpus, n=5, target_hop_group="medium")
        for c in chains:
            assert 3 <= len(c["norm_ids"]) <= 4

    def test_returns_deep_chains(self):
        G, corpus = self._make_graph()
        chains = select_rule_selection(G, corpus, n=3, target_hop_group="deep")
        for c in chains:
            assert len(c["norm_ids"]) >= 5

    def test_no_hop_group_returns_chains(self):
        G, corpus = self._make_graph()
        chains = select_rule_selection(G, corpus, n=5)
        assert len(chains) > 0

    def test_empty_graph_returns_empty(self):
        G = nx.DiGraph()
        assert select_rule_selection(G, {}, n=5) == []
