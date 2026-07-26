"""Тесты для src/mhgb/eval/context_builders.py"""

import networkx as nx
import pytest

from mhgb.eval.context_builders import (
    DOCTRINAL_EDGE_TYPES,
    STRUCTURAL_EDGE_TYPES,
    _get_neighbors,
    _to_chunk,
    build_full_graph_context,
    build_full_graph_edges_context,
)
from mhgb.schemas.task import ContextChunk


# ---------------------------------------------------------------------------
# Фикстуры
# ---------------------------------------------------------------------------

@pytest.fixture
def corpus():
    """Минимальный корпус: 6 статей."""
    return {
        "ТК_261": {"id": "ТК_261", "law_short": "ТК", "article": "261",
                   "title": "Гарантии беременным", "text": "Текст 261"},
        "ТК_81":  {"id": "ТК_81",  "law_short": "ТК", "article": "81",
                   "title": "",                         "text": "Текст 81"},
        "ТК_256": {"id": "ТК_256", "law_short": "ТК", "article": "256",
                   "title": "Отпуск по уходу",         "text": "Текст 256"},
        "ГК_10":  {"id": "ГК_10",  "law_short": "ГК", "article": "10",
                   "title": "Пределы прав",             "text": "Текст ГК_10"},
        "ГК_11":  {"id": "ГК_11",  "law_short": "ГК", "article": "11",
                   "title": "",                         "text": "Текст ГК_11"},
        "ТК_392": {"id": "ТК_392", "law_short": "ТК", "article": "392",
                   "title": "Сроки обращения",         "text": "Текст 392"},
    }


@pytest.fixture
def corpus_with_dates():
    """Корпус с временны́ми метаданными для тестирования P2-0.1."""
    return {
        "ТК_261": {"id": "ТК_261", "law_short": "ТК", "article": "261",
                   "title": "Гарантии беременным", "text": "Текст 261",
                   "valid_from": "01.02.2002", "valid_to": None, "is_repealed": False},
        "ТК_81":  {"id": "ТК_81",  "law_short": "ТК", "article": "81",
                   "title": "Расторжение",         "text": "Текст 81",
                   "valid_from": None, "valid_to": "15.03.2023", "is_repealed": True},
        "ТК_256": {"id": "ТК_256", "law_short": "ТК", "article": "256",
                   "title": "Отпуск по уходу",     "text": "Текст 256",
                   "valid_from": None, "valid_to": None, "is_repealed": None},
    }


@pytest.fixture
def graph():
    """
    Граф:
      ТК_261 --исключает-->    ТК_81
      ТК_261 --дополняет-->    ТК_256
      ТК_256 --ссылается_на--> ТК_392
      ГК_10  --применяется_к-> ГК_11
      ТК_392 --приоритет-->    ГК_10
    """
    G = nx.DiGraph()
    G.add_edge("ТК_261", "ТК_81",  edge_type="исключает")
    G.add_edge("ТК_261", "ТК_256", edge_type="дополняет")
    G.add_edge("ТК_256", "ТК_392", edge_type="ссылается_на")
    G.add_edge("ГК_10",  "ГК_11",  edge_type="применяется_к")
    G.add_edge("ТК_392", "ГК_10",  edge_type="приоритет")
    return G


# ---------------------------------------------------------------------------
# _to_chunk
# ---------------------------------------------------------------------------

class TestToChunk:
    def test_found_returns_chunk(self, corpus):
        chunk = _to_chunk("ТК_261", corpus)
        assert isinstance(chunk, ContextChunk)
        assert chunk.norm_id == "ТК_261"
        assert chunk.text == "Текст 261"

    def test_not_found_returns_none(self, corpus):
        assert _to_chunk("НЕСУЩЕСТВУЮЩАЯ_99", corpus) is None

    def test_empty_title_uses_fallback(self, corpus):
        chunk = _to_chunk("ТК_81", corpus)
        assert chunk is not None
        assert "81" in chunk.title
        assert "ТК" in chunk.title

    def test_edge_type_preserved(self, corpus):
        chunk = _to_chunk("ТК_261", corpus, edge_type="исключает")
        assert chunk.edge_type == "исключает"

    def test_no_edge_type_is_none(self, corpus):
        chunk = _to_chunk("ТК_261", corpus)
        assert chunk.edge_type is None

    def test_related_to_preserved(self, corpus):
        chunk = _to_chunk("ТК_261", corpus, related_to="ТК_81")
        assert chunk.related_to == "ТК_81"

    def test_no_related_to_is_none(self, corpus):
        chunk = _to_chunk("ТК_261", corpus)
        assert chunk.related_to is None


# ---------------------------------------------------------------------------
# _get_neighbors
# ---------------------------------------------------------------------------

class TestGetNeighbors:
    def test_out_edges_returned(self, graph):
        neighbors = {n: e for n, e, _ in _get_neighbors(["ТК_261"], graph)}
        assert "ТК_81" in neighbors
        assert "ТК_256" in neighbors

    def test_in_edges_returned(self, graph):
        # ТК_256 получает входящее ребро от ТК_261
        neighbors = {n: e for n, e, _ in _get_neighbors(["ТК_256"], graph)}
        assert "ТК_261" in neighbors

    def test_seed_excluded_from_result(self, graph):
        neighbors = {n: e for n, e, _ in _get_neighbors(["ТК_261"], graph)}
        assert "ТК_261" not in neighbors

    def test_allowed_types_filters(self, graph):
        neighbors = {n: e for n, e, _ in _get_neighbors(["ТК_261"], graph, allowed_types=STRUCTURAL_EDGE_TYPES)}
        # дополняет — не структурный, исключает — не структурный → пусто
        assert "ТК_81" not in neighbors
        assert "ТК_256" not in neighbors

    def test_node_not_in_graph(self, graph):
        result = _get_neighbors(["НЕСУЩЕСТВУЮЩАЯ_99"], graph)
        assert result == []

    def test_deduplication(self, graph):
        # ТК_261 и ТК_256 оба указывают на ТК_392 через ссылается_на (только ТК_256)
        # но ТК_261 дополняет ТК_256, а ТК_256 ссылается_на ТК_392
        neighbors = [n for n, _, _ in _get_neighbors(["ТК_261", "ТК_256"], graph)]
        assert neighbors.count("ТК_392") <= 1

    def test_returns_3_tuples(self, graph):
        result = _get_neighbors(["ТК_261"], graph)
        assert all(len(t) == 3 for t in result)

    def test_source_seed_in_third_element(self, graph):
        # ТК_261 --исключает--> ТК_81; source должен быть ТК_261
        result = _get_neighbors(["ТК_261"], graph)
        sources = {n: src for n, _, src in result}
        assert sources["ТК_81"] == "ТК_261"
        assert sources["ТК_256"] == "ТК_261"


# ---------------------------------------------------------------------------
# build_full_graph_context
# ---------------------------------------------------------------------------

class TestBuildFullGraph:
    def test_seeds_included(self, graph, corpus):
        chunks = build_full_graph_context(["ТК_261"], graph, corpus)
        norm_ids = [c.norm_id for c in chunks]
        assert "ТК_261" in norm_ids

    def test_all_neighbors_included(self, graph, corpus):
        chunks = build_full_graph_context(["ТК_261"], graph, corpus)
        norm_ids = [c.norm_id for c in chunks]
        assert "ТК_81" in norm_ids   # исключает
        assert "ТК_256" in norm_ids  # дополняет

    def test_missing_corpus_entry_skipped(self, graph, corpus):
        # ТК_999 есть в графе, но не в корпусе
        graph.add_edge("ТК_261", "ТК_999", edge_type="ссылается_на")
        chunks = build_full_graph_context(["ТК_261"], graph, corpus)
        assert all(c.norm_id != "ТК_999" for c in chunks)


# ---------------------------------------------------------------------------
# build_structural_only_context
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# build_doctrinal_only_context
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# build_structural_only_edges_context  (P2-0.3)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# build_doctrinal_only_edges_context  (P2-0.3)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# build_full_graph_edges_context  (P2-0.1)
# ---------------------------------------------------------------------------

class TestBuildFullGraphEdges:
    def test_same_norm_ids_as_full_graph(self, graph, corpus):
        chunks_edges = build_full_graph_edges_context(["ТК_261"], graph, corpus)
        chunks_full  = build_full_graph_context(["ТК_261"], graph, corpus)
        assert {c.norm_id for c in chunks_edges} == {c.norm_id for c in chunks_full}

    def test_neighbor_chunks_have_edge_type(self, graph, corpus):
        chunks = build_full_graph_edges_context(["ТК_261"], graph, corpus)
        neighbor_chunks = [c for c in chunks if c.norm_id != "ТК_261"]
        assert all(c.edge_type is not None for c in neighbor_chunks)

    def test_seed_chunk_edge_type_is_none(self, graph, corpus):
        chunks = build_full_graph_edges_context(["ТК_261"], graph, corpus)
        seed = next(c for c in chunks if c.norm_id == "ТК_261")
        assert seed.edge_type is None

    def test_neighbor_chunks_have_related_to(self, graph, corpus):
        chunks = build_full_graph_edges_context(["ТК_261"], graph, corpus)
        neighbor_chunks = [c for c in chunks if c.norm_id != "ТК_261"]
        assert all(c.related_to is not None for c in neighbor_chunks)

    def test_related_to_is_seed_norm(self, graph, corpus):
        chunks = build_full_graph_edges_context(["ТК_261"], graph, corpus)
        chunk_81 = next(c for c in chunks if c.norm_id == "ТК_81")
        assert chunk_81.related_to == "ТК_261"

    def test_valid_from_populated(self, graph, corpus_with_dates):
        chunks = build_full_graph_edges_context(["ТК_261"], graph, corpus_with_dates)
        seed = next(c for c in chunks if c.norm_id == "ТК_261")
        assert seed.valid_from == "01.02.2002"

    def test_valid_from_none_when_absent(self, graph, corpus_with_dates):
        chunks = build_full_graph_edges_context(["ТК_261"], graph, corpus_with_dates)
        chunk_256 = next(c for c in chunks if c.norm_id == "ТК_256")
        assert chunk_256.valid_from is None

    def test_is_repealed_populated(self, graph, corpus_with_dates):
        chunks = build_full_graph_edges_context(["ТК_261"], graph, corpus_with_dates)
        chunk_81 = next(c for c in chunks if c.norm_id == "ТК_81")
        assert chunk_81.is_repealed is True

    def test_is_repealed_false_not_none(self, graph, corpus_with_dates):
        chunks = build_full_graph_edges_context(["ТК_261"], graph, corpus_with_dates)
        seed = next(c for c in chunks if c.norm_id == "ТК_261")
        assert seed.is_repealed is False

    def test_seed_related_to_is_none(self, graph, corpus):
        chunks = build_full_graph_edges_context(["ТК_261"], graph, corpus)
        seed = next(c for c in chunks if c.norm_id == "ТК_261")
        assert seed.related_to is None


# ---------------------------------------------------------------------------
# _cosine_similarity
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# build_flat_rag_context
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# build_flat_rag_seeded_context  (P2-0.2)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# EmbedderProtocol
# ---------------------------------------------------------------------------
