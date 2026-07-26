"""
Smoke-тесты для src/mhgb/build_graph.py.
LLM не вызывается (llm_classify=False).
"""

import pytest
import networkx as nx
from mhgb.build_graph import (
    extract_article_nums,
    extract_explicit_refs,
    build_graph,
    save_graph,
    load_graph,
)


class TestExtractArticleNums:
    def test_single_number(self):
        assert extract_article_nums("261") == ["261"]

    def test_two_numbers_with_i(self):
        result = extract_article_nums("81 и 82")
        assert "81" in result and "82" in result

    def test_comma_separated(self):
        result = extract_article_nums("81, 82, 83")
        assert len(result) == 3
        assert "81" in result and "82" in result and "83" in result

    def test_dotted_number(self):
        assert extract_article_nums("261.1") == ["261.1"]

    def test_empty_string(self):
        assert extract_article_nums("") == []

    def test_no_numbers(self):
        assert extract_article_nums("никаких цифр здесь нет") == []

    def test_range_with_dash(self):
        result = extract_article_nums("81-83")
        assert "81" in result and "83" in result


class TestExtractExplicitRefs:
    def test_finds_reference_same_law(self):
        article = {
            "id": "ТК_261",
            "law_short": "ТК",
            "paragraphs": [
                "применяется в соответствии со статьей 256 настоящего кодекса"
            ],
        }
        edges = extract_explicit_refs(article)
        targets = [t for _, t in edges]
        assert "ТК_256" in targets

    def test_no_self_reference(self):
        article = {
            "id": "ТК_261",
            "law_short": "ТК",
            "paragraphs": ["статья 261 настоящего кодекса применяется к трудовым спорам"],
        }
        edges = extract_explicit_refs(article)
        assert ("ТК_261", "ТК_261") not in edges

    def test_empty_paragraphs(self):
        article = {"id": "ТК_81", "law_short": "ТК", "paragraphs": []}
        assert extract_explicit_refs(article) == []

    def test_no_refs_plain_text(self):
        article = {
            "id": "ТК_81",
            "law_short": "ТК",
            "paragraphs": ["Трудовой договор может быть расторгнут работодателем."],
        }
        assert extract_explicit_refs(article) == []

    def test_multiple_refs_in_one_paragraph(self):
        article = {
            "id": "ТК_261",
            "law_short": "ТК",
            "paragraphs": ["статьями 256 и 257 настоящего кодекса"],
        }
        edges = extract_explicit_refs(article)
        targets = [t for _, t in edges]
        assert "ТК_256" in targets
        assert "ТК_257" in targets

    def test_yo_normalization(self):
        # ё в "статьёй" нормализуется к е — ребро должно найтись
        article = {
            "id": "ТК_261",
            "law_short": "ТК",
            "paragraphs": ["согласно статьёй 256 настоящего кодекса"],
        }
        edges = extract_explicit_refs(article)
        targets = [t for _, t in edges]
        assert "ТК_256" in targets


class TestBuildGraph:
    def test_all_nodes_added(self, tmp_corpus):
        articles = list(tmp_corpus.values())
        G = build_graph(articles, llm_classify=False)
        for art in articles:
            assert art["id"] in G.nodes

    def test_node_attributes(self, tmp_corpus):
        articles = list(tmp_corpus.values())
        G = build_graph(articles, llm_classify=False)
        node = G.nodes["ТК_261"]
        assert node["law"] == "ТК"
        assert node["article"] == "261"

    def test_regex_edge_created(self, tmp_corpus):
        # ТК_261 содержит ссылку на ст.256 — ребро должно появиться
        articles = list(tmp_corpus.values())
        G = build_graph(articles, llm_classify=False)
        assert G.has_edge("ТК_261", "ТК_256")

    def test_no_llm_called_when_disabled(self, tmp_corpus, monkeypatch):
        import mhgb.build_graph as bg
        called = []
        monkeypatch.setattr(bg, "classify_edge_llm", lambda *a, **kw: called.append(1))
        build_graph(list(tmp_corpus.values()), llm_classify=False)
        assert len(called) == 0

    def test_is_directed_graph(self, tmp_corpus):
        G = build_graph(list(tmp_corpus.values()), llm_classify=False)
        assert isinstance(G, nx.DiGraph)


class TestSaveLoadGraph:
    def test_roundtrip_nodes(self, tmp_graph, tmp_path):
        path = tmp_path / "graph_test.json"
        save_graph(tmp_graph, path)
        G2 = load_graph(path)
        assert tmp_graph.number_of_nodes() == G2.number_of_nodes()

    def test_roundtrip_edges(self, tmp_graph, tmp_path):
        path = tmp_path / "graph_test.json"
        save_graph(tmp_graph, path)
        G2 = load_graph(path)
        assert tmp_graph.number_of_edges() == G2.number_of_edges()

    def test_roundtrip_edge_types(self, tmp_graph, tmp_path):
        path = tmp_path / "graph_test.json"
        save_graph(tmp_graph, path)
        G2 = load_graph(path)
        assert G2["ТК_261"]["ТК_256"]["edge_type"] == "применяется_к"
        assert G2["ТК_261"]["ТК_81"]["edge_type"] == "исключает"
