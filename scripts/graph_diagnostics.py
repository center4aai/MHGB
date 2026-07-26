"""
Шаг 2.1: Диагностика текущего графа.

Вывод:
  - Распределение нод по кодексам
  - Распределение рёбер по типам
  - Внутри-кодексные vs межкодексные рёбра
  - Семантические рёбра по кодексам
  - Изолированные ноды и компоненты связности

Сохраняет: data/graph_diagnostics_v1.json
Запуск:    uv run python scripts/graph_diagnostics.py
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
from networkx.readwrite import json_graph

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mhgb.config import cfg
from mhgb.logging_setup import logger

SEMANTIC_TYPES = {"исключает", "дополняет", "приоритет", "применяется_к"}


def load_graph(path: Path) -> nx.DiGraph:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return json_graph.node_link_graph(data)


def run_diagnostics(G: nx.DiGraph) -> dict:
    # --- Ноды по кодексам ---
    nodes_by_law: Counter = Counter()
    for _, attrs in G.nodes(data=True):
        nodes_by_law[attrs.get("law", "?")] += 1

    # --- Рёбра по типам ---
    edges_by_type: Counter = Counter()
    for _, _, attrs in G.edges(data=True):
        edges_by_type[attrs.get("edge_type", "?")] += 1

    # --- Внутри- vs межкодексные ---
    intra = 0
    inter = 0
    inter_pairs: Counter = Counter()
    semantic_by_law: defaultdict = defaultdict(Counter)

    for src, dst, attrs in G.edges(data=True):
        src_law = G.nodes[src].get("law", "?")
        dst_law = G.nodes[dst].get("law", "?")
        etype = attrs.get("edge_type", "?")

        if src_law == dst_law:
            intra += 1
        else:
            inter += 1
            pair = tuple(sorted([src_law, dst_law]))
            inter_pairs[f"{pair[0]}↔{pair[1]}"] += 1

        if etype in SEMANTIC_TYPES:
            semantic_by_law[src_law][etype] += 1

    # --- Изолированные ноды ---
    isolated = list(nx.isolates(G))

    # --- Компоненты связности (слабые) ---
    wcc = list(nx.weakly_connected_components(G))
    wcc_sizes = sorted([len(c) for c in wcc], reverse=True)

    report = {
        "nodes_total": G.number_of_nodes(),
        "edges_total": G.number_of_edges(),
        "nodes_by_law": dict(nodes_by_law.most_common()),
        "edges_by_type": dict(edges_by_type.most_common()),
        "intra_codex_edges": intra,
        "inter_codex_edges": inter,
        "inter_codex_pairs": dict(inter_pairs.most_common(20)),
        "semantic_edges_by_law": {
            law: dict(ctypes) for law, ctypes in sorted(semantic_by_law.items())
        },
        "isolated_nodes_count": len(isolated),
        "isolated_nodes_sample": isolated[:20],
        "weakly_connected_components_count": len(wcc),
        "wcc_sizes_top10": wcc_sizes[:10],
    }
    return report


def log_report(r: dict) -> None:
    logger.info("=== Диагностика графа ===")
    logger.info("Нод: {}  |  Рёбер: {}", r["nodes_total"], r["edges_total"])

    logger.info("--- Ноды по кодексам ---")
    for law, cnt in r["nodes_by_law"].items():
        logger.info("  {:6s}  {:4d}", law, cnt)

    logger.info("--- Рёбра по типам ---")
    for etype, cnt in r["edges_by_type"].items():
        logger.info("  {:20s}  {:4d}", etype, cnt)

    logger.info(
        "--- Внутри-кодексные: {}  |  Межкодексные: {} ---",
        r["intra_codex_edges"], r["inter_codex_edges"],
    )
    if r["inter_codex_pairs"]:
        logger.info("  Топ межкодексных пар:")
        for pair, cnt in list(r["inter_codex_pairs"].items())[:10]:
            logger.info("    {}  {}", pair, cnt)

    logger.info("--- Семантические рёбра по кодексам ---")
    if r["semantic_edges_by_law"]:
        for law, ctypes in r["semantic_edges_by_law"].items():
            logger.info("  {:6s}  {}", law, dict(ctypes))
    else:
        logger.info("  (нет семантических рёбер)")

    logger.info(
        "--- Изолированных нод: {}  |  Слабых компонент: {} ---",
        r["isolated_nodes_count"], r["weakly_connected_components_count"],
    )
    logger.info("  Топ-10 размеров компонент: {}", r["wcc_sizes_top10"])


def main() -> None:
    graph_path = cfg.data_dir / "graph.json"
    if not graph_path.exists():
        logger.error("graph.json не найден: {}", graph_path)
        sys.exit(1)

    logger.info("Загружаю граф из {}", graph_path)
    G = load_graph(graph_path)

    report = run_diagnostics(G)
    log_report(report)

    out_path = cfg.data_dir / "graph_diagnostics_v1.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info("Отчёт сохранён: {}", out_path)


if __name__ == "__main__":
    main()
