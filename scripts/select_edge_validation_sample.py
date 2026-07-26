"""Стратифицированная выборка рёбер графа для валидации юристами (P2-1.1, D3.2/D3.7).

Две группы:
  • Доктринальные (~50, пропорционально): исключает / дополняет / приоритет —
    содержательные LLM-рёбра, их precision и валидирует панель (D3.7).
  • Структурные (~20, контрольная группа): ссылается_на / применяется_к —
    почти тривиальные (ссылается_на — regex), служат «якорем» согласия.

Seed фиксируется; результат → data/edge_validation_sample.jsonl + companion meta.
Реальные source/target сохраняются; анонимизация ID — на этапе экспорта (P2-1.2).

Запуск:
  uv run python scripts/select_edge_validation_sample.py            # 50 + 20
  uv run python scripts/select_edge_validation_sample.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GRAPH = ROOT / "data" / "graph.json"
CORPUS = ROOT / "data" / "corpus.jsonl"
OUT = ROOT / "data" / "edge_validation_sample.jsonl"

DOCTRINAL = ("исключает", "дополняет", "приоритет")
STRUCTURAL = ("ссылается_на", "применяется_к")


def _largest_remainder(counts: dict[str, int], total: int) -> dict[str, int]:
    """Пропорциональная разбивка `total` по `counts` с методом наибольшего остатка."""
    grand = sum(counts.values())
    if grand == 0:
        return {k: 0 for k in counts}
    raw = {k: v / grand * total for k, v in counts.items()}
    base = {k: int(x) for k, x in raw.items()}
    rem = total - sum(base.values())
    # раздаём остаток ячейкам с наибольшей дробной частью
    order = sorted(counts, key=lambda k: raw[k] - base[k], reverse=True)
    for k in order[:rem]:
        base[k] += 1
    return base


def _load_corpus() -> dict[str, dict]:
    corpus: dict[str, dict] = {}
    with CORPUS.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            # при дублях id берём запись с непустым текстом
            if r["id"] not in corpus or (r.get("text") and not corpus[r["id"]].get("text")):
                corpus[r["id"]] = r
    return corpus


def _enrich(edge: dict, corpus: dict, group: str, idx: int) -> dict:
    src, tgt = edge["source"], edge["target"]
    sc, tc = corpus.get(src, {}), corpus.get(tgt, {})
    return {
        "edge_idx": idx,
        "group": group,
        "edge_type": edge["edge_type"],
        "source": src,
        "target": tgt,
        "source_title": sc.get("title", ""),
        "source_text": sc.get("text", ""),
        "target_title": tc.get("title", ""),
        "target_text": tc.get("text", ""),
        "llm_explanation": edge.get("explanation", ""),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Выборка рёбер для валидации юристами")
    ap.add_argument("--n-doctrinal", type=int, default=50)
    ap.add_argument("--n-structural", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--dry-run", action="store_true", help="только показать план, не писать файл")
    args = ap.parse_args()

    graph = json.loads(GRAPH.read_text(encoding="utf-8"))
    edges = graph["edges"]
    by_type: dict[str, list] = {}
    for e in edges:
        by_type.setdefault(e["edge_type"], []).append(e)

    doc_counts = {t: len(by_type.get(t, [])) for t in DOCTRINAL}
    str_counts = {t: len(by_type.get(t, [])) for t in STRUCTURAL}
    doc_targets = _largest_remainder(doc_counts, args.n_doctrinal)
    str_targets = _largest_remainder(str_counts, args.n_structural)

    print("Доктринальные (всего в графе → выбрать):")
    for t in DOCTRINAL:
        print(f"  {t:12} {doc_counts[t]:5} → {doc_targets[t]}")
    print("Структурные (контроль):")
    for t in STRUCTURAL:
        print(f"  {t:12} {str_counts[t]:5} → {str_targets[t]}")
    print(f"Итого: {sum(doc_targets.values())} доктр + {sum(str_targets.values())} структ "
          f"= {sum(doc_targets.values()) + sum(str_targets.values())} рёбер  seed={args.seed}")

    if args.dry_run:
        print("\n[dry-run] файл не записан.")
        return

    rng = random.Random(args.seed)
    corpus = _load_corpus()
    selected: list[dict] = []
    idx = 1
    for group, targets in (("doctrinal", doc_targets), ("structural", str_targets)):
        for etype in (DOCTRINAL if group == "doctrinal" else STRUCTURAL):
            pool = by_type.get(etype, [])
            k = min(targets[etype], len(pool))
            for e in rng.sample(pool, k):
                selected.append(_enrich(e, corpus, group, idx))
                idx += 1

    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in selected:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # companion meta
    meta = {
        "seed": args.seed,
        "n_total": len(selected),
        "n_doctrinal": sum(1 for r in selected if r["group"] == "doctrinal"),
        "n_structural": sum(1 for r in selected if r["group"] == "structural"),
        "by_type": dict(Counter(r["edge_type"] for r in selected)),
        "graph_totals": {**doc_counts, **str_counts},
    }
    meta_path = out_path.with_name(out_path.stem + "_meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n✅ {len(selected)} рёбер → {out_path.relative_to(ROOT)}")
    print(f"   meta → {meta_path.relative_to(ROOT)}")
    print(f"   по типам: {meta['by_type']}")


if __name__ == "__main__":
    main()
