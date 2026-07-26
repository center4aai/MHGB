"""
Norm Coverage metrics — Level 1 (deterministic).

Two coverage views:
  compute_list_coverage      — explicit article list from ПРИМЕНИМЫЕ СТАТЬИ section
  compute_reasoning_coverage — article mentions extracted from reasoning text

Both return precision / recall / F1 against task's gold_articles.

Additional consistency check:
  list_vs_reasoning_consistency — Jaccard between list articles and reasoning articles
"""

from __future__ import annotations

from typing import TypedDict

from mhgb.eval.response_parser import extract_norms_from_text


class CoverageResult(TypedDict):
    precision: float
    recall: float
    f1: float


def _prf(predicted: set[str], gold: set[str]) -> CoverageResult:
    """Precision / Recall / F1 for two sets of norm IDs."""
    if not gold:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    tp = len(predicted & gold)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(gold)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_list_coverage(
    predicted_articles: set[str],
    gold_articles: set[str],
) -> CoverageResult:
    """
    Coverage by the explicit article list the model reported in ПРИМЕНИМЫЕ СТАТЬИ.

    predicted_articles — norm IDs extracted from that section (e.g. {"ТК_261", "ТК_256"})
    gold_articles      — ground-truth norm IDs from the task
    """
    return _prf(predicted_articles, gold_articles)


def compute_reasoning_coverage(
    reasoning_text: str,
    gold_articles: set[str],
    corpus: dict,
) -> CoverageResult:
    """
    Coverage by article mentions found anywhere in the reasoning text.

    Uses extract_norms_from_text to find norm IDs; only norm IDs present in
    corpus are counted (hallucinates are filtered out automatically).
    """
    predicted = extract_norms_from_text(reasoning_text, corpus)
    return _prf(predicted, gold_articles)


def list_vs_reasoning_consistency(
    list_articles: set[str],
    reasoning_text: str,
    corpus: dict,
) -> float:
    """
    Jaccard similarity between the articles in the explicit list and the articles
    mentioned in the reasoning text.

    Value of 1.0 means the model cited exactly the same norms in both places.
    Low values indicate the model listed norms it never reasoned about (or vice versa).
    Returns 1.0 when both sets are empty (trivially consistent).
    """
    reasoning_articles = extract_norms_from_text(reasoning_text, corpus)
    union = list_articles | reasoning_articles
    if not union:
        return 1.0
    return len(list_articles & reasoning_articles) / len(union)
