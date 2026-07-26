"""Тесты P2-1.3: метрики импорта экспертной валидации (Fleiss/precision/alt/Cohen).

Скрипт лежит в scripts/, поэтому грузим его через importlib.
Тестируются ЧИСТЫЕ функции на синтетике (xlsx не нужен).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "import_expert_validation", _ROOT / "scripts" / "import_expert_validation.py")
imp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(imp)


def _mk(tasks: dict, edges: dict) -> dict:
    return {"tasks": tasks, "edges": edges}


def _crit(**kw):
    """Хелпер: задача с критериями (по умолчанию все True) + alt."""
    base = {k: True for k in imp.CRIT_KEYS}
    base["alt"] = kw.pop("alt", "нет")
    base.update(kw)
    return base


# ---------------------------------------------------------------------------
# Fleiss κ по задачам
# ---------------------------------------------------------------------------

def test_task_fleiss_unanimous_is_one():
    a = _mk({"T01": _crit(), "T02": _crit()}, {})
    annots = [a, a, a]
    r = imp.compute_task_fleiss(annots)
    # все критерии единогласны → κ=1.0 по каждому
    for k in imp.CRIT_KEYS:
        assert r["per_criterion"][k] == pytest.approx(1.0)
    assert r["aggregate"] == pytest.approx(1.0)
    assert r["n_items_pooled"] == 2 * len(imp.CRIT_KEYS)


def test_task_fleiss_skips_unfilled_rows():
    a = _mk({"T01": _crit(К1=True), "T02": _crit(К1=None)}, {})
    b = _mk({"T01": _crit(К1=True), "T02": _crit(К1=True)}, {})
    c = _mk({"T01": _crit(К1=True), "T02": _crit(К1=True)}, {})
    r = imp.compute_task_fleiss([a, b, c])
    # T02 по К1 имеет None у одного юриста → строка отброшена для К1
    assert r["per_criterion"]["К1"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Precision рёбер + Wilson CI
# ---------------------------------------------------------------------------

def test_edge_precision_majority_and_groups():
    keymap = {"edges": {
        "E01": {"group": "doctrinal"}, "E02": {"group": "doctrinal"},
        "E03": {"group": "doctrinal"}, "E04": {"group": "structural"}}}
    # E01,E02 единогласно да; E03 2нет/1да → majority нет; E04 да
    A = _mk({}, {"E01": {"correct": True}, "E02": {"correct": True},
                 "E03": {"correct": False}, "E04": {"correct": True}})
    B = _mk({}, {"E01": {"correct": True}, "E02": {"correct": True},
                 "E03": {"correct": False}, "E04": {"correct": True}})
    C = _mk({}, {"E01": {"correct": True}, "E02": {"correct": True},
                 "E03": {"correct": True}, "E04": {"correct": True}})
    ep = imp.compute_edge_precision([A, B, C], keymap)
    assert ep["doctrinal"]["correct"] == 2 and ep["doctrinal"]["n"] == 3
    assert ep["doctrinal"]["precision"] == pytest.approx(2 / 3)
    assert ep["structural"]["precision"] == pytest.approx(1.0)
    assert ep["all"]["precision"] == pytest.approx(0.75)
    lo, hi = ep["doctrinal"]["wilson_ci"]
    assert 0.0 <= lo < ep["doctrinal"]["precision"] < hi <= 1.0


# ---------------------------------------------------------------------------
# Доля альтернативных цепочек
# ---------------------------------------------------------------------------

def test_alternative_fraction_by_type():
    keymap = {"tasks": {"T01": "r1", "T02": "r2", "T03": "r3", "T04": "r4"}}
    type_by_id = {"r1": "conflict_resolution", "r2": "conflict_resolution",
                  "r3": "temporal_validity", "r4": "temporal_validity"}
    A = _mk({"T01": _crit(alt="нет"), "T02": _crit(alt="существенная"),
             "T03": _crit(alt="нет"), "T04": _crit(alt="незначительная")}, {})
    B = _mk({"T01": _crit(alt="нет"), "T02": _crit(alt="существенная"),
             "T03": _crit(alt="нет"), "T04": _crit(alt="нет")}, {})
    C = _mk({"T01": _crit(alt="нет"), "T02": _crit(alt="существенная"),
             "T03": _crit(alt="нет"), "T04": _crit(alt="нет")}, {})
    af = imp.compute_alternative_fraction([A, B, C], keymap, type_by_id)
    assert af["conflict_resolution"]["fraction"] == pytest.approx(0.5)  # T02 нетривиально
    assert af["temporal_validity"]["fraction"] == pytest.approx(0.0)    # T04 majority нет
    assert af["all"]["fraction"] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Cohen κ: LLM vs консенсус юристов
# ---------------------------------------------------------------------------

def test_llm_vs_expert_cohen_perfect():
    keymap = {"tasks": {"T01": "r1", "T02": "r2"}}
    A = _mk({"T01": _crit(К2=False), "T02": _crit()}, {})
    annots = [A, A, A]
    llm = {
        "r1": {"checks": {v: (False if k == "К2" else True) for k, v in imp.CRIT_TO_LLM.items()}},
        "r2": {"checks": {v: True for v in imp.CRIT_TO_LLM.values()}},
    }
    ck = imp.compute_llm_vs_expert_cohen(annots, keymap, llm)
    # консенсус юристов совпадает с LLM по всем (task,criterion) → κ=1.0
    assert ck["aggregate"] == pytest.approx(1.0)
