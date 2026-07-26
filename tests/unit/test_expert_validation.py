"""
Тесты для src/mhgb/validation/expert_validation.py.
"""

import pytest

from mhgb.validation.expert_validation import (
    HOP_GROUPS,
    TASK_TYPES,
    compare_llm_vs_expert,
    compute_agreement,
    compute_cohens_kappa,
    select_stratified_sample,
)


# ---------------------------------------------------------------------------
# Вспомогательные объекты
# ---------------------------------------------------------------------------

def _task(task_type: str, hop_group: str, mode: str = "closed", task_id: str | None = None) -> dict:
    return {
        "id": task_id or f"{task_type}_{hop_group}_{mode}",
        "type": task_type,
        "hop_group": hop_group,
        "mode": mode,
        "branch_of_law": "трудовое",
    }


def _make_full_corpus(n_per_cell: int = 5) -> list[dict]:
    """Корпус с n_per_cell задачами в каждой ячейке матрицы (closed + open)."""
    tasks = []
    for tt in TASK_TYPES:
        for hg in HOP_GROUPS:
            for i in range(n_per_cell):
                tasks.append(_task(tt, hg, "closed", f"{tt}_{hg}_closed_{i}"))
                tasks.append(_task(tt, hg, "open",   f"{tt}_{hg}_open_{i}"))
    return tasks


# ---------------------------------------------------------------------------
# select_stratified_sample
# ---------------------------------------------------------------------------

class TestSelectStratifiedSample:

    def test_empty_tasks_returns_empty(self):
        assert select_stratified_sample([]) == []

    def test_returns_only_closed_mode(self):
        tasks = _make_full_corpus()
        sample = select_stratified_sample(tasks, n_per_cell=2)
        assert all(t["mode"] == "closed" for t in sample)

    def test_respects_n_per_cell(self):
        tasks = _make_full_corpus(n_per_cell=10)
        sample = select_stratified_sample(tasks, n_per_cell=3)
        # 4 типа × 3 hop_groups × 3 = 36 максимум
        assert len(sample) <= 4 * 3 * 3

    def test_all_cells_covered(self):
        tasks = _make_full_corpus(n_per_cell=5)
        sample = select_stratified_sample(tasks, n_per_cell=2)
        covered = {(t["type"], t["hop_group"]) for t in sample}
        expected = {(tt, hg) for tt in TASK_TYPES for hg in HOP_GROUPS}
        assert covered == expected

    def test_fewer_tasks_than_n_per_cell(self):
        tasks = [_task("rule_selection", "shallow", "closed", "t1")]
        sample = select_stratified_sample(tasks, n_per_cell=10)
        assert len(sample) == 1

    def test_reproducible_with_same_seed(self):
        tasks = _make_full_corpus()
        s1 = select_stratified_sample(tasks, n_per_cell=3, seed=7)
        s2 = select_stratified_sample(tasks, n_per_cell=3, seed=7)
        assert [t["id"] for t in s1] == [t["id"] for t in s2]

    def test_different_seeds_may_differ(self):
        tasks = _make_full_corpus(n_per_cell=10)
        s1 = select_stratified_sample(tasks, n_per_cell=5, seed=1)
        s2 = select_stratified_sample(tasks, n_per_cell=5, seed=99)
        assert [t["id"] for t in s1] != [t["id"] for t in s2]

    def test_no_open_tasks_in_sample(self):
        tasks = [_task("rule_selection", "medium", "open")]
        assert select_stratified_sample(tasks) == []


# ---------------------------------------------------------------------------
# compute_cohens_kappa
# ---------------------------------------------------------------------------

class TestCohenKappa:

    def test_perfect_agreement_returns_one(self):
        a = [True, False, True, True, False]
        assert compute_cohens_kappa(a, a) == pytest.approx(1.0)

    def test_complete_disagreement_negative(self):
        a = [True, True, False, False]
        b = [False, False, True, True]
        kappa = compute_cohens_kappa(a, b)
        assert kappa < 0.0

    def test_empty_lists_return_zero(self):
        assert compute_cohens_kappa([], []) == pytest.approx(0.0)

    def test_all_same_label_both(self):
        # все True у обоих → p_e = 1, возвращаем 0.0
        a = [True, True, True]
        assert compute_cohens_kappa(a, a) == pytest.approx(0.0)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="Длины"):
            compute_cohens_kappa([True, False], [True])

    def test_known_value(self):
        # 4 согласия из 5: p_o=0.8; p_e зависит от распределения
        a = [True, True, False, True, False]
        b = [True, True, False, False, False]
        kappa = compute_cohens_kappa(a, b)
        assert -1.0 <= kappa <= 1.0


# ---------------------------------------------------------------------------
# compute_agreement
# ---------------------------------------------------------------------------

class TestComputeAgreement:

    def test_perfect_agreement(self):
        a = [True, False, True]
        assert compute_agreement(a, a) == pytest.approx(1.0)

    def test_no_agreement(self):
        a = [True, False]
        b = [False, True]
        assert compute_agreement(a, b) == pytest.approx(0.0)

    def test_partial_agreement(self):
        a = [True, True, False, False]
        b = [True, False, False, True]
        assert compute_agreement(a, b) == pytest.approx(0.5)

    def test_empty_lists_return_zero(self):
        assert compute_agreement([], []) == pytest.approx(0.0)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="Длины"):
            compute_agreement([True], [True, False])

    def test_result_in_0_1(self):
        a = [True, False, True, False, True]
        b = [True, True, False, False, True]
        result = compute_agreement(a, b)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# compare_llm_vs_expert
# ---------------------------------------------------------------------------

class TestCompareLlmVsExpert:

    def _result(self, task_id: str, is_valid: bool) -> dict:
        return {"task_id": task_id, "is_valid": is_valid}

    def test_perfect_agreement(self):
        llm    = [self._result("t1", True), self._result("t2", False)]
        expert = [self._result("t1", True), self._result("t2", False)]
        cmp = compare_llm_vs_expert(llm, expert)
        assert cmp["agreement"] == pytest.approx(1.0)

    def test_total_is_intersection(self):
        llm    = [self._result("t1", True), self._result("t3", True)]
        expert = [self._result("t1", True), self._result("t2", False)]
        cmp = compare_llm_vs_expert(llm, expert)
        assert cmp["total"] == 1

    def test_counts_both_valid(self):
        llm    = [self._result("t1", True), self._result("t2", True)]
        expert = [self._result("t1", True), self._result("t2", False)]
        cmp = compare_llm_vs_expert(llm, expert)
        assert cmp["counts"]["both_valid"] == 1

    def test_counts_both_rejected(self):
        llm    = [self._result("t1", False), self._result("t2", True)]
        expert = [self._result("t1", False), self._result("t2", True)]
        cmp = compare_llm_vs_expert(llm, expert)
        assert cmp["counts"]["both_rejected"] == 1

    def test_empty_intersection(self):
        llm    = [self._result("t1", True)]
        expert = [self._result("t2", True)]
        cmp = compare_llm_vs_expert(llm, expert)
        assert cmp["total"] == 0
        assert cmp["agreement"] == pytest.approx(0.0)

    def test_result_has_required_keys(self):
        llm    = [self._result("t1", True)]
        expert = [self._result("t1", True)]
        cmp = compare_llm_vs_expert(llm, expert)
        assert set(cmp.keys()) == {"total", "agreement", "cohens_kappa", "counts"}
