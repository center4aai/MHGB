"""
Тесты для src/mhgb/eval/step_correctness.py.
LLM не вызывается — используется mock-судья.
"""

import pytest
from mhgb.eval.step_correctness import (
    EMPTY_RESPONSE_MARKER,
    LLMJudge,
    StepCorrectnessResult,
    _is_empty_response,
    _parse_score,
    evaluate_step_correctness,
)


# ---------------------------------------------------------------------------
# Mock-судья
# ---------------------------------------------------------------------------

class MockJudge:
    """Судья с предсказуемыми ответами для тестов."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return self._responses.pop(0)


def _step(n: int, norm_id: str | None = "ТК_261", reasoning: str = "Запрет") -> dict:
    return {"step": n, "norm_id": norm_id, "reasoning": reasoning, "is_critical": True}


# ---------------------------------------------------------------------------
# _parse_score
# ---------------------------------------------------------------------------

class TestParseScore:
    def test_score_one(self):
        assert _parse_score("Score: 1\nОбоснование: верно.") == 1.0

    def test_score_zero(self):
        assert _parse_score("Score: 0\nОбоснование: не применена.") == 0.0

    def test_score_half(self):
        assert _parse_score("Score: 0.5\nОбоснование: частично.") == 0.5

    def test_score_inline_without_label(self):
        assert _parse_score("Ответ верен. 1") == 1.0

    def test_score_half_inline(self):
        assert _parse_score("Частично правильно: 0.5") == 0.5

    def test_no_score_returns_zero(self):
        assert _parse_score("Нет оценки в тексте.") == 0.0

    def test_empty_returns_zero(self):
        assert _parse_score("") == 0.0

    def test_prefers_first_match(self):
        # Первое найденное число — 0.5
        assert _parse_score("Score: 0.5. Итого: 1") == 0.5

    def test_returns_float(self):
        result = _parse_score("Score: 1")
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# evaluate_step_correctness — основные сценарии
# ---------------------------------------------------------------------------

class TestEvaluateStepCorrectness:

    def test_all_correct_steps(self):
        gold_chain = [_step(1, "ТК_261"), _step(2, "ТК_256")]
        judge = MockJudge(["Score: 1\nОбоснование: верно.", "Score: 1\nОбоснование: верно."])
        result = evaluate_step_correctness(gold_chain, "ответ модели", judge)
        assert result["step_correctness"] == pytest.approx(1.0)
        assert result["step_scores"] == [1.0, 1.0]

    def test_mixed_scores(self):
        gold_chain = [_step(1, "ТК_261"), _step(2, "ТК_256"), _step(3, "ТК_81")]
        judge = MockJudge(["Score: 1", "Score: 0.5", "Score: 0"])
        result = evaluate_step_correctness(gold_chain, "текст", judge)
        assert result["step_scores"] == [1.0, 0.5, 0.0]
        assert result["step_correctness"] == pytest.approx((1.0 + 0.5 + 0.0) / 3)

    def test_all_zero_scores(self):
        gold_chain = [_step(1, "ТК_261"), _step(2, "ТК_256")]
        judge = MockJudge(["Score: 0", "Score: 0"])
        result = evaluate_step_correctness(gold_chain, "пустой ответ", judge)
        assert result["step_correctness"] == pytest.approx(0.0)

    def test_single_step(self):
        gold_chain = [_step(1, "ТК_261")]
        judge = MockJudge(["Score: 0.5\nОбоснование: частично."])
        result = evaluate_step_correctness(gold_chain, "ответ", judge)
        assert result["step_correctness"] == pytest.approx(0.5)
        assert len(result["step_scores"]) == 1

    def test_empty_gold_chain_returns_zero(self):
        result = evaluate_step_correctness([], "ответ", MockJudge([]))
        assert result["step_correctness"] == pytest.approx(0.0)
        assert result["step_scores"] == []
        assert result["judge_explanations"] == []

    def test_none_norm_id_steps_skipped(self):
        gold_chain = [
            _step(1, "ТК_261"),
            _step(2, None),         # заключение — пропускается
            _step(3, "ТК_256"),
        ]
        judge = MockJudge(["Score: 1", "Score: 1"])
        result = evaluate_step_correctness(gold_chain, "ответ", judge)
        assert len(result["step_scores"]) == 2
        assert len(judge.calls) == 2

    def test_all_steps_none_returns_zero(self):
        gold_chain = [_step(1, None), _step(2, None)]
        result = evaluate_step_correctness(gold_chain, "ответ", MockJudge([]))
        assert result["step_correctness"] == pytest.approx(0.0)
        assert result["step_scores"] == []

    def test_explanations_match_judge_responses(self):
        gold_chain = [_step(1, "ТК_261"), _step(2, "ТК_256")]
        responses = ["Score: 1\nОбоснование: норма применена.", "Score: 0\nОбоснование: пропущена."]
        judge = MockJudge(responses[:])
        result = evaluate_step_correctness(gold_chain, "ответ", judge)
        assert result["judge_explanations"] == responses

    def test_result_keys(self):
        judge = MockJudge(["Score: 1"])
        result = evaluate_step_correctness([_step(1)], "ответ", judge)
        assert set(result.keys()) == {"step_scores", "step_correctness", "judge_explanations"}

    def test_step_correctness_in_0_1(self):
        gold_chain = [_step(i) for i in range(1, 5)]
        judge = MockJudge(["Score: 1", "Score: 0.5", "Score: 0", "Score: 1"])
        result = evaluate_step_correctness(gold_chain, "ответ", judge)
        assert 0.0 <= result["step_correctness"] <= 1.0

    def test_judge_called_once_per_evaluable_step(self):
        gold_chain = [_step(1, "ТК_261"), _step(2, None), _step(3, "ТК_256")]
        judge = MockJudge(["Score: 1", "Score: 1"])
        evaluate_step_correctness(gold_chain, "ответ", judge)
        assert len(judge.calls) == 2

    def test_user_prompt_contains_norm_id(self):
        gold_chain = [_step(1, "ТК_392", "срок давности")]
        judge = MockJudge(["Score: 1"])
        evaluate_step_correctness(gold_chain, "ответ модели", judge)
        _, user_prompt = judge.calls[0]
        assert "ТК_392" in user_prompt

    def test_user_prompt_contains_model_response(self):
        gold_chain = [_step(1, "ТК_261")]
        judge = MockJudge(["Score: 1"])
        evaluate_step_correctness(gold_chain, "уникальный_ответ_модели", judge)
        _, user_prompt = judge.calls[0]
        assert "уникальный_ответ_модели" in user_prompt

    def test_system_prompt_passed_to_judge(self):
        gold_chain = [_step(1, "ТК_261")]
        judge = MockJudge(["Score: 1"])
        evaluate_step_correctness(gold_chain, "ответ", judge)
        system_prompt, _ = judge.calls[0]
        assert len(system_prompt) > 0


# ---------------------------------------------------------------------------
# LLMJudge Protocol
# ---------------------------------------------------------------------------

class TestLLMJudgeProtocol:
    def test_mock_satisfies_protocol(self):
        judge = MockJudge(["Score: 1"])
        assert isinstance(judge, LLMJudge)

    def test_class_without_complete_does_not_satisfy(self):
        class BadJudge:
            pass
        assert not isinstance(BadJudge(), LLMJudge)


# ---------------------------------------------------------------------------
# Фикс B — short-circuit на пустом ответе (НЕ вызывает судью)
# ---------------------------------------------------------------------------

class _NeverCalledJudge:
    """Падает, если его вызвали — проверка, что гейт срабатывает ДО судьи."""
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        raise AssertionError("судья НЕ должен вызываться на пустом ответе")


class TestEmptyResponseShortCircuit:
    def test_is_empty_response_predicate(self):
        assert _is_empty_response("")
        assert _is_empty_response("   \n\t ")
        assert _is_empty_response(None)
        assert not _is_empty_response("Незаконно.")
        assert not _is_empty_response("  x  ")

    @pytest.mark.parametrize("empty", ["", "   ", "\n\n\t ", None])
    def test_empty_zeroed_without_judge(self, empty):
        gold = [_step(1, "ТК_261"), _step(2, "ТК_256")]
        res = evaluate_step_correctness(gold, empty, _NeverCalledJudge())
        assert res["step_correctness"] == 0.0
        assert res["step_scores"] == [0.0, 0.0]
        assert all(EMPTY_RESPONSE_MARKER in e for e in res["judge_explanations"])

    def test_valid_short_answer_calls_judge(self):
        # непустой короткий ответ — судья ВЫЗВАН, не зануляется
        judge = MockJudge(["Score: 1"])
        res = evaluate_step_correctness([_step(1, "ТК_261")], "Незаконно.", judge)
        assert len(judge.calls) == 1
        assert res["step_correctness"] == 1.0
        assert EMPTY_RESPONSE_MARKER not in res["judge_explanations"][0]

    def test_empty_with_no_evaluable_steps(self):
        # нет шагов с norm_id + пустой ответ → пустые списки, 0.0, без падения
        res = evaluate_step_correctness([], "", _NeverCalledJudge())
        assert res["step_correctness"] == 0.0
        assert res["step_scores"] == []
