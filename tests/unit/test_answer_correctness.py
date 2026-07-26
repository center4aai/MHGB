"""
Тесты для src/mhgb/eval/answer_correctness.py.
LLM не вызывается — используется mock-судья.
"""

import pytest
from mhgb.eval.answer_correctness import (
    VALID_SCORES,
    AnswerCorrectnessResult,
    _parse_score,
    evaluate_answer_correctness,
)
from mhgb.eval.step_correctness import EMPTY_RESPONSE_MARKER


# ---------------------------------------------------------------------------
# Mock-судья (переиспользует тот же интерфейс что и в test_step_correctness)
# ---------------------------------------------------------------------------

class MockJudge:
    def __init__(self, response: str):
        self.response = response
        self.calls: list[tuple[str, str]] = []

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return self.response


def _meta(**kw) -> dict:
    base = {"type": "rule_selection", "mode": "closed", "hop_group": "shallow",
            "question": "Законно ли увольнение?"}
    base.update(kw)
    return base


# ---------------------------------------------------------------------------
# _parse_score
# ---------------------------------------------------------------------------

class TestParseScore:
    def test_score_one(self):
        assert _parse_score("Score: 1\nОбоснование: верно.") == 1.0

    def test_score_zero(self):
        assert _parse_score("Score: 0\nОбоснование: неверно.") == 0.0

    def test_score_0_33(self):
        assert _parse_score("Score: 0.33\nОбоснование: частично.") == pytest.approx(0.33)

    def test_score_0_67(self):
        assert _parse_score("Score: 0.67\nОбоснование: почти верно.") == pytest.approx(0.67)

    def test_bare_value_fallback(self):
        assert _parse_score("Оценка 0.67 — почти верно") == pytest.approx(0.67)

    def test_label_takes_priority_over_bare(self):
        # В тексте есть и "0.33" и "Score: 1" — приоритет у Score:
        assert _parse_score("упоминается 0.33, Score: 1") == 1.0

    def test_no_score_returns_zero(self):
        assert _parse_score("Нет ничего похожего на оценку.") == 0.0

    def test_empty_returns_zero(self):
        assert _parse_score("") == 0.0

    def test_case_insensitive_label(self):
        assert _parse_score("score: 0.67") == pytest.approx(0.67)

    def test_returns_float(self):
        assert isinstance(_parse_score("Score: 1"), float)

    def test_all_valid_scores_parsed(self):
        for v in [0, 0.33, 0.67, 1]:
            assert _parse_score(f"Score: {v}") == pytest.approx(v)


# ---------------------------------------------------------------------------
# evaluate_answer_correctness
# ---------------------------------------------------------------------------

class TestEvaluateAnswerCorrectness:

    def test_perfect_answer(self):
        judge = MockJudge("Score: 1\nОбоснование: верно.")
        result = evaluate_answer_correctness("незаконно", "Увольнение незаконно.", _meta(), judge)
        assert result["score"] == pytest.approx(1.0)

    def test_zero_score(self):
        judge = MockJudge("Score: 0\nОбоснование: противоречит эталону.")
        result = evaluate_answer_correctness("незаконно", "Увольнение законно.", _meta(), judge)
        assert result["score"] == pytest.approx(0.0)

    def test_partial_score_0_33(self):
        judge = MockJudge("Score: 0.33\nОбоснование: есть существенная ошибка.")
        result = evaluate_answer_correctness("незаконно", "компенсация неверная", _meta(), judge)
        assert result["score"] == pytest.approx(0.33)

    def test_partial_score_0_67(self):
        judge = MockJudge("Score: 0.67\nОбоснование: мелкий пробел.")
        result = evaluate_answer_correctness("незаконно", "незаконно, срок не указан", _meta(), judge)
        assert result["score"] == pytest.approx(0.67)

    def test_explanation_returned(self):
        response = "Score: 1\nОбоснование: всё правильно."
        judge = MockJudge(response)
        result = evaluate_answer_correctness("а", "б", _meta(), judge)
        assert result["explanation"] == response

    def test_result_keys(self):
        judge = MockJudge("Score: 1")
        result = evaluate_answer_correctness("а", "б", _meta(), judge)
        assert set(result.keys()) == {"score", "explanation"}

    def test_judge_called_once(self):
        judge = MockJudge("Score: 1")
        evaluate_answer_correctness("а", "б", _meta(), judge)
        assert len(judge.calls) == 1

    def test_gold_answer_in_user_prompt(self):
        judge = MockJudge("Score: 1")
        evaluate_answer_correctness("эталонный_ответ_XYZ", "б", _meta(), judge)
        _, user_prompt = judge.calls[0]
        assert "эталонный_ответ_XYZ" in user_prompt

    def test_model_answer_in_user_prompt(self):
        judge = MockJudge("Score: 1")
        evaluate_answer_correctness("а", "уникальный_ответ_модели_ABC", _meta(), judge)
        _, user_prompt = judge.calls[0]
        assert "уникальный_ответ_модели_ABC" in user_prompt

    def test_task_type_in_user_prompt(self):
        judge = MockJudge("Score: 1")
        evaluate_answer_correctness("а", "б", _meta(type="temporal_validity"), judge)
        _, user_prompt = judge.calls[0]
        assert "temporal_validity" in user_prompt

    def test_question_in_user_prompt_when_provided(self):
        judge = MockJudge("Score: 1")
        evaluate_answer_correctness("а", "б", _meta(question="Вопрос о сроке?"), judge)
        _, user_prompt = judge.calls[0]
        assert "Вопрос о сроке?" in user_prompt

    def test_no_question_in_meta_no_error(self):
        judge = MockJudge("Score: 1")
        result = evaluate_answer_correctness("а", "б", {"type": "rule_selection"}, judge)
        assert "score" in result

    def test_system_prompt_contains_scale(self):
        judge = MockJudge("Score: 1")
        evaluate_answer_correctness("а", "б", _meta(), judge)
        system_prompt, _ = judge.calls[0]
        assert "0.33" in system_prompt
        assert "0.67" in system_prompt

    def test_score_in_valid_scores(self):
        for raw in ["Score: 0", "Score: 0.33", "Score: 0.67", "Score: 1"]:
            judge = MockJudge(raw)
            result = evaluate_answer_correctness("а", "б", _meta(), judge)
            assert result["score"] in VALID_SCORES


# ---------------------------------------------------------------------------
# VALID_SCORES
# ---------------------------------------------------------------------------

class TestValidScores:
    def test_contains_four_values(self):
        assert len(VALID_SCORES) == 4

    def test_values(self):
        assert 0.0 in VALID_SCORES
        assert 1.0 in VALID_SCORES
        assert 0.33 in VALID_SCORES
        assert 0.67 in VALID_SCORES


# ---------------------------------------------------------------------------
# Фикс B — short-circuit на пустом ответе (НЕ вызывает судью)
# ---------------------------------------------------------------------------

class _NeverCalledJudge:
    """Падает, если его вызвали — проверка, что гейт срабатывает ДО судьи."""
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        raise AssertionError("судья НЕ должен вызываться на пустом ответе")


class TestEmptyResponseShortCircuit:
    @pytest.mark.parametrize("empty", ["", "   ", "\n\t ", None])
    def test_empty_zeroed_without_judge(self, empty):
        res = evaluate_answer_correctness("Незаконно", empty, _meta(), _NeverCalledJudge())
        assert res["score"] == 0.0
        assert EMPTY_RESPONSE_MARKER in res["explanation"]

    def test_valid_short_answer_calls_judge(self):
        judge = MockJudge("Score: 1\nОбоснование: верно.")
        res = evaluate_answer_correctness("Незаконно", "Незаконно.", _meta(), judge)
        assert len(judge.calls) == 1
        assert res["score"] == 1.0
        assert EMPTY_RESPONSE_MARKER not in res["explanation"]
