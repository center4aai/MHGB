"""
Тесты для src/mhgb/validation/llm_validator.py.
LLM не вызывается — используется mock-судья.
"""

import json
import pathlib

import pytest

from mhgb.validation.llm_validator import (
    CHECK_NAMES,
    _build_user_prompt,
    _parse_validation,
    run_validation,
    validate_task,
)


# ---------------------------------------------------------------------------
# Вспомогательные объекты
# ---------------------------------------------------------------------------

class MockJudge:
    """Судья с предсказуемыми ответами."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return self._responses.pop(0)


def _valid_json(**overrides) -> str:
    data: dict = {name: True for name in CHECK_NAMES}
    data["issues"] = []
    data["explanation"] = "Задача корректна."
    data.update(overrides)
    return json.dumps(data, ensure_ascii=False)


def _invalid_json(**overrides) -> str:
    data: dict = {name: False for name in CHECK_NAMES}
    data["issues"] = ["Фабула неполная"]
    data["explanation"] = "Задача некорректна."
    data.update(overrides)
    return json.dumps(data, ensure_ascii=False)


def _sample_task(**overrides) -> dict:
    base = {
        "id": "test_task_1",
        "type": "rule_selection",
        "mode": "closed",
        "hop_group": "shallow",
        "norm_ids": ["ТК_261", "ТК_256"],
        "fabula": "Сотрудница Иванова находится в декрете. Работодатель сократил её 15.02.2026.",
        "question": "Законно ли увольнение Ивановой?",
        "answer": "Нет, незаконно согласно ст. 261 ТК РФ.",
        "gold_chain": [
            {"step": 1, "norm_id": "ТК_261", "reasoning": "Запрет увольнения в декрете."},
            {"step": 2, "norm_id": "ТК_256", "reasoning": "Определяет период декрета."},
            {"step": 3, "norm_id": None, "conclusion": "Увольнение незаконно."},
        ],
        "context_chunks": None,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# _parse_validation
# ---------------------------------------------------------------------------

class TestParseValidation:

    def test_all_true_json(self):
        checks, issues, _ = _parse_validation(_valid_json())
        assert all(checks.values())
        assert issues == []

    def test_all_false_json(self):
        checks, issues, _ = _parse_validation(_invalid_json())
        assert not any(checks.values())
        assert "Фабула неполная" in issues

    def test_partial_checks(self):
        response = _valid_json(fabula_complete=False, no_logical_gaps=False)
        checks, _, _ = _parse_validation(response)
        assert checks["fabula_complete"] is False
        assert checks["no_logical_gaps"] is False
        assert checks["gold_chain_matches_norm_ids"] is True

    def test_has_all_check_names(self):
        checks, _, _ = _parse_validation(_valid_json())
        assert set(checks.keys()) == set(CHECK_NAMES)

    def test_issues_list_preserved(self):
        response = _invalid_json(issues=["Пробел в шаге 2", "Неполная фабула"])
        _, issues, _ = _parse_validation(response)
        assert len(issues) == 2
        assert "Пробел в шаге 2" in issues

    def test_explanation_preserved(self):
        response = _valid_json(explanation="Задача абсолютно корректна.")
        _, _, explanation = _parse_validation(response)
        assert "абсолютно корректна" in explanation

    def test_invalid_json_returns_all_false(self):
        checks, _, _ = _parse_validation("Это не JSON вообще")
        assert all(v is False for v in checks.values())

    def test_empty_string_returns_all_false(self):
        checks, _, _ = _parse_validation("")
        assert all(v is False for v in checks.values())

    def test_markdown_wrapped_json(self):
        inner = _valid_json()
        response = f"```json\n{inner}\n```"
        checks, _, _ = _parse_validation(response)
        assert all(checks.values())

    def test_regex_fallback_partial(self):
        raw = '"gold_chain_matches_norm_ids": true\n"fabula_complete": false'
        checks, _, _ = _parse_validation(raw)
        assert checks["gold_chain_matches_norm_ids"] is True
        assert checks["fabula_complete"] is False

    def test_returns_three_elements(self):
        result = _parse_validation(_valid_json())
        assert len(result) == 3


# ---------------------------------------------------------------------------
# _build_user_prompt
# ---------------------------------------------------------------------------

class TestBuildUserPrompt:

    def test_contains_norm_ids(self):
        prompt = _build_user_prompt(_sample_task())
        assert "ТК_261" in prompt

    def test_contains_fabula(self):
        prompt = _build_user_prompt(_sample_task())
        assert "Иванова" in prompt

    def test_contains_question(self):
        prompt = _build_user_prompt(_sample_task())
        assert "Законно ли" in prompt

    def test_contains_gold_chain_step(self):
        prompt = _build_user_prompt(_sample_task())
        assert "Шаг 1" in prompt

    def test_open_book_includes_context_chunks(self):
        task = _sample_task(
            mode="open",
            context_chunks=[
                {"norm_id": "ТК_261", "title": "Гарантии", "text": "Текст нормы..."},
            ],
        )
        prompt = _build_user_prompt(task)
        assert "ТК_261" in prompt

    def test_closed_book_no_context_line(self):
        prompt = _build_user_prompt(_sample_task(mode="closed", context_chunks=None))
        assert "Контекст" not in prompt


# ---------------------------------------------------------------------------
# validate_task
# ---------------------------------------------------------------------------

class TestValidateTask:

    def test_valid_task_is_valid_true(self):
        result = validate_task(_sample_task(), MockJudge([_valid_json()]))
        assert result["is_valid"] is True

    def test_invalid_task_is_valid_false(self):
        result = validate_task(_sample_task(), MockJudge([_invalid_json()]))
        assert result["is_valid"] is False

    def test_result_has_required_keys(self):
        result = validate_task(_sample_task(), MockJudge([_valid_json()]))
        assert set(result.keys()) == {"is_valid", "checks", "issues", "judge_explanation"}

    def test_checks_has_five_keys(self):
        result = validate_task(_sample_task(), MockJudge([_valid_json()]))
        assert len(result["checks"]) == 5

    def test_partial_failure_is_invalid(self):
        result = validate_task(_sample_task(), MockJudge([_valid_json(fabula_complete=False)]))
        assert result["is_valid"] is False

    def test_judge_called_once(self):
        judge = MockJudge([_valid_json()])
        validate_task(_sample_task(), judge)
        assert len(judge.calls) == 1

    def test_user_prompt_contains_fabula(self):
        judge = MockJudge([_valid_json()])
        validate_task(_sample_task(fabula="УникальнаяФабула_XYZ"), judge)
        _, user_prompt = judge.calls[0]
        assert "УникальнаяФабула_XYZ" in user_prompt

    def test_issues_empty_when_valid(self):
        result = validate_task(_sample_task(), MockJudge([_valid_json(issues=[])]))
        assert result["issues"] == []

    def test_issues_present_when_invalid(self):
        result = validate_task(
            _sample_task(),
            MockJudge([_invalid_json(issues=["Логика нарушена в шаге 2"])]),
        )
        assert "Логика нарушена в шаге 2" in result["issues"]

    def test_judge_explanation_in_result(self):
        result = validate_task(_sample_task(), MockJudge([_valid_json()]))
        assert isinstance(result["judge_explanation"], str)
        assert len(result["judge_explanation"]) > 0


# ---------------------------------------------------------------------------
# run_validation
# ---------------------------------------------------------------------------

class TestRunValidation:

    def _make_tasks(self, n: int) -> list[dict]:
        return [_sample_task(id=f"task_{i}") for i in range(n)]

    def test_empty_tasks_returns_zero_counts(self):
        report = run_validation([], MockJudge([]))
        assert report["total"] == 0
        assert report["valid_count"] == 0
        assert report["rejected_count"] == 0

    def test_empty_valid_rate_is_zero(self):
        report = run_validation([], MockJudge([]))
        assert report["valid_rate"] == 0.0

    def test_all_valid_tasks(self):
        tasks = self._make_tasks(3)
        report = run_validation(tasks, MockJudge([_valid_json()] * 3))
        assert report["valid_count"] == 3
        assert report["rejected_count"] == 0

    def test_all_rejected_tasks(self):
        tasks = self._make_tasks(2)
        report = run_validation(tasks, MockJudge([_invalid_json()] * 2))
        assert report["valid_count"] == 0
        assert report["rejected_count"] == 2

    def test_mixed_valid_rejected(self):
        tasks = self._make_tasks(4)
        responses = [_valid_json(), _invalid_json(), _valid_json(), _invalid_json()]
        report = run_validation(tasks, MockJudge(responses))
        assert report["valid_count"] == 2
        assert report["rejected_count"] == 2

    def test_report_has_results_list(self):
        tasks = self._make_tasks(2)
        report = run_validation(tasks, MockJudge([_valid_json()] * 2))
        assert len(report["results"]) == 2

    def test_check_breakdown_all_passed(self):
        tasks = self._make_tasks(3)
        report = run_validation(tasks, MockJudge([_valid_json()] * 3))
        assert report["check_breakdown"]["gold_chain_matches_norm_ids"] == 3
        assert report["check_breakdown"]["fabula_complete"] == 3

    def test_check_breakdown_none_passed(self):
        tasks = self._make_tasks(2)
        report = run_validation(tasks, MockJudge([_invalid_json()] * 2))
        for name in CHECK_NAMES:
            assert report["check_breakdown"][name] == 0

    def test_valid_rate_calculation(self):
        tasks = self._make_tasks(4)
        responses = [_valid_json(), _invalid_json(), _valid_json(), _invalid_json()]
        report = run_validation(tasks, MockJudge(responses))
        assert report["valid_rate"] == pytest.approx(0.5)

    def test_output_file_created(self, tmp_path):
        tasks = self._make_tasks(1)
        output = str(tmp_path / "report.json")
        run_validation(tasks, MockJudge([_valid_json()]), output_path=output)
        assert pathlib.Path(output).exists()

    def test_output_file_is_valid_json(self, tmp_path):
        tasks = self._make_tasks(2)
        output = str(tmp_path / "sub" / "report.json")
        run_validation(tasks, MockJudge([_valid_json()] * 2), output_path=output)
        with open(output, encoding="utf-8") as f:
            data = json.load(f)
        assert "total" in data
        assert "results" in data

    def test_result_task_ids_preserved(self):
        tasks = [_sample_task(id="task_aaa"), _sample_task(id="task_bbb")]
        report = run_validation(tasks, MockJudge([_valid_json()] * 2))
        ids = [r["task_id"] for r in report["results"]]
        assert "task_aaa" in ids
        assert "task_bbb" in ids
