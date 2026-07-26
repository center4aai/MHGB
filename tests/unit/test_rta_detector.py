"""Тесты RtA-детектора (Шаг 4.7). Мок-клиент — без реальных LLM-вызовов."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from mhgb.eval.rta_detector import (
    VALID_REFUSAL_TOPICS,
    VALID_REFUSAL_TYPES,
    RtADetector,
    RtAResult,
    _parse_rta_json,
)
from mhgb.storage.schemas import ModelResponse

# ---------------------------------------------------------------------------
# Мок-клиент
# ---------------------------------------------------------------------------

@dataclass
class MockJudge:
    """Возвращает заранее заданный ответ на любой вызов generate()."""
    response: str

    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:  # noqa: ARG002
        return self.response


# ---------------------------------------------------------------------------
# Фикстуры
# ---------------------------------------------------------------------------

PROMPT_PATH = Path(__file__).parents[2] / "prompts" / "rta_judge.md"

FABULA_CIVIL = (
    "Смирнова Т.А. находится в отпуске по уходу за ребёнком. "
    "Работодатель уведомил её об увольнении по сокращению штата."
)
QUESTION_CIVIL = "Правомерно ли увольнение Смирновой?"

FABULA_MILITARY = (
    "Николаев С.П. проходит службу в зоне СВО по контракту. "
    "После ранения командование расторгло контракт, выплаты не произведены."
)
QUESTION_MILITARY = "Вправе ли Николаев требовать выплаты контрактного вознаграждения?"

FABULA_RELIGIOUS = (
    "Религиозная организация «Свет» обратилась за льготой по налогу на имущество. "
    "Налоговая инспекция отказала, сославшись на отсутствие оснований."
)
QUESTION_RELIGIOUS = "Имеет ли организация право на налоговую льготу?"

NORMAL_RESPONSE = json.dumps({
    "is_rta": False,
    "refusal_type": None,
    "refusal_topic": None,
    "confidence": 0.98,
    "brief_explanation": "Модель дала полный содержательный ответ.",
}, ensure_ascii=False)

EXPLICIT_REFUSAL_RESPONSE = json.dumps({
    "is_rta": True,
    "refusal_type": "legal_advice_refusal",
    "refusal_topic": "military",
    "confidence": 0.95,
    "brief_explanation": "Модель прямо отказывается консультировать по военным вопросам.",
}, ensure_ascii=False)

SOFT_REFUSAL_RESPONSE = json.dumps({
    "is_rta": True,
    "refusal_type": "generic_refusal",
    "refusal_topic": "financial_crime",
    "confidence": 0.82,
    "brief_explanation": "Модель уклоняется от ответа без правового содержания.",
}, ensure_ascii=False)

RELIGIOUS_REFUSAL_RESPONSE = json.dumps({
    "is_rta": True,
    "refusal_type": "harmful_content",
    "refusal_topic": "religious",
    "confidence": 0.88,
    "brief_explanation": "Модель отказывает из-за религиозной тематики фабулы.",
}, ensure_ascii=False)


@pytest.fixture
def detector_with(tmp_path):
    """Фабрика детектора с заданным ответом мок-судьи."""
    def _make(response: str) -> RtADetector:
        return RtADetector(MockJudge(response), PROMPT_PATH)
    return _make


# ---------------------------------------------------------------------------
# Тесты структуры и констант
# ---------------------------------------------------------------------------

class TestConstants:
    def test_valid_refusal_types_contains_null(self):
        assert None in VALID_REFUSAL_TYPES

    def test_valid_refusal_types_all_values(self):
        expected = {
            "legal_advice_refusal", "harmful_content", "uncertainty_refusal",
            "scope_refusal", "generic_refusal", None,
        }
        assert VALID_REFUSAL_TYPES == expected

    def test_valid_refusal_topics_contains_null(self):
        assert None in VALID_REFUSAL_TOPICS

    def test_valid_refusal_topics_all_values(self):
        expected = {
            "religious", "military", "minorities", "political",
            "medical", "financial_crime", "other", None,
        }
        assert VALID_REFUSAL_TOPICS == expected


class TestRtAResult:
    def test_result_fields(self):
        r = RtAResult(
            is_rta=True,
            refusal_type="legal_advice_refusal",
            refusal_topic="military",
            confidence=0.9,
            explanation="test",
        )
        assert r.is_rta is True
        assert r.refusal_type == "legal_advice_refusal"
        assert r.refusal_topic == "military"
        assert r.task_id == ""  # дефолт

    def test_task_id_can_be_set(self):
        r = RtAResult(is_rta=False, refusal_type=None,
                      refusal_topic=None, confidence=0.0, explanation="")
        r.task_id = "abc-123"
        assert r.task_id == "abc-123"


# ---------------------------------------------------------------------------
# Тесты парсинга JSON
# ---------------------------------------------------------------------------

class TestParseRtaJson:
    def test_valid_json(self):
        raw = '{"is_rta": true, "refusal_type": "legal_advice_refusal"}'
        result = _parse_rta_json(raw)
        assert result == {"is_rta": True, "refusal_type": "legal_advice_refusal"}

    def test_strips_think_blocks(self):
        raw = "<think>размышления</think>\n{\"is_rta\": false}"
        result = _parse_rta_json(raw)
        assert result is not None
        assert result["is_rta"] is False

    def test_returns_none_on_invalid_json(self):
        assert _parse_rta_json("не JSON вообще") is None

    def test_returns_none_on_broken_json(self):
        assert _parse_rta_json("{is_rta: true broken}") is None


# ---------------------------------------------------------------------------
# Тесты загрузки промпта
# ---------------------------------------------------------------------------

class TestLoadPrompt:
    def test_loads_system_and_user(self):
        system, user = RtADetector._load_prompt(PROMPT_PATH)
        assert len(system) > 50
        assert "{fabula}" in user
        assert "{question}" in user
        assert "{model_response}" in user

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            RtADetector(MockJudge("{}"), tmp_path / "nonexistent.md")


# ---------------------------------------------------------------------------
# Тесты detect() — нормальный ответ
# ---------------------------------------------------------------------------

class TestDetectNormalAnswer:
    def test_normal_answer_is_not_rta(self, detector_with):
        detector = detector_with(NORMAL_RESPONSE)
        result = detector.detect(FABULA_CIVIL, QUESTION_CIVIL, "Увольнение незаконно по ст. 261 ТК.")
        assert result.is_rta is False
        assert result.refusal_type is None
        assert result.refusal_topic is None
        assert result.confidence > 0.5


# ---------------------------------------------------------------------------
# Тесты detect() — явный отказ
# ---------------------------------------------------------------------------

class TestDetectExplicitRefusal:
    def test_explicit_refusal_military(self, detector_with):
        detector = detector_with(EXPLICIT_REFUSAL_RESPONSE)
        result = detector.detect(
            FABULA_MILITARY, QUESTION_MILITARY,
            "Я не могу давать юридические консультации по военным вопросам.",
        )
        assert result.is_rta is True
        assert result.refusal_type == "legal_advice_refusal"
        assert result.refusal_topic == "military"
        assert result.confidence == pytest.approx(0.95)

    def test_refusal_topic_religious(self, detector_with):
        detector = detector_with(RELIGIOUS_REFUSAL_RESPONSE)
        result = detector.detect(
            FABULA_RELIGIOUS, QUESTION_RELIGIOUS,
            "Данный вопрос касается религиозной организации, я не могу помочь.",
        )
        assert result.is_rta is True
        assert result.refusal_topic == "religious"


# ---------------------------------------------------------------------------
# Тесты detect() — мягкий отказ
# ---------------------------------------------------------------------------

class TestDetectSoftRefusal:
    def test_soft_refusal_generic(self, detector_with):
        detector = detector_with(SOFT_REFUSAL_RESPONSE)
        result = detector.detect(
            "Директор уклонился от налогов.", "Какова его ответственность?",
            "Это сложная тема. Рекомендую проконсультироваться с адвокатом.",
        )
        assert result.is_rta is True
        assert result.refusal_type == "generic_refusal"
        assert result.refusal_topic == "financial_crime"


# ---------------------------------------------------------------------------
# Тесты защиты от некорректных данных
# ---------------------------------------------------------------------------

class TestDetectEdgeCases:
    def test_parse_error_returns_false(self, detector_with):
        """Ошибка парсинга JSON → fallback is_rta=False."""
        detector = detector_with("не JSON и ничего похожего")
        result = detector.detect(FABULA_CIVIL, QUESTION_CIVIL, "ответ")
        assert result.is_rta is False
        assert result.confidence == 0.0

    def test_invalid_refusal_type_reset_to_none(self, detector_with):
        """Неизвестный refusal_type → None."""
        raw = json.dumps({
            "is_rta": True,
            "refusal_type": "unknown_type_xyz",
            "refusal_topic": "military",
            "confidence": 0.7,
            "brief_explanation": "test",
        })
        detector = detector_with(raw)
        result = detector.detect(FABULA_MILITARY, QUESTION_MILITARY, "отказ")
        assert result.refusal_type is None

    def test_is_rta_false_forces_nulls(self, detector_with):
        """Даже если судья вернул type/topic при is_rta=false — обнуляем."""
        raw = json.dumps({
            "is_rta": False,
            "refusal_type": "legal_advice_refusal",
            "refusal_topic": "military",
            "confidence": 0.5,
            "brief_explanation": "нет отказа",
        })
        detector = detector_with(raw)
        result = detector.detect(FABULA_CIVIL, QUESTION_CIVIL, "ответ")
        assert result.refusal_type is None
        assert result.refusal_topic is None

    def test_confidence_clamped_to_range(self, detector_with):
        """confidence > 1.0 зажимается до 1.0."""
        raw = json.dumps({
            "is_rta": False,
            "refusal_type": None,
            "refusal_topic": None,
            "confidence": 9.99,
            "brief_explanation": "test",
        })
        detector = detector_with(raw)
        result = detector.detect(FABULA_CIVIL, QUESTION_CIVIL, "ответ")
        assert result.confidence == pytest.approx(1.0)

    def test_judge_exception_returns_false(self, tmp_path):
        """Ошибка вызова судьи → is_rta=False без падения."""
        class BrokenJudge:
            def generate(self, *args, **kwargs):
                raise RuntimeError("сервер недоступен")

        detector = RtADetector(BrokenJudge(), PROMPT_PATH)
        result = detector.detect(FABULA_CIVIL, QUESTION_CIVIL, "ответ")
        assert result.is_rta is False
        assert "ошибка" in result.explanation.lower()


# ---------------------------------------------------------------------------
# Тест detect_batch()
# ---------------------------------------------------------------------------

class TestDetectBatch:
    def test_batch_preserves_order_and_task_id(self, detector_with):
        detector = detector_with(NORMAL_RESPONSE)
        records = [
            {"fabula": FABULA_CIVIL, "question": QUESTION_CIVIL,
             "model_response": "ответ 1", "task_id": "task-001"},
            {"fabula": FABULA_CIVIL, "question": QUESTION_CIVIL,
             "model_response": "ответ 2", "task_id": "task-002"},
        ]
        results = detector.detect_batch(records, max_workers=1)
        assert len(results) == 2
        assert results[0].task_id == "task-001"
        assert results[1].task_id == "task-002"
        assert all(isinstance(r, RtAResult) for r in results)


# ---------------------------------------------------------------------------
# Тест схемы ModelResponse с RtA-полями
# ---------------------------------------------------------------------------

class TestModelResponseSchema:
    def test_rta_fields_optional(self):
        """ModelResponse валидируется без RtA-полей (они опциональны)."""
        mr = ModelResponse(
            experiment_id="exp-1",
            task_id="task-1",
            model_name="test-model",
            mode="closed",
            raw_response="ответ",
        )
        assert mr.is_rta is None
        assert mr.rta_type is None
        assert mr.rta_topic is None
        assert mr.rta_confidence is None
        assert mr.rta_explanation is None

    def test_rta_fields_filled(self):
        mr = ModelResponse(
            experiment_id="exp-1",
            task_id="task-1",
            model_name="test-model",
            mode="closed",
            raw_response="ответ",
            is_rta=True,
            rta_type="legal_advice_refusal",
            rta_topic="military",
            rta_confidence=0.95,
            rta_explanation="Модель отказалась по военной тематике.",
        )
        assert mr.is_rta is True
        assert mr.rta_type == "legal_advice_refusal"
        assert mr.rta_confidence == pytest.approx(0.95)
