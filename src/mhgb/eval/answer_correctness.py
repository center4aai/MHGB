"""
Answer Correctness metric — Level 3 (LLM-judge).

Оценивает финальный вывод модели (секция ВЫВОД) по 4-балльной шкале:
  1    — ответ полностью верен
  0.67 — ответ преимущественно верен, незначительные пробелы
  0.33 — ответ частично верен, но есть существенные ошибки
  0    — ответ неверен или прямо противоречит эталону
"""

from __future__ import annotations

import re
from typing import Any, TypedDict

from mhgb.eval.step_correctness import EMPTY_RESPONSE_MARKER, LLMJudge, _is_empty_response


# ---------------------------------------------------------------------------
# Результат оценки
# ---------------------------------------------------------------------------

class AnswerCorrectnessResult(TypedDict):
    score: float        # одно из {0, 0.33, 0.67, 1}
    explanation: str    # обоснование судьи


# Допустимые значения шкалы
VALID_SCORES: frozenset[float] = frozenset({0.0, 0.33, 0.67, 1.0})

# ---------------------------------------------------------------------------
# Системный промпт судьи
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = """\
Ты — судья-эксперт по правовому рассуждению. Твоя задача — оценить, насколько \
правильно тестируемая языковая модель ответила на юридический вопрос.

Шкала оценки:
  Score: 1    — ответ полностью верен: вывод совпадает с эталоном по сути
  Score: 0.67 — ответ преимущественно верен: правильное направление, но есть мелкие пробелы
  Score: 0.33 — ответ частично верен: верно отдельные элементы, но допущена существенная ошибка
  Score: 0    — ответ неверен или прямо противоречит эталону

Оценивай только финальный вывод (законно/незаконно/иное), а не полноту перечисления норм.

Отвечай строго по формату:
Score: <0|0.33|0.67|1>
Обоснование: <одно-два предложения>
"""

# ---------------------------------------------------------------------------
# Парсинг оценки
# ---------------------------------------------------------------------------

# Сначала ищем явный маркер "Score:", затем любое допустимое значение.
_SCORE_LABEL_RE = re.compile(r"Score:\s*(0\.67|0\.33|1|0)", re.IGNORECASE)
_SCORE_BARE_RE  = re.compile(r"\b(0\.67|0\.33|1|0)\b")


def _parse_score(judge_response: str) -> float:
    """
    Извлекает оценку из ответа судьи.
    Допустимые значения: 0, 0.33, 0.67, 1.
    При неудаче возвращает 0.0 (консервативный дефолт).
    """
    m = _SCORE_LABEL_RE.search(judge_response)
    if m:
        return float(m.group(1))
    m = _SCORE_BARE_RE.search(judge_response)
    if m:
        return float(m.group(1))
    return 0.0


# ---------------------------------------------------------------------------
# Построение промпта
# ---------------------------------------------------------------------------

def _build_user_prompt(
    gold_answer: str,
    model_answer: str,
    task_meta: dict[str, Any],
) -> str:
    task_type  = task_meta.get("type", "?")
    task_mode  = task_meta.get("mode", "?")
    hop_group  = task_meta.get("hop_group", "?")
    question   = task_meta.get("question", "")

    meta_line = f"Тип задачи: {task_type} | Режим: {task_mode} | Сложность: {hop_group}"
    question_line = f"Вопрос: {question}\n" if question else ""

    return (
        f"{meta_line}\n"
        f"{question_line}"
        f"\nЭталонный ответ:\n{gold_answer}\n"
        f"\nОтвет тестируемой модели:\n{model_answer}\n\n"
        "Оцени правильность финального вывода модели."
    )


# ---------------------------------------------------------------------------
# Основная функция
# ---------------------------------------------------------------------------

def evaluate_answer_correctness(
    gold_answer: str,
    model_answer: str,
    task_meta: dict[str, Any],
    judge_client: LLMJudge,
) -> AnswerCorrectnessResult:
    """
    Оценивает финальный ответ модели по 4-балльной шкале.

    gold_answer  — эталонный ответ из задачи
    model_answer — вывод из секции ВЫВОД структурированного ответа модели
    task_meta    — метаданные задачи (type, mode, hop_group, question)
    judge_client — LLM-судья (реализует Protocol LLMJudge)
    """
    # Фикс B (short-circuit ДО судьи): пустой ответ → 0, судья не зовётся.
    # Иначе судья «додумывает» вывод из эталона рядом в промпте (галлюцинация на пустом).
    if _is_empty_response(model_answer):
        return {"score": 0.0, "explanation": EMPTY_RESPONSE_MARKER}

    user_prompt = _build_user_prompt(gold_answer, model_answer, task_meta)
    response = judge_client.complete(_JUDGE_SYSTEM, user_prompt)
    score = _parse_score(response)
    return {"score": score, "explanation": response}
