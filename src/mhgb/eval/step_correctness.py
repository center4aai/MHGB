"""
Step Correctness metric — Level 2 (LLM-judge).

Оценивает, правильно ли модель применила каждую норму из gold_chain.

Схема оценки (по каждому шагу):
  1 — норма применена правильно и обоснование корректно
  0.5 — норма упомянута, но обоснование частично неверно или неполно
  0 — норма не применена или применена принципиально неверно

Итоговый step_correctness = среднее по всем оцениваемым шагам.
Шаги с norm_id=None (заключения) пропускаются.
"""

from __future__ import annotations

import re
from typing import Any, Protocol, runtime_checkable, TypedDict


# ---------------------------------------------------------------------------
# LLMJudge Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class LLMJudge(Protocol):
    """Минимальный интерфейс судьи-LLM."""

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Вызвать LLM с system + user промптами, вернуть текстовый ответ."""
        ...


# ---------------------------------------------------------------------------
# Результат оценки
# ---------------------------------------------------------------------------

class StepCorrectnessResult(TypedDict):
    step_scores: list[float]        # оценка по каждому шагу: 0 / 0.5 / 1
    step_correctness: float         # среднее по шагам, в [0, 1]
    judge_explanations: list[str]   # полный ответ судьи по каждому шагу


# Машиночитаемый маркер для записей, занулённых short-circuit'ом (фикс B):
# пустой ответ модели → балл 0 БЕЗ вызова судьи. Позволяет в анализе отличить
# «занулено гейтом» (overflow/пустота harness'а или модели) от «судья дал 0 на
# непустом ответе». Подстрока EMPTY_RESPONSE_SHORTCIRCUIT — для фильтрации.
EMPTY_RESPONSE_MARKER = "EMPTY_RESPONSE_SHORTCIRCUIT"


def _is_empty_response(text: str | None) -> bool:
    """True, если ответ модели вырожденный: None / "" / только whitespace.

    НЕ ловит усечённые-с-текстом (finish=length, но content есть) — это реальный
    (пусть неполный) ответ, его оценивает судья. Только полное отсутствие текста.
    """
    return not (text or "").strip()


# ---------------------------------------------------------------------------
# Системный промпт судьи
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = """\
Ты — судья-эксперт по правовому рассуждению. Твоя задача — оценить, насколько \
правильно тестируемая языковая модель применила конкретную правовую норму в своём ответе.

Критерии оценки:
  Score: 1   — норма применена правильно: она упомянута и обоснование соответствует эталону
  Score: 0.5 — норма упомянута, но обоснование неполное или содержит ошибку
  Score: 0   — норма не применена или применена принципиально неверно

Отвечай строго по формату:
Score: <0|0.5|1>
Обоснование: <одно-два предложения>
"""

# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

# Ищем "Score: 0", "Score: 0.5", "Score: 1" или просто "0.5"/"0"/"1" в ответе судьи.
# Порядок важен: 0.5 должен проверяться до "0" и "1".
_SCORE_RE = re.compile(r"\b(0\.5|1|0)\b")


def _parse_score(judge_response: str) -> float:
    """
    Извлекает оценку 0 / 0.5 / 1 из текстового ответа судьи.
    При неудаче возвращает 0.0 (консервативный дефолт).
    """
    m = _SCORE_RE.search(judge_response)
    return float(m.group(1)) if m else 0.0


def _build_user_prompt(step: dict[str, Any], model_response: str) -> str:
    norm_id  = step.get("norm_id") or "?"
    step_num = step.get("step", "?")
    gold_reasoning = step.get("reasoning", "—")
    return (
        f"Шаг {step_num}: норма {norm_id}\n"
        f"Эталонное обоснование: {gold_reasoning}\n\n"
        f"Ответ тестируемой модели:\n{model_response}\n\n"
        f"Оцени, правильно ли модель применила норму {norm_id} в своём ответе."
    )


# ---------------------------------------------------------------------------
# Основная функция
# ---------------------------------------------------------------------------

def evaluate_step_correctness(
    gold_chain: list[dict[str, Any]],
    model_response: str,
    judge_client: LLMJudge,
) -> StepCorrectnessResult:
    """
    Оценивает применение каждой нормы из gold_chain в ответе модели.

    gold_chain      — список шагов задачи (dicts с полями step, norm_id, reasoning)
    model_response  — полный текстовый ответ тестируемой модели
    judge_client    — LLM-судья (реализует Protocol LLMJudge)

    Шаги с norm_id=None (заключения) пропускаются.
    """
    evaluable = [s for s in gold_chain if s.get("norm_id")]

    # Фикс B (short-circuit ДО судьи): пустой ответ → 0 по всем шагам, судья не зовётся.
    # Структурный сбой судьи на вырожденном входе (нет текста → «додумывает» из эталона)
    # устраняется в корне для ВСЕХ вызывающих (основной прогон, cross-judge).
    if _is_empty_response(model_response):
        return {
            "step_scores": [0.0] * len(evaluable),
            "step_correctness": 0.0,
            "judge_explanations": [EMPTY_RESPONSE_MARKER] * len(evaluable),
        }

    step_scores: list[float] = []
    judge_explanations: list[str] = []

    for step in evaluable:
        user_prompt = _build_user_prompt(step, model_response)
        response = judge_client.complete(_JUDGE_SYSTEM, user_prompt)
        score = _parse_score(response)
        step_scores.append(score)
        judge_explanations.append(response)

    step_correctness = sum(step_scores) / len(step_scores) if step_scores else 0.0

    return {
        "step_scores": step_scores,
        "step_correctness": step_correctness,
        "judge_explanations": judge_explanations,
    }
