"""
LLM-валидатор задач MHGB (Шаг 6.1).

Автоматически проверяет качество сгенерированных задач по 5 критериям:
  1. gold_chain_matches_norm_ids — все нормы в gold_chain присутствуют в norm_ids
  2. task_solvable_from_context — задача решаема из контекста / вопрос однозначен
  3. fabula_complete — фабула содержит все детали для правоприменения
  4. no_logical_gaps — цепочка рассуждений непрерывна, без пробелов
  5. answer_consistent — ответ логически следует из gold_chain

Задача валидна (is_valid=True), если все 5 проверок пройдены.
"""

from __future__ import annotations

import json
import pathlib
import re
from typing import Any, TypedDict

from mhgb.eval.step_correctness import LLMJudge


# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

CHECK_NAMES: list[str] = [
    "gold_chain_matches_norm_ids",
    "task_solvable_from_context",
    "fabula_complete",
    "no_logical_gaps",
    "answer_consistent",
]


# ---------------------------------------------------------------------------
# Результат валидации
# ---------------------------------------------------------------------------

class ValidationResult(TypedDict):
    is_valid: bool
    checks: dict[str, bool]
    issues: list[str]
    judge_explanation: str


# ---------------------------------------------------------------------------
# Системный промпт судьи
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = """\
Ты — эксперт по качеству юридических задач для тестирования языковых моделей.
Оцени задачу по 5 критериям и верни результат строго в формате JSON (без markdown-обёртки).

Критерии:
1. gold_chain_matches_norm_ids — каждый norm_id в gold_chain присутствует в списке norm_ids задачи
2. task_solvable_from_context — для open-book: контекст достаточен; для closed-book: вопрос однозначен
3. fabula_complete — фабула содержит все необходимые факты (кто, что, когда)
4. no_logical_gaps — каждый шаг gold_chain логически следует из предыдущего без пробелов
5. answer_consistent — эталонный ответ логически вытекает из gold_chain и отвечает на вопрос

Формат ответа:
{
  "gold_chain_matches_norm_ids": true/false,
  "task_solvable_from_context": true/false,
  "fabula_complete": true/false,
  "no_logical_gaps": true/false,
  "answer_consistent": true/false,
  "issues": ["описание проблемы 1", ...],
  "explanation": "общее обоснование"
}

"issues" — пустой список, если все проверки прошли.
"""


# ---------------------------------------------------------------------------
# Построение промпта
# ---------------------------------------------------------------------------

def _build_user_prompt(task: dict[str, Any]) -> str:
    task_type = task.get("type", "?")
    task_mode = task.get("mode", "?")
    hop_group = task.get("hop_group", "?")
    norm_ids  = task.get("norm_ids", [])
    fabula    = task.get("fabula", "")
    question  = task.get("question", "")
    answer    = task.get("answer", "")

    gold_chain = task.get("gold_chain", [])
    gold_chain_str = "\n".join(
        f"  Шаг {s.get('step', '?')}: norm_id={s.get('norm_id')} — "
        f"{s.get('reasoning', s.get('conclusion', ''))}"
        for s in gold_chain
    )

    context_info = ""
    context_chunks = task.get("context_chunks")
    if context_chunks:
        chunk_ids = [c.get("norm_id", "?") for c in context_chunks]
        context_info = f"\nКонтекст (norm_ids): {chunk_ids}"

    return (
        f"Тип: {task_type} | Режим: {task_mode} | Сложность: {hop_group}\n"
        f"norm_ids: {norm_ids}\n"
        f"\nФабула:\n{fabula}\n"
        f"\nВопрос:\n{question}\n"
        f"\nЭталонный ответ:\n{answer}\n"
        f"\ngold_chain:\n{gold_chain_str}"
        f"{context_info}\n\n"
        "Оцени задачу по всем 5 критериям."
    )


# ---------------------------------------------------------------------------
# Парсинг ответа судьи
# ---------------------------------------------------------------------------

_BOOL_RE = re.compile(
    r'"(gold_chain_matches_norm_ids|task_solvable_from_context|fabula_complete'
    r'|no_logical_gaps|answer_consistent)"\s*:\s*(true|false)',
    re.IGNORECASE,
)


def _parse_validation(
    judge_response: str,
) -> tuple[dict[str, bool], list[str], str]:
    """
    Разбирает JSON-ответ судьи.
    Возвращает (checks, issues, explanation).
    При неудаче — все checks=False, пустые issues.
    """
    json_match = re.search(r'\{[\s\S]+\}', judge_response)
    if json_match:
        try:
            data = json.loads(json_match.group())
            checks = {name: bool(data.get(name, False)) for name in CHECK_NAMES}
            issues = [str(i) for i in data.get("issues", [])]
            explanation = str(data.get("explanation", judge_response))
            return checks, issues, explanation
        except (json.JSONDecodeError, ValueError):
            pass

    # Regex-fallback: ищем отдельные bool-поля
    checks = {name: False for name in CHECK_NAMES}
    for m in _BOOL_RE.finditer(judge_response):
        key, val = m.group(1), m.group(2)
        checks[key] = val.lower() == "true"
    return checks, [], judge_response


# ---------------------------------------------------------------------------
# Валидация одной задачи
# ---------------------------------------------------------------------------

def validate_task(
    task: dict[str, Any],
    judge_client: LLMJudge,
) -> ValidationResult:
    """
    Валидирует одну задачу по 5 критериям качества.

    task         — dict задачи (type, mode, norm_ids, fabula, question, answer, gold_chain, ...)
    judge_client — LLM-судья (реализует Protocol LLMJudge)
    """
    user_prompt = _build_user_prompt(task)
    response = judge_client.complete(_JUDGE_SYSTEM, user_prompt)
    checks, issues, explanation = _parse_validation(response)
    return {
        "is_valid": all(checks.values()),
        "checks": checks,
        "issues": issues,
        "judge_explanation": explanation,
    }


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_validation(
    tasks: list[dict[str, Any]],
    judge_client: LLMJudge,
    storage: Any = None,
    batch_id: str | None = None,
    output_path: str | None = None,
) -> dict[str, Any]:
    """
    Валидирует список задач, возвращает агрегированный отчёт.

    tasks        — список задач (dicts)
    judge_client — LLM-судья
    storage      — опциональный MongoStorage для сохранения ValidationLog
    batch_id     — ID батча задач (для MongoDB)
    output_path  — путь для сохранения JSON-отчёта

    Ключи отчёта: total, valid_count, rejected_count, valid_rate,
                  check_breakdown, results
    """
    results: list[dict[str, Any]] = []
    check_breakdown: dict[str, int] = {name: 0 for name in CHECK_NAMES}

    for task in tasks:
        result = validate_task(task, judge_client)
        task_id = task.get("id", "unknown")

        results.append({"task_id": task_id, **result})

        for name in CHECK_NAMES:
            if result["checks"].get(name, False):
                check_breakdown[name] += 1

        if storage is not None:
            _save_to_mongo(storage, task_id, batch_id, result)

    total = len(tasks)
    valid_count = sum(1 for r in results if r["is_valid"])

    report: dict[str, Any] = {
        "total": total,
        "valid_count": valid_count,
        "rejected_count": total - valid_count,
        "valid_rate": valid_count / total if total else 0.0,
        "check_breakdown": check_breakdown,
        "results": results,
    }

    if output_path is not None:
        pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    return report


def _save_to_mongo(
    storage: Any,
    task_id: str,
    batch_id: str | None,
    result: ValidationResult,
) -> None:
    try:
        from mhgb.storage.schemas import ValidationLog
        log = ValidationLog(
            task_id=task_id,
            task_batch_id=batch_id or "",
            validator="llm",
            is_valid=result["is_valid"],
            checks=result["checks"],
            issues=result["issues"],
            judge_explanation=result["judge_explanation"],
        )
        storage.db["validation_logs"].insert_one(log.model_dump())
    except Exception:
        pass
