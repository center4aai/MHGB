"""
RtA-детектор (Refusal to Answer) — Шаг 4.7.

Запускается ПОСТФАКТУМ отдельным скриптом, не inline в run_main_experiment.py.
Судья: тот же qwen3-27b-server, отдельный короткий промпт (~800–1200 токенов).

Пример использования:
    detector = RtADetector(judge_client, "prompts/rta_judge.md")
    result = detector.detect(fabula=..., question=..., model_response=...)
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

from mhgb.logging_setup import logger
from mhgb.models.llm_client import LLMClient

# ---------------------------------------------------------------------------
# Допустимые значения классификаторов
# ---------------------------------------------------------------------------

VALID_REFUSAL_TYPES: frozenset[str | None] = frozenset({
    "legal_advice_refusal",
    "harmful_content",
    "uncertainty_refusal",
    "scope_refusal",
    "generic_refusal",
    None,
})

VALID_REFUSAL_TOPICS: frozenset[str | None] = frozenset({
    "religious",
    "military",
    "minorities",
    "political",
    "medical",
    "financial_crime",
    "other",
    None,
})


# ---------------------------------------------------------------------------
# Результат детектора
# ---------------------------------------------------------------------------

@dataclass
class RtAResult:
    """Результат классификации одного ответа модели."""
    is_rta: bool
    refusal_type: str | None   # из VALID_REFUSAL_TYPES
    refusal_topic: str | None  # из VALID_REFUSAL_TOPICS
    confidence: float          # 0.0–1.0
    explanation: str
    task_id: str = field(default="")  # заполняется в detect_batch


# ---------------------------------------------------------------------------
# Вспомогательная функция парсинга JSON
# ---------------------------------------------------------------------------

def _parse_rta_json(raw: str) -> dict | None:
    """Убирает <think>-блоки, извлекает первый JSON-объект из текста."""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


# ---------------------------------------------------------------------------
# Детектор
# ---------------------------------------------------------------------------

class RtADetector:
    """
    Детектор отказов LLM отвечать на юридический вопрос.

    judge_client — любой LLMClient (Protocol), обычно qwen3-27b-server.
    prompt_path  — путь к prompts/rta_judge.md.
    """

    def __init__(self, judge_client: LLMClient, prompt_path: str | Path) -> None:
        self._client = judge_client
        prompt_path = Path(prompt_path)
        if not prompt_path.exists():
            raise FileNotFoundError(f"Промпт RtA не найден: {prompt_path}")
        self._system_prompt, self._user_template = self._load_prompt(prompt_path)

    # -- загрузка промпта --

    @staticmethod
    def _load_prompt(path: Path) -> tuple[str, str]:
        """Читает system-промпт и user-шаблон из .md файла."""
        text = path.read_text(encoding="utf-8")
        sys_m = re.search(
            r"## System prompt\s*```(.*?)```",
            text, re.DOTALL,
        )
        usr_m = re.search(
            r"## User prompt.*?```(.*?)```",
            text, re.DOTALL,
        )
        if not sys_m or not usr_m:
            raise ValueError(f"Не найден system или user промпт в {path}")
        return sys_m.group(1).strip(), usr_m.group(1).strip()

    # -- сборка и парсинг --

    def _build_user_prompt(
        self, fabula: str, question: str, model_response: str
    ) -> str:
        return (
            self._user_template
            .replace("{fabula}", fabula)
            .replace("{question}", question)
            .replace("{model_response}", model_response)
        )

    def _parse_result(self, raw: str) -> RtAResult:
        data = _parse_rta_json(raw)
        if data is None:
            logger.warning("RtADetector: не удалось распарсить JSON → is_rta=False")
            return RtAResult(
                is_rta=False,
                refusal_type=None,
                refusal_topic=None,
                confidence=0.0,
                explanation="Ошибка парсинга ответа судьи",
            )

        is_rta = bool(data.get("is_rta", False))

        refusal_type: str | None = data.get("refusal_type") or None
        refusal_topic: str | None = data.get("refusal_topic") or None

        if refusal_type not in VALID_REFUSAL_TYPES:
            logger.warning(
                f"RtADetector: неизвестный refusal_type={refusal_type!r}, сбрасываем в None"
            )
            refusal_type = None
        if refusal_topic not in VALID_REFUSAL_TOPICS:
            logger.warning(
                f"RtADetector: неизвестный refusal_topic={refusal_topic!r}, сбрасываем в None"
            )
            refusal_topic = None

        # Консистентность: при is_rta=False поля должны быть None
        if not is_rta:
            refusal_type = None
            refusal_topic = None

        try:
            confidence = float(data.get("confidence", 0.0))
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = 0.0

        explanation = str(data.get("brief_explanation", ""))

        return RtAResult(
            is_rta=is_rta,
            refusal_type=refusal_type,
            refusal_topic=refusal_topic,
            confidence=confidence,
            explanation=explanation,
        )

    # -- публичные методы --

    def detect(
        self,
        fabula: str,
        question: str,
        model_response: str,
    ) -> RtAResult:
        """
        Единичный вызов судьи.

        fabula, question, model_response — все три обязательны.
        При ошибке вызова возвращает is_rta=False с объяснением.
        """
        user_prompt = self._build_user_prompt(fabula, question, model_response)
        try:
            raw = self._client.generate(self._system_prompt, user_prompt)
        except Exception as exc:
            logger.warning(f"RtADetector.detect: ошибка вызова судьи: {exc}")
            return RtAResult(
                is_rta=False,
                refusal_type=None,
                refusal_topic=None,
                confidence=0.0,
                explanation=f"Ошибка вызова судьи: {exc}",
            )
        return self._parse_result(raw)

    def detect_batch(
        self,
        records: list[dict],
        max_workers: int = 4,
    ) -> list[RtAResult]:
        """
        Батчевый прогон с прогресс-баром tqdm.

        Каждый dict в records должен содержать:
          fabula         — обязательно
          question       — обязательно
          model_response — обязательно
          task_id        — опционально, переносится в RtAResult.task_id
        """
        results: list[RtAResult | None] = [None] * len(records)

        def _process(idx: int, rec: dict) -> tuple[int, RtAResult]:
            res = self.detect(
                fabula=rec["fabula"],
                question=rec["question"],
                model_response=rec["model_response"],
            )
            res.task_id = rec.get("task_id", "")
            return idx, res

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_process, i, rec): i
                for i, rec in enumerate(records)
            }
            with tqdm(total=len(records), desc="RtA detection", unit="запись") as pbar:
                for future in as_completed(futures):
                    idx, res = future.result()
                    results[idx] = res
                    pbar.update(1)

        return results  # type: ignore[return-value]
