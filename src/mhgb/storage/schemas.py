"""
Pydantic-схемы для коллекций MongoDB.

Каждая модель соответствует одной коллекции.
Поле `doc_id` — UUID-строка, используется для межколлекционных ссылок.
MongoDB-шный `_id` (ObjectId) создаётся автоматически при вставке.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

class CorpusVersion(BaseModel):
    """Версия корпуса НПА (результат parse_docs.py)."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(default_factory=_uuid)
    tag: str                          # e.g. "v1_initial", "v2_full_corpus"
    created_at: datetime = Field(default_factory=_now)
    article_count: int
    law_counts: dict[str, int]        # {"ТК": 531, "ГК": 1713, ...}
    checksum: str                     # SHA-256 corpus.jsonl
    notes: str | None = None


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

class GraphVersion(BaseModel):
    """Версия графа НПА (результат build_graph.py)."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(default_factory=_uuid)
    tag: str
    corpus_version_id: str            # ссылка на CorpusVersion.doc_id
    created_at: datetime = Field(default_factory=_now)
    node_count: int
    edge_count: int
    edge_type_counts: dict[str, int]  # {"ссылается_на": 4024, "исключает": 28, ...}
    params: dict[str, Any]            # max_pairs, model_name, prompt_version
    file_path: str | None = None      # путь к graph_vN.json


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

class TaskBatch(BaseModel):
    """Батч сгенерированных задач (результат generate_tasks.py)."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(default_factory=_uuid)
    tag: str
    graph_version_id: str
    corpus_version_id: str
    created_at: datetime = Field(default_factory=_now)
    task_count: int
    task_type_counts: dict[str, int]  # {"issue_spotting": 50, ...}
    generator_model: str
    seed: int | None = None
    notes: str | None = None


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

class Experiment(BaseModel):
    """Метаданные одного прогона модели на задачах."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(default_factory=_uuid)
    name: str
    task_batch_id: str
    model_name: str
    model_size_b: int | None = None   # размер в миллиардах параметров
    adapted_to_ru: bool = False
    modes: list[Literal["closed", "open"]]
    context_config: Literal["full_graph", "full_graph_edges"] | None = None
    created_at: datetime = Field(default_factory=_now)
    status: Literal["running", "completed", "failed"] = "running"
    params: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Model responses
# ---------------------------------------------------------------------------

class ModelResponse(BaseModel):
    """Сырой и распаршенный ответ модели на одну задачу."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(default_factory=_uuid)
    experiment_id: str
    task_id: str
    model_name: str
    mode: Literal["closed", "open"]
    raw_response: str
    parsed_articles: list[str] | None = None   # ["ТК_261", "ТК_256"]
    parsed_reasoning: str | None = None
    parsed_conclusion: str | None = None
    created_at: datetime = Field(default_factory=_now)
    latency_ms: int | None = None
    tokens_input: int | None = None
    tokens_output: int | None = None
    # RtA-поля (заполняются постфактум скриптом run_rta_detection.py)
    is_rta: bool | None = None
    rta_type: str | None = None
    rta_topic: str | None = None
    rta_confidence: float | None = None
    rta_explanation: str | None = None


# ---------------------------------------------------------------------------
# Evaluations
# ---------------------------------------------------------------------------

class Evaluation(BaseModel):
    """Оценки трёх уровней метрики для одного ответа модели."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(default_factory=_uuid)
    experiment_id: str
    task_id: str
    model_response_id: str
    # Уровень 1 — Norm Coverage (детерминированная)
    norm_coverage_list: float | None = None       # precision/recall по списку статей
    norm_coverage_reasoning: float | None = None  # по упоминаниям в тексте
    list_reasoning_consistency: float | None = None
    # Уровень 2 — Step Correctness (LLM-judge)
    step_correctness: float | None = None
    step_scores: list[float] = Field(default_factory=list)
    # Уровень 3 — Answer Correctness (LLM-judge)
    answer_correctness: float | None = None
    # Итого
    final_score: float | None = None
    gap_score: float | None = None                # open_score - closed_score
    judge_logs: list[dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class ValidationLog(BaseModel):
    """Результат LLM-валидатора или юриста на одну задачу."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(default_factory=_uuid)
    task_id: str
    task_batch_id: str
    validator: Literal["llm", "expert"]
    is_valid: bool
    checks: dict[str, bool] = Field(default_factory=dict)
    # e.g. {"gold_chain_matches_norm_ids": True, "task_solvable_from_context": False, ...}
    issues: list[str] = Field(default_factory=list)
    judge_explanation: str | None = None
    created_at: datetime = Field(default_factory=_now)
