"""
Pydantic-схема задачи MHGB (v2).

Используется для:
- валидации задач, генерируемых generate_tasks.py
- сериализации в tasks_raw.jsonl и MongoDB (task_batches)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Формат структурированного ответа — хранится в каждой задаче и используется
# при составлении промптов для тестируемых моделей.
EXPECTED_ANSWER_FORMAT = (
    "ПРИМЕНИМЫЕ СТАТЬИ: [нормы в формате «ст. N КодексаX», через запятую]\n"
    "ЦЕПОЧКА РАССУЖДЕНИЙ: [пошаговое правовое обоснование]\n"
    "ВЫВОД: [краткий итог: законно / незаконно / иное]"
)


# ---------------------------------------------------------------------------
# Вспомогательные типы
# ---------------------------------------------------------------------------

TaskType = Literal[
    "issue_spotting",
    "rule_selection",
    "conflict_resolution",
    "temporal_validity",
]

TaskMode = Literal["closed", "open"]

HopGroup = Literal["shallow", "medium", "deep"]

BloomLevel = Literal["analyze", "apply", "evaluate", "apply+analyze"]

ContextConfig = Literal[
    "full_graph",
    "full_graph_edges",       # основной open-book режим: full_graph + типы связей + даты
]

ValidationStatus = Literal[
    "pending",
    "llm_validated",
    "expert_validated",
    "rejected",
]

# Автоматический маппинг type → bloom_level
_BLOOM_MAP: dict[str, BloomLevel] = {
    "issue_spotting":      "analyze",
    "rule_selection":      "apply",
    "conflict_resolution": "evaluate",
    "temporal_validity":   "apply+analyze",
}


def _hop_group(hop_count: int) -> HopGroup:
    if hop_count <= 2:
        return "shallow"
    if hop_count <= 4:
        return "medium"
    return "deep"


# ---------------------------------------------------------------------------
# Вложенные модели
# ---------------------------------------------------------------------------

class GoldChainStep(BaseModel):
    """Один шаг в цепочке правового рассуждения."""

    model_config = ConfigDict(extra="forbid")

    step:        int
    norm_id:     str
    reasoning:   str
    is_critical: bool = True


class ContextChunk(BaseModel):
    """Чанк контекста для open-book задачи."""

    model_config = ConfigDict(extra="forbid")

    norm_id:    str
    title:      str
    text:       str
    edge_type:  str | None = None
    valid_from: str | None = None   # дата вступления в силу (из корпуса)
    valid_to:   str | None = None   # дата утраты силы (если есть)
    is_repealed: bool | None = None # статья утратила силу
    related_to: str | None = None   # norm_id seed-нормы, с которой есть связь


# ---------------------------------------------------------------------------
# Основная схема Task
# ---------------------------------------------------------------------------

def _now() -> datetime:
    return datetime.now(timezone.utc)


class Task(BaseModel):
    """Задача MHGB для тестирования LLM на правоприменительном рассуждении."""

    model_config = ConfigDict(extra="forbid")

    id:           str
    mode:         TaskMode
    type:         TaskType
    bloom_level:  BloomLevel | None = None   # заполняется автоматически
    hop_count:    int = Field(ge=1)
    hop_group:    HopGroup | None = None     # заполняется автоматически
    branch_of_law: str

    norm_ids:  list[str] = Field(min_length=1)
    fabula:    str
    question:  str
    answer:    str

    gold_chain:     list[GoldChainStep] = Field(min_length=1)
    context_chunks: list[ContextChunk] | None = None
    context_config: ContextConfig | None = None

    expected_answer_format: str = Field(default=EXPECTED_ANSWER_FORMAT)

    graph_version_id:  str
    corpus_version_id: str

    validation_status: ValidationStatus = "pending"
    validation_meta:   dict[str, Any] = Field(default_factory=dict)

    generated_at:    datetime = Field(default_factory=_now)
    generator_model: str

    @model_validator(mode="after")
    def _fill_derived(self) -> "Task":
        if self.bloom_level is None:
            self.bloom_level = _BLOOM_MAP[self.type]
        if self.hop_group is None:
            self.hop_group = _hop_group(self.hop_count)
        return self

    @model_validator(mode="after")
    def _check_open_has_chunks(self) -> "Task":
        if self.mode == "open" and not self.context_chunks:
            raise ValueError("open-book задача должна содержать context_chunks")
        if self.mode == "closed" and self.context_chunks is not None:
            raise ValueError("closed-book задача не должна содержать context_chunks")
        return self

    @model_validator(mode="after")
    def _check_hop_count_matches_chain(self) -> "Task":
        if len(self.gold_chain) != self.hop_count:
            raise ValueError(
                f"hop_count={self.hop_count} не совпадает с len(gold_chain)={len(self.gold_chain)}"
            )
        return self
