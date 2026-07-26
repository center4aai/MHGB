"""
Context builders для open-book режима.

Две конфигурации контекста с единым интерфейсом → list[ContextChunk]:
  build_full_graph_context       — seed-нормы + все соседи (без метаданных рёбер)
  build_full_graph_edges_context — то же + типы связей и даты действия (основной режим)
"""
from __future__ import annotations

import hashlib
import random

from mhgb.schemas.task import ContextChunk

STRUCTURAL_EDGE_TYPES: frozenset[str] = frozenset({"ссылается_на", "применяется_к"})
DOCTRINAL_EDGE_TYPES: frozenset[str] = frozenset({"исключает", "приоритет", "дополняет"})

# Оценка числа токенов: ~3 символа на токен для русского юридического текста.
# Бюджет 40 000 est. токенов ≈ 55–60K реальных токенов; оставляет ~12K на промпт/ответ.
_CHARS_PER_TOKEN: int = 3
DEFAULT_TOKEN_BUDGET: int = 40_000


# ---------------------------------------------------------------------------
# Внутренние хелперы
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """Грубая оценка числа токенов для русского текста."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _to_chunk(
    norm_id: str,
    corpus: dict,
    edge_type: str | None = None,
    related_to: str | None = None,
) -> ContextChunk | None:
    """Создаёт ContextChunk из записи корпуса. None если норма не найдена."""
    article = corpus.get(norm_id)
    if article is None:
        return None
    title = article.get("title") or f"Статья {article.get('article', '')} {article.get('law_short', '')}"
    return ContextChunk(
        norm_id=norm_id,
        title=title.strip(),
        text=article.get("text", ""),
        edge_type=edge_type,
        valid_from=article.get("valid_from"),
        valid_to=article.get("valid_to"),
        is_repealed=article.get("is_repealed"),
        related_to=related_to,
    )


def _get_neighbors(
    seed_ids: list[str],
    graph,
    allowed_types: frozenset[str] | None = None,
) -> list[tuple[str, str, str]]:
    """
    Возвращает (neighbor_norm_id, edge_type, source_seed_id) для всех соседей seed_ids.
    Обходит рёбра в обе стороны; семена исключаются из результата.
    allowed_types=None — принимать все типы рёбер.
    """
    seed_set = set(seed_ids)
    seen: set[str] = set()
    result: list[tuple[str, str, str]] = []

    for node in seed_ids:
        if node not in graph:
            continue
        for _, neighbor, data in graph.out_edges(node, data=True):
            etype = data.get("edge_type", "")
            if allowed_types is not None and etype not in allowed_types:
                continue
            if neighbor not in seed_set and neighbor not in seen:
                seen.add(neighbor)
                result.append((neighbor, etype, node))
        for neighbor, _, data in graph.in_edges(node, data=True):
            etype = data.get("edge_type", "")
            if allowed_types is not None and etype not in allowed_types:
                continue
            if neighbor not in seed_set and neighbor not in seen:
                seen.add(neighbor)
                result.append((neighbor, etype, node))

    return result


def _build_context(
    norm_ids: list[str],
    graph,
    corpus: dict,
    allowed_types: frozenset[str] | None = None,
    max_chunks: int | None = None,
    token_budget: int | None = DEFAULT_TOKEN_BUDGET,
    task_id: str | None = None,
) -> list[ContextChunk]:
    """Семена (edge_type=None) + соседи через рёбра allowed_types.

    Ограничения применяются совместно:
      max_chunks    — лимит по числу чанков (соседи).
      token_budget  — лимит по оценочному числу токенов (seed-нормы включаются всегда,
                      соседи добавляются пока бюджет не исчерпан).
    task_id         — если задан, весь итоговый список чанков (seeds + соседи) перемешивается
                      детерминированно (SHA-256 хэш task_id → seed), чтобы исключить
                      позиционный bias: gold-нормы не лежат всегда в начале контекста.
    """
    chunks: list[ContextChunk] = []
    used_tokens: int = 0

    # seed-нормы — всегда включаем независимо от бюджета
    for nid in norm_ids:
        chunk = _to_chunk(nid, corpus, edge_type=None)
        if chunk is not None:
            chunks.append(chunk)
            used_tokens += _estimate_tokens(chunk.text)

    available = (max_chunks - len(chunks)) if max_chunks is not None else None

    for neighbor, etype, source in _get_neighbors(norm_ids, graph, allowed_types):
        if available is not None and available <= 0:
            break
        chunk = _to_chunk(neighbor, corpus, edge_type=etype, related_to=source)
        if chunk is not None:
            chunk_tokens = _estimate_tokens(chunk.text)
            if token_budget is not None and used_tokens + chunk_tokens > token_budget:
                break
            chunks.append(chunk)
            used_tokens += chunk_tokens
            if available is not None:
                available -= 1

    # Детерминированное перемешивание: исключает позиционный bias (gold-нормы ≠ всегда первые).
    # hash() нельзя: PYTHONHASHSEED рандомизирует между запусками.
    # SHA-256 стабилен между сессиями и платформами.
    if task_id is not None:
        seed_int = int(hashlib.sha256(task_id.encode()).hexdigest(), 16) % (2 ** 32)
        random.Random(seed_int).shuffle(chunks)

    return chunks


# ---------------------------------------------------------------------------
# Публичные функции
# ---------------------------------------------------------------------------

def build_full_graph_context(
    norm_ids: list[str],
    graph,
    corpus: dict,
    max_chunks: int | None = 25,
    token_budget: int | None = DEFAULT_TOKEN_BUDGET,
    task_id: str | None = None,
) -> list[ContextChunk]:
    """Контекст: gold_chain нормы + прямые соседи по любому типу рёбер.
    max_chunks=25 и token_budget=40K защищают от переполнения контекстного окна.
    """
    return _build_context(norm_ids, graph, corpus, allowed_types=None,
                          max_chunks=max_chunks, token_budget=token_budget,
                          task_id=task_id)


def build_full_graph_edges_context(
    norm_ids: list[str],
    graph,
    corpus: dict,
    max_chunks: int | None = 25,
    token_budget: int | None = DEFAULT_TOKEN_BUDGET,
    task_id: str | None = None,
) -> list[ContextChunk]:
    """Контекст ((3)): gold_chain нормы + соседи + ненаправленные типы связей + даты.

    Данные идентичны full_graph_context; чанки содержат edge_type, related_to,
    valid_from/valid_to/is_repealed — для рендера промпта с мета-информацией.
    Порядок чанков перемешан детерминированно per-task (SHA-256 от task_id), чтобы
    исключить позиционный bias: gold-нормы не лежат всегда в начале контекста.
    """
    return _build_context(norm_ids, graph, corpus, allowed_types=None,
                          max_chunks=max_chunks, token_budget=token_budget,
                          task_id=task_id)
