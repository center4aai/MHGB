"""
Шаг 3: Генерация задач из графа норм.

Для каждого типа задач:
  1. Селектор выбирает цепочку нод из графа
  2. Промпт передаётся LLM-генератору
  3. Ответ парсится → структурированная задача (closed + open-book)
  4. Результат сохраняется в data/tasks_raw.jsonl

Запуск:
  uv run python src/mhgb/generate_tasks.py --type all --n 50
  uv run python src/mhgb/generate_tasks.py --type conflict_resolution --n 20
"""

import argparse
import json
import os
import random
import re
import sys
import uuid
from pathlib import Path

import networkx as nx
from tqdm import tqdm

from mhgb.config import cfg
from mhgb.logging_setup import logger
from mhgb.schemas.task import EXPECTED_ANSWER_FORMAT

REPO_ROOT = cfg.repo_root
DATA_DIR  = cfg.data_dir

SEMANTIC_EDGE_TYPES = {"исключает", "дополняет", "приоритет", "применяется_к"}
CONFLICT_EDGE_TYPES = {"исключает", "приоритет"}

TASK_TYPES  = ["issue_spotting", "rule_selection", "conflict_resolution", "temporal_validity"]
HOP_GROUPS  = ["shallow", "medium", "deep"]

MAX_HOP_DEPTH = 10  # практический потолок длины цепочки

# Диапазоны глубины для каждой hop-группы [min, max]
HOP_GROUP_DEPTHS: dict[str, tuple[int, int]] = {
    "shallow": (1, 2),
    "medium":  (3, 4),
    "deep":    (5, MAX_HOP_DEPTH),
}

GENERATOR_API_BASE = cfg.generator_api_base
GENERATOR_MODEL    = cfg.generator_model

# ---------------------------------------------------------------------------
# Загрузка данных
# ---------------------------------------------------------------------------

def load_data() -> tuple[dict, nx.DiGraph]:
    corpus = {}
    with open(DATA_DIR / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            a = json.loads(line)
            corpus[a["id"]] = a
    G = nx.node_link_graph(
        json.loads((DATA_DIR / "graph.json").read_text(encoding="utf-8")),
        directed=True,
    )
    return corpus, G


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _is_usable(nid: str, corpus: dict) -> bool:
    """Нода пригодна для задач: не утратила силу, есть текст."""
    art = corpus.get(nid, {})
    return (
        not art.get("is_repealed")
        and len(art.get("text", "")) >= 100
    )


def _norm_block(nid: str, corpus: dict) -> str:
    """Форматированный блок нормы для промпта."""
    art = corpus.get(nid, {})
    vf = f" (ред. от {art['valid_from']})" if art.get("valid_from") else ""
    return (
        f"[{nid}]{vf} {art.get('title', '')}\n"
        f"{art.get('text', '')[:1200]}"
    )


def _hop_group_of(hop_count: int) -> str:
    if hop_count <= 2:
        return "shallow"
    if hop_count <= 4:
        return "medium"
    return "deep"


def _parse_llm_json(raw: str) -> dict | None:
    """Убирает <think>-блоки, извлекает JSON."""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


def _build_task(
    task_type: str,
    norm_ids: list[str],
    corpus: dict,
    llm_output: dict,
    edge_meta: list[dict] | None = None,
) -> list[dict]:
    """
    Из ответа LLM формирует пару задач: closed-book и open-book.
    Возвращает список из 2 dict.
    """
    base_id = str(uuid.uuid4())

    context_chunks = [
        {"norm_id": nid, "text": corpus.get(nid, {}).get("text", "")}
        for nid in norm_ids
    ]

    hop_count = len(norm_ids)
    shared = {
        "type": task_type,
        "hop_count": hop_count,
        "hop_group": _hop_group_of(hop_count),
        "norm_ids": norm_ids,
        "fabula": llm_output.get("fabula", ""),
        "question": llm_output.get("question", ""),
        "answer": llm_output.get("answer", ""),
        "gold_chain": llm_output.get("gold_chain", []),
        "expected_answer_format": EXPECTED_ANSWER_FORMAT,
    }
    if edge_meta:
        shared["edge_meta"] = edge_meta

    closed = {"id": f"{base_id}_closed", "mode": "closed", "context_chunks": None, **shared}
    open_  = {"id": f"{base_id}_open",   "mode": "open",   "context_chunks": context_chunks, **shared}
    return [closed, open_]


# ---------------------------------------------------------------------------
# Вспомогательный DFS для построения цепочек нужной глубины
# ---------------------------------------------------------------------------

def _dfs_chains(
    G: nx.DiGraph,
    corpus: dict,
    start_pool: list[str],
    min_d: int,
    max_d: int,
    n: int,
    filter_fn=None,
) -> list[dict]:
    """
    DFS-генератор цепочек заданной глубины из пула стартовых нод.
    filter_fn(chain) → bool — дополнительный фильтр на итоговую цепочку.
    Возвращает список dict с ключами norm_ids + edges.
    """
    results, seen, attempts = [], set(), 0
    while len(results) < n and attempts < n * 50:
        attempts += 1
        depth = random.randint(min_d, max_d)
        node  = random.choice(start_pool)
        chain = [node]
        edges = []
        for _ in range(depth - 1):
            neighbors = [
                v for _, v in G.out_edges(node)
                if _is_usable(v, corpus) and v not in chain
            ]
            if not neighbors:
                break
            next_node = random.choice(neighbors)
            nd = G.get_edge_data(node, next_node, {})
            edges.append({
                "source": node, "target": next_node,
                "edge_type": nd.get("edge_type", "ссылается_на"),
                "explanation": nd.get("explanation", ""),
            })
            chain.append(next_node)
            node = next_node
        key = tuple(chain)
        if len(chain) >= min_d and key not in seen:
            if filter_fn is None or filter_fn(chain):
                seen.add(key)
                results.append({"norm_ids": chain, "edges": edges})
    return results


# ---------------------------------------------------------------------------
# Селекторы цепочек
# ---------------------------------------------------------------------------

def select_issue_spotting(
    G: nx.DiGraph, corpus: dict, n: int, target_hop_group: str | None = None
) -> list[dict]:
    """
    shallow: одиночные нормы.
    medium/deep: цепочки, стартующие предпочтительно от мульти-отраслевых нод
                 (out-соседи охватывают ≥2 отрасли права).
    """
    if target_hop_group is None or target_hop_group == "shallow":
        candidates = [nid for nid in G.nodes() if _is_usable(nid, corpus)]
        sample = random.sample(candidates, min(n, len(candidates)))
        return [{"norm_ids": [nid]} for nid in sample]

    min_d, max_d = HOP_GROUP_DEPTHS[target_hop_group]

    # Мульти-отраслевые ноды — предпочтительные стартовые точки
    multi_branch = [
        nid for nid in G.nodes()
        if _is_usable(nid, corpus) and len({
            corpus.get(v, {}).get("branch_of_law", "")
            for _, v in G.out_edges(nid)
            if _is_usable(v, corpus)
        }) >= 2
    ]
    all_with_out = [
        nid for nid in G.nodes()
        if _is_usable(nid, corpus)
        and any(_is_usable(v, corpus) for _, v in G.out_edges(nid))
    ]
    if not all_with_out:
        return []

    results, seen, attempts = [], set(), 0
    while len(results) < n and attempts < n * 50:
        attempts += 1
        depth = random.randint(min_d, max_d)
        # 70% от мульти-отраслевых нод, если они есть
        if multi_branch and random.random() < 0.7:
            node = random.choice(multi_branch)
        else:
            node = random.choice(all_with_out)
        chain = [node]
        edges = []
        for _ in range(depth - 1):
            neighbors = [
                v for _, v in G.out_edges(node)
                if _is_usable(v, corpus) and v not in chain
            ]
            if not neighbors:
                break
            next_node = random.choice(neighbors)
            nd = G.get_edge_data(node, next_node, {})
            edges.append({
                "source": node, "target": next_node,
                "edge_type": nd.get("edge_type", "ссылается_на"),
                "explanation": nd.get("explanation", ""),
            })
            chain.append(next_node)
            node = next_node
        key = tuple(chain)
        if len(chain) >= min_d and key not in seen:
            seen.add(key)
            results.append({"norm_ids": chain, "edges": edges})
    return results


def select_rule_selection(
    G: nx.DiGraph, corpus: dict, n: int, target_hop_group: str | None = None
) -> list[dict]:
    """Цепочки через семантические/структурные рёбра, глубина до MAX_HOP_DEPTH."""
    min_d, max_d = HOP_GROUP_DEPTHS.get(target_hop_group, (2, MAX_HOP_DEPTH)) \
        if target_hop_group else (2, MAX_HOP_DEPTH)

    sem_sources = [
        u for u, v, d in G.edges(data=True)
        if d.get("edge_type") in SEMANTIC_EDGE_TYPES
        and _is_usable(u, corpus) and _is_usable(v, corpus)
    ]
    # Если нет семантических рёбер — fallback на все рёбра
    if not sem_sources:
        sem_sources = [
            u for u, v, _ in G.edges()
            if _is_usable(u, corpus) and _is_usable(v, corpus)
        ]
    if not sem_sources:
        return []

    results, seen, attempts = [], set(), 0
    while len(results) < n and attempts < n * 50:
        attempts += 1
        depth = random.randint(min_d, max_d)
        node  = random.choice(sem_sources)
        chain = [node]
        for _ in range(depth - 1):
            neighbors = [
                v for _, v, d in G.out_edges(node, data=True)
                if _is_usable(v, corpus) and v not in chain
            ]
            if not neighbors:
                break
            node = random.choice(neighbors)
            chain.append(node)
        key = tuple(chain)
        if len(chain) >= min_d and key not in seen:
            seen.add(key)
            edges = []
            for i in range(len(chain) - 1):
                d = G.get_edge_data(chain[i], chain[i + 1], {})
                edges.append({
                    "source": chain[i],
                    "target": chain[i + 1],
                    "edge_type": d.get("edge_type", "ссылается_на"),
                    "explanation": d.get("explanation", ""),
                })
            results.append({"norm_ids": chain, "edges": edges})
    return results


def select_conflict_resolution(
    G: nx.DiGraph, corpus: dict, n: int, target_hop_group: str | None = None
) -> list[dict]:
    """
    shallow: пары с конфликтным ребром (исключает/приоритет).
    medium/deep: конфликтное ребро как seed, цепочка продолжается через DFS.
    """
    conflict_edges = [
        (u, v, d)
        for u, v, d in G.edges(data=True)
        if d.get("edge_type") in CONFLICT_EDGE_TYPES
        and _is_usable(u, corpus) and _is_usable(v, corpus)
    ]
    if not conflict_edges:
        return []

    if target_hop_group is None or target_hop_group == "shallow":
        sample = random.sample(conflict_edges, min(n, len(conflict_edges)))
        return [
            {
                "norm_ids": [u, v],
                "edges": [{
                    "source": u, "target": v,
                    "edge_type": d.get("edge_type"),
                    "explanation": d.get("explanation", ""),
                }],
            }
            for u, v, d in sample
        ]

    # medium/deep: стартуем с конфликтной пары, продолжаем DFS
    min_d, max_d = HOP_GROUP_DEPTHS[target_hop_group]
    results, seen, attempts = [], set(), 0
    while len(results) < n and attempts < n * 50:
        attempts += 1
        u, v, d = random.choice(conflict_edges)
        chain = [u, v]
        edges = [{
            "source": u, "target": v,
            "edge_type": d.get("edge_type"),
            "explanation": d.get("explanation", ""),
        }]
        node = v
        target_depth = random.randint(min_d, max_d)
        while len(chain) < target_depth:
            neighbors = [
                nb for _, nb in G.out_edges(node)
                if _is_usable(nb, corpus) and nb not in chain
            ]
            if not neighbors:
                break
            next_node = random.choice(neighbors)
            nd = G.get_edge_data(node, next_node, {})
            edges.append({
                "source": node, "target": next_node,
                "edge_type": nd.get("edge_type", "ссылается_на"),
                "explanation": nd.get("explanation", ""),
            })
            chain.append(next_node)
            node = next_node
        key = tuple(chain)
        if len(chain) >= min_d and key not in seen:
            seen.add(key)
            results.append({"norm_ids": chain, "edges": edges})
    return results


def select_temporal_validity(
    G: nx.DiGraph, corpus: dict, n: int, target_hop_group: str | None = None
) -> list[dict]:
    """
    shallow: одиночные нормы с known valid_from.
    medium/deep: цепочки, в которых ≥2 нормы имеют valid_from.
    """
    if target_hop_group is None or target_hop_group == "shallow":
        candidates = [
            nid for nid in G.nodes()
            if _is_usable(nid, corpus) and corpus.get(nid, {}).get("valid_from")
        ]
        sample = random.sample(candidates, min(n, len(candidates)))
        return [{"norm_ids": [nid]} for nid in sample]

    min_d, max_d = HOP_GROUP_DEPTHS[target_hop_group]

    # Стартуем только от нод с valid_from, у которых есть usable out-соседи
    temporal_starts = [
        nid for nid in G.nodes()
        if _is_usable(nid, corpus)
        and corpus.get(nid, {}).get("valid_from")
        and any(_is_usable(v, corpus) for _, v in G.out_edges(nid))
    ]
    if not temporal_starts:
        return []

    def has_enough_dates(chain: list[str]) -> bool:
        return sum(1 for nid in chain if corpus.get(nid, {}).get("valid_from")) >= 2

    return _dfs_chains(G, corpus, temporal_starts, min_d, max_d, n, filter_fn=has_enough_dates)


SELECTORS = {
    "issue_spotting":      select_issue_spotting,
    "rule_selection":      select_rule_selection,
    "conflict_resolution": select_conflict_resolution,
    "temporal_validity":   select_temporal_validity,
}


# ---------------------------------------------------------------------------
# Промпты
# ---------------------------------------------------------------------------

_SYSTEM_BASE = f"""Ты — автор юридических задач для исследовательского бенчмарка оценки LLM.
Твои задачи тестируют правоприменительное рассуждение по российскому праву.

Требования к задаче:
- Фабула: реалистичная жизненная ситуация (3–5 предложений), факты конкретны
- Вопрос: открытый, требует развёрнутого правового обоснования
- Ответ: эталонный ответ СТРОГО в формате:
  {EXPECTED_ANSWER_FORMAT}
  Только итоговый вывод — без рассуждений вслух, без «уточним», без самоисправлений.
- gold_chain: пошаговая цепочка рассуждений от фабулы к выводу

Верни ТОЛЬКО валидный JSON, без пояснений вне JSON."""


def _prompt_issue_spotting(chain: dict, corpus: dict) -> tuple[str, str]:
    norm_ids = chain["norm_ids"]

    if len(norm_ids) == 1:
        nid = norm_ids[0]
        norm = _norm_block(nid, corpus)
        user = f"""Дана норма российского права:

{norm}

Придумай ситуацию, в которой возникает правовой вопрос, регулируемый именно этой статьёй.

Верни JSON:
{{
  "fabula": "...",
  "question": "Каковы правовые последствия / какие права имеет X / законно ли действие Y?",
  "answer": "Развёрнутый правовой ответ...",
  "gold_chain": [
    {{"step": 1, "norm_id": "{nid}", "reasoning": "Эта норма применима, потому что..."}},
    {{"step": 2, "norm_id": null, "conclusion": "Итог: ..."}}
  ]
}}"""
        return _SYSTEM_BASE, user

    # medium/deep: несколько норм, каждая порождает отдельный правовой вопрос
    norms_text = "\n\n".join(_norm_block(nid, corpus) for nid in norm_ids)
    branches = list(dict.fromkeys(
        corpus.get(nid, {}).get("branch_of_law", "")
        for nid in norm_ids
        if corpus.get(nid, {}).get("branch_of_law")
    ))
    gold_steps = "\n    ".join(
        f'{{"step": {i + 1}, "norm_id": "{nid}", "reasoning": "Эта норма применима, потому что..."}}'
        for i, nid in enumerate(norm_ids)
    )
    branches_str = ", ".join(branches) if branches else "различных отраслей"
    user = f"""Даны нормы российского права из {branches_str}:

{norms_text}

Придумай сложную ситуацию, в которой ВСЕ эти нормы становятся релевантными одновременно, каждая порождая отдельный самостоятельный правовой вопрос.
Задача должна требовать идентификации ВСЕХ правовых проблем, а не только одной.

Верни JSON:
{{
  "fabula": "Многогранная ситуация с несколькими правовыми аспектами...",
  "question": "Какие правовые вопросы возникают в данной ситуации и какие нормы их регулируют?",
  "answer": "Развёрнутый ответ с перечислением всех правовых вопросов и применимых норм...",
  "gold_chain": [
    {gold_steps},
    {{"step": {len(norm_ids) + 1}, "norm_id": null, "conclusion": "Итог: в ситуации возникают следующие правовые вопросы: ..."}}
  ]
}}"""
    return _SYSTEM_BASE, user


def _prompt_rule_selection(chain: dict, corpus: dict) -> tuple[str, str]:
    norms_text = "\n\n".join(_norm_block(nid, corpus) for nid in chain["norm_ids"])
    edges_text = "; ".join(
        f"{e['source']} →[{e['edge_type']}]→ {e['target']}"
        + (f" ({e['explanation']})" if e.get("explanation") else "")
        for e in chain.get("edges", [])
    )
    gold_steps = "\n    ".join(
        f'{{"step": {i+1}, "norm_id": "{nid}", "reasoning": "..."}}'
        for i, nid in enumerate(chain["norm_ids"])
    )
    user = f"""Дана цепочка взаимосвязанных норм российского права:

{norms_text}

Связи между нормами: {edges_text}

Придумай ситуацию, которую можно правильно разрешить, только применив ВСЕ эти нормы последовательно.

Верни JSON:
{{
  "fabula": "...",
  "question": "Какие нормы применяются и каков правовой итог?",
  "answer": "Развёрнутый ответ с обоснованием через все нормы...",
  "gold_chain": [
    {gold_steps},
    {{"step": {len(chain["norm_ids"])+1}, "norm_id": null, "conclusion": "Итог: ..."}}
  ]
}}"""
    return _SYSTEM_BASE, user


def _prompt_conflict_resolution(chain: dict, corpus: dict) -> tuple[str, str]:
    norm_ids = chain["norm_ids"]
    edge = chain["edges"][0]
    et, expl = edge["edge_type"], edge.get("explanation", "")
    norm_a, norm_b = norm_ids[0], norm_ids[1]

    if len(norm_ids) == 2:
        user = f"""Даны две конфликтующие нормы российского права:

Норма A:
{_norm_block(norm_a, corpus)}

Норма B:
{_norm_block(norm_b, corpus)}

Тип конфликта: {et}{(' — ' + expl) if expl else ''}

Придумай ситуацию, в которой обе нормы претендуют на применение, но противоречат друг другу.
Задача должна требовать разрешения коллизии с чётким обоснованием.

Верни JSON:
{{
  "fabula": "...",
  "question": "Какая норма применяется в данной ситуации и почему?",
  "answer": "Развёрнутый ответ с разрешением коллизии...",
  "gold_chain": [
    {{"step": 1, "norm_id": "{norm_a}", "reasoning": "Норма A применима, потому что..."}},
    {{"step": 2, "norm_id": "{norm_b}", "reasoning": "Норма B конкурирует, потому что..."}},
    {{"step": 3, "norm_id": null, "conclusion": "Коллизия разрешается в пользу ... потому что ..."}}
  ]
}}"""
        return _SYSTEM_BASE, user

    # medium/deep: конфликт как отправная точка, цепочка для разрешения
    norms_text = "\n\n".join(_norm_block(nid, corpus) for nid in norm_ids)
    edges_text = "; ".join(
        f"{e['source']} →[{e['edge_type']}]→ {e['target']}"
        + (f" ({e['explanation']})" if e.get("explanation") else "")
        for e in chain["edges"]
    )
    gold_steps = "\n    ".join(
        f'{{"step": {i + 1}, "norm_id": "{nid}", "reasoning": "..."}}'
        for i, nid in enumerate(norm_ids)
    )
    user = f"""Даны нормы российского права, связанные конфликтом и цепочкой применения:

Конфликтующие нормы:
Норма A: {_norm_block(norm_a, corpus)}

Норма B: {_norm_block(norm_b, corpus)}

Дополнительные нормы цепочки:
{"".join(_norm_block(nid, corpus) + chr(10) for nid in norm_ids[2:])}
Связи: {edges_text}
Тип конфликта между A и B: {et}{(' — ' + expl) if expl else ''}

Придумай ситуацию, в которой нормы A и B конфликтуют, а разрешение этого конфликта требует последовательного применения всей цепочки норм.

Верни JSON:
{{
  "fabula": "...",
  "question": "Какая норма применяется в данной ситуации и каков полный правовой итог?",
  "answer": "Развёрнутый ответ с разрешением коллизии через все нормы цепочки...",
  "gold_chain": [
    {gold_steps},
    {{"step": {len(norm_ids) + 1}, "norm_id": null, "conclusion": "Коллизия разрешается в пользу ... потому что ..."}}
  ]
}}"""
    return _SYSTEM_BASE, user


def _prompt_temporal_validity(chain: dict, corpus: dict) -> tuple[str, str]:
    norm_ids = chain["norm_ids"]

    if len(norm_ids) == 1:
        nid = norm_ids[0]
        art = corpus.get(nid, {})
        valid_from = art.get("valid_from", "неизвестна")
        user = f"""Дана норма российского права, действующая с {valid_from}:

{_norm_block(nid, corpus)}

Придумай ситуацию, в которой важно определить, действовала ли эта редакция нормы на момент события.
Дата события в фабуле должна быть КОНКРЕТНОЙ (формат ДД.ММ.ГГГГ) и значимой относительно {valid_from}.

Верни JSON:
{{
  "fabula": "Ситуация с конкретной датой события...",
  "question": "Какая редакция нормы действовала на момент события и каковы последствия?",
  "answer": "Развёрнутый ответ с учётом даты вступления нормы в силу...",
  "gold_chain": [
    {{"step": 1, "norm_id": "{nid}", "reasoning": "Дата события ... относительно даты {valid_from} означает..."}},
    {{"step": 2, "norm_id": null, "conclusion": "Итог: ..."}}
  ]
}}"""
        return _SYSTEM_BASE, user

    # medium/deep: цепочка норм с разными датами редакций
    norms_text = "\n\n".join(_norm_block(nid, corpus) for nid in norm_ids)
    dates_info = "; ".join(
        f"{nid}: ред. с {corpus.get(nid, {}).get('valid_from', 'неизвестна')}"
        for nid in norm_ids
        if corpus.get(nid, {}).get("valid_from")
    )
    gold_steps_list = []
    for i, nid in enumerate(norm_ids):
        vf = corpus.get(nid, {}).get("valid_from", "неизвестна")
        gold_steps_list.append(
            f'{{"step": {i + 1}, "norm_id": "{nid}", "reasoning": "На дату события норма {nid} (ред. с {vf}) ..."}}'
        )
    gold_steps = "\n    ".join(gold_steps_list)
    user = f"""Дана цепочка взаимосвязанных норм российского права с разными датами редакций:

{norms_text}

Даты редакций: {dates_info}

Придумай ситуацию с КОНКРЕТНОЙ датой события, в которой правильное решение зависит от того, какие редакции всех этих норм действовали на этот момент.
Дата события должна быть выбрана так, чтобы для части норм она попадала до даты редакции, для части — после.

Верни JSON:
{{
  "fabula": "Ситуация с конкретной датой события и несколькими нормами...",
  "question": "Какие редакции норм действовали на момент события и каков правовой итог их совместного применения?",
  "answer": "Развёрнутый ответ с анализом дат каждой нормы и итоговым выводом...",
  "gold_chain": [
    {gold_steps},
    {{"step": {len(norm_ids) + 1}, "norm_id": null, "conclusion": "Итог: ..."}}
  ]
}}"""
    return _SYSTEM_BASE, user


PROMPT_BUILDERS = {
    "issue_spotting":      _prompt_issue_spotting,
    "rule_selection":      _prompt_rule_selection,
    "conflict_resolution": _prompt_conflict_resolution,
    "temporal_validity":   _prompt_temporal_validity,
}


# ---------------------------------------------------------------------------
# LLM-вызов
# ---------------------------------------------------------------------------

def call_llm(system: str, user: str, client, max_tokens: int = 1500) -> dict | None:
    for attempt in range(3):
        try:
            from openai import OpenAI
            resp = client.chat.completions.create(
                model=GENERATOR_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                temperature=0.7,
                max_tokens=max_tokens,
                extra_body={"cache_prompt": True},
            )
            raw = resp.choices[0].message.content.strip()
            return _parse_llm_json(raw)
        except Exception as e:
            if attempt == 2:
                print(f"  [LLM ERROR] {e}", file=sys.stderr)
    return None


def _max_tokens_for_chain(norm_count: int) -> int:
    """1500 / 2500 / 3500 в зависимости от глубины цепочки."""
    if norm_count <= 2:
        return 1500
    if norm_count <= 4:
        return 2500
    return 3500


# ---------------------------------------------------------------------------
# Генерация по типу
# ---------------------------------------------------------------------------

def generate_for_type(
    task_type: str,
    corpus: dict,
    G: nx.DiGraph,
    client,
    n: int,
    target_hop_group: str | None = None,
) -> list[dict]:
    chains = SELECTORS[task_type](G, corpus, n, target_hop_group)
    if not chains:
        return []

    tasks = []
    prompt_builder = PROMPT_BUILDERS[task_type]
    label = f"{task_type}/{target_hop_group or 'any'}"

    for chain in tqdm(chains, desc=label, unit="задач"):
        system, user = prompt_builder(chain, corpus)
        max_tokens = _max_tokens_for_chain(len(chain["norm_ids"]))
        result = call_llm(system, user, client, max_tokens=max_tokens)
        if result is None:
            continue
        if not all(result.get(k) for k in ("fabula", "question", "answer", "gold_chain")):
            continue
        pair = _build_task(
            task_type=task_type,
            norm_ids=chain["norm_ids"],
            corpus=corpus,
            llm_output=result,
            edge_meta=chain.get("edges"),
        )
        tasks.extend(pair)

    return tasks


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _load_existing_cells(path: Path) -> dict[tuple[str, str], int]:
    """Читает out-файл и возвращает число уникальных фабул на ячейку (type, hop_group)."""
    if not path.exists():
        return {}
    counts: dict[tuple[str, str], set] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                t = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (t.get("type", ""), t.get("hop_group", ""))
            fabula = t.get("fabula", "")
            if key[0] and key[1] and fabula:
                counts.setdefault(key, set()).add(fabula)
    return {k: len(v) for k, v in counts.items()}


def _print_matrix(matrix: dict[tuple[str, str], int]) -> None:
    """Печатает матрицу уникальных фабул 4 типа × 3 hop-группы."""
    col_w = 9
    header = f"{'':26s}" + "".join(f"{g:>{col_w}}" for g in HOP_GROUPS)
    print(header)
    print("-" * (26 + col_w * len(HOP_GROUPS)))
    for tt in TASK_TYPES:
        row = f"{tt:26s}"
        for hg in HOP_GROUPS:
            cnt = matrix.get((tt, hg), 0)
            row += f"{cnt:>{col_w}}"
        print(row)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Генерация задач MHGB")
    parser.add_argument("--type", default="all", choices=TASK_TYPES + ["all"],
                        help="Тип задач (default: all)")
    parser.add_argument("--target-per-cell", type=int, default=25,
                        help="Целевое число уникальных фабул на ячейку матрицы (default: 25)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=str(DATA_DIR / "tasks_raw.jsonl"))
    args = parser.parse_args()

    random.seed(args.seed)

    print("Загрузка данных...")
    corpus, G = load_data()
    print(f"  Корпус: {len(corpus)} статей, граф: {G.number_of_nodes()} нод, {G.number_of_edges()} рёбер")

    from openai import OpenAI
    client = OpenAI(base_url=GENERATOR_API_BASE, api_key="mhgb")

    types_to_run = TASK_TYPES if args.type == "all" else [args.type]
    all_tasks: list[dict] = []
    # Матрица: (type, hop_group) → число уникальных фабул
    matrix: dict[tuple[str, str], int] = {}

    out_path = Path(args.out)
    out_path.parent.mkdir(exist_ok=True)

    existing = _load_existing_cells(out_path)
    if existing:
        total_existing = sum(existing.values())
        print(f"Найден существующий файл: {total_existing} фабул по {len(existing)} ячейкам — уже выполненные будут пропущены")

    # Порядок генерации: issue_spotting medium/deep — последними
    GENERATION_ORDER = [
        ("issue_spotting",      "shallow"),
        ("rule_selection",      "shallow"),
        ("rule_selection",      "medium"),
        ("rule_selection",      "deep"),
        ("conflict_resolution", "shallow"),
        ("conflict_resolution", "medium"),
        ("conflict_resolution", "deep"),
        ("temporal_validity",   "shallow"),
        ("temporal_validity",   "medium"),
        ("temporal_validity",   "deep"),
        ("issue_spotting",      "medium"),
        ("issue_spotting",      "deep"),
    ]

    with open(out_path, "a" if out_path.exists() else "w", encoding="utf-8") as f:
        for task_type, hop_group in GENERATION_ORDER:
            if task_type not in types_to_run:
                continue

            already = existing.get((task_type, hop_group), 0)
            if already >= args.target_per_cell:
                print(f"\n→ {task_type} / {hop_group} — пропускаем ({already} фабул уже есть)")
                matrix[(task_type, hop_group)] = already
                continue

            needed = args.target_per_cell - already
            print(f"\n→ {task_type} / {hop_group} (target={args.target_per_cell}, уже есть: {already}, нужно: {needed})")
            tasks = generate_for_type(
                task_type, corpus, G, client,
                n=needed,
                target_hop_group=hop_group,
            )
            for t in tasks:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
            f.flush()
            all_tasks.extend(tasks)
            fabulas = len(tasks) // 2  # closed+open → 2 задачи на фабулу
            matrix[(task_type, hop_group)] = already + fabulas
            print(f"  Сгенерировано: {fabulas} фабул × 2 режима = {len(tasks)} задач")

    print(f"\nИтого: {len(all_tasks)} записей → {out_path}")
    print(f"  closed: {sum(1 for t in all_tasks if t['mode'] == 'closed')}")
    print(f"  open:   {sum(1 for t in all_tasks if t['mode'] == 'open')}")
    print("\nМатрица (уникальных фабул):")
    _print_matrix(matrix)


if __name__ == "__main__":
    main()
