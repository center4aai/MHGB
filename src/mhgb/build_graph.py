"""
Шаг 2: Построение графа правовых норм (Legal Knowledge Graph).

Два прохода:
1. Regex-извлечение явных ссылок между статьями → рёбра типа 'ссылается_на'
2. LLM-извлечение семантических рёбер для пар статей с явными связями
   → типы: исключает | дополняет | приоритет | применяется_к

Результат:
  data/graph.json       — все ноды + рёбра (для загрузки в NetworkX)
  data/graph_stats.txt  — краткая статистика
"""

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from collections import defaultdict

import networkx as nx
from openai import OpenAI
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Конфиг
# ---------------------------------------------------------------------------

from mhgb.config import cfg
from mhgb.logging_setup import logger

REPO_ROOT = cfg.repo_root
DATA_DIR  = cfg.data_dir

GENERATOR_API_BASE = cfg.generator_api_base
GENERATOR_MODEL    = cfg.generator_model

PROMPTS_DIR    = REPO_ROOT / "prompts"
PROMPT_VERSION = "v1"  # bump при изменении промпта — инвалидирует кэш

# Короткие алиасы кодексов для нормализации ссылок.
# Порядок важен: длинные/точные — раньше, чтобы короткие не перехватили первыми.
LAW_ALIASES: dict[str, str | None] = {
    "настоящего кодекса": None,  # тот же кодекс, что источник
    # полные названия в родительном падеже
    "кодекса об административных правонарушениях": "КоАП",
    "градостроительного кодекса": "ГрК",
    "уголовного кодекса": "УК",
    "налогового кодекса": "НК",
    "земельного кодекса": "ЗК",
    "трудового кодекса": "ТК",
    "гражданского кодекса": "ГК",
    "семейного кодекса": "СК",
    "жилищного кодекса": "ЖК",
    "конституции российской федерации": "КРФ",
    "конституции рф": "КРФ",
    "конституции": "КРФ",
    # короткие формы с «рф»
    "коап рф": "КоАП",
    "грк рф": "ГрК",
    "ук рф": "УК",
    "нк рф": "НК",
    "зк рф": "ЗК",
    "тк рф": "ТК",
    "гк рф": "ГК",
    "ск рф": "СК",
    "жк рф": "ЖК",
    # короткие формы без «рф»
    "коап": "КоАП",
    "грк": "ГрК",
    "ук": "УК",
    "нк": "НК",
    "зк": "ЗК",
    "тк": "ТК",
    "гк": "ГК",
    "ск": "СК",
    "жк": "ЖК",
}

EDGE_TYPES = {"исключает", "дополняет", "приоритет", "применяется_к", "ссылается_на"}


def _load_system_prompt() -> str:
    prompt_file = PROMPTS_DIR / "edge_classification.md"
    if prompt_file.exists():
        return prompt_file.read_text(encoding="utf-8").strip()
    return (
        "Ты — эксперт по российскому праву. Твоя задача — определить тип правовой связи между двумя статьями закона.\n\n"
        "Возможные типы рёбер:\n"
        "- исключает: статья A запрещает или делает невозможным применение статьи B в определённых случаях\n"
        "- дополняет: статья A добавляет условия, права или обязанности к статье B\n"
        "- приоритет: статья A имеет приоритет над статьёй B (как специальная норма над общей, или более новая над старой)\n"
        "- применяется_к: статья A устанавливает правила, применимые к ситуациям, регулируемым статьёй B\n"
        "- ссылается_на: нейтральная ссылка без чёткой семантики\n\n"
        'Верни ТОЛЬКО JSON без пояснений:\n{"edge_type": "<тип>", "direction": "<A→B или B→A>", "explanation": "<одно предложение>"}\n\n'
        'Если связи нет — верни: {"edge_type": null, "direction": null, "explanation": "нет значимой правовой связи"}'
    )


SYSTEM_PROMPT = _load_system_prompt()

# ---------------------------------------------------------------------------
# Regex-извлечение явных ссылок
# ---------------------------------------------------------------------------

# Паттерн: "статьёй 261", "статей 81 и 82", "статьи 261.1"
# с опциональным указанием кодекса
_ART_NUM = r"[\d]+(?:\.\d+)?"
# Разделители: запятая, тире, "и" (союз в списках типа "статьями 257 и 258")
_ART_SEP   = r"(?:\s*(?:[,–-]|и)\s*)"
_ART_RANGE = rf"{_ART_NUM}(?:{_ART_SEP}{_ART_NUM})*"
_KODEX = (
    r"(?:"
    r"настоящего\s+[Кк]одекса"
    r"|[Кк]одекса\s+об\s+административных\s+правонарушениях(?:\s+(?:РФ|Российской\s+Федерации))?"
    r"|(?:Трудового|Гражданского|Семейного|Жилищного|Уголовного|Налогового|Земельного|Градостроительного)"
    r"\s+кодекса(?:\s+(?:РФ|Российской\s+Федерации))?"
    r"|Конституции(?:\s+(?:РФ|Российской\s+Федерации))?"
    r"|КоАП(?:\s+РФ)?"
    r"|(?:ТК|ГК|СК|ЖК|УК|НК|ЗК|ГрК)(?:\s+РФ)?"
    r")?"
)

# Основа "стат" покрывает все падежи:
#   статей(р.мн), статья(им), статьи(р.ед/им.мн), статье(д/пр),
#   статью(в), статьей(тв.ед), статьям(д.мн), статьями(тв.мн), статьях(пр.мн)
REF_PATTERN = re.compile(
    rf"стат(?:ей|ьями?|ьях|ьей|ьи|ье|ью|ья)\s+({_ART_RANGE})\s*{_KODEX}",
    re.IGNORECASE,
)


def extract_article_nums(raw: str) -> list[str]:
    """Из '81 и 82' или '81, 82, 83' вытаскивает ['81', '82', '83']."""
    return re.findall(r"\d+(?:\.\d+)?", raw)


def extract_explicit_refs(article: dict) -> list[tuple[str, str]]:
    """
    Возвращает список (source_id, target_id) для явных ссылок в тексте статьи.
    target_id формируется как '{law_short}_{article_num}' — если кодекс указан,
    иначе — тот же кодекс, что и источник.
    """
    edges = []
    law_short = article["law_short"]

    for para in article["paragraphs"]:
        # Нормализуем ё→е: в НПА ё и е взаимозаменяемы, regex работает только с е
        para_norm = para.replace("ё", "е").replace("Ё", "Е")
        for m in REF_PATTERN.finditer(para_norm):
            nums_raw = m.group(1)
            # Определяем кодекс-цель
            context = m.group(0).lower()
            target_law = law_short  # по умолчанию — тот же
            for alias, short in LAW_ALIASES.items():
                if alias in context and short is not None:
                    target_law = short
                    break

            for num in extract_article_nums(nums_raw):
                target_id = f"{target_law}_{num}"
                if target_id != article["id"]:
                    edges.append((article["id"], target_id))

    return list(set(edges))


# ---------------------------------------------------------------------------
# LLM-извлечение семантических рёбер
# ---------------------------------------------------------------------------

def build_llm_client() -> OpenAI:
    return OpenAI(
        base_url=GENERATOR_API_BASE,
        api_key="mhgb",  # llama.cpp не требует реального ключа
    )


def classify_edge_llm(
    client: OpenAI,
    art_a: dict,
    art_b: dict,
) -> dict | None:
    """
    Запрашивает LLM тип связи между art_a и art_b.
    Возвращает dict с ключами edge_type, direction, explanation или None при ошибке.
    """
    user_msg = (
        f"Статья A: {art_a['id']} — {art_a['title']}\n"
        f"Текст A: {art_a['text'][:600]}\n\n"
        f"Статья B: {art_b['id']} — {art_b['title']}\n"
        f"Текст B: {art_b['text'][:600]}"
    )

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=GENERATOR_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=200,
                extra_body={"cache_prompt": True},
            )
            raw = resp.choices[0].message.content.strip()
            # Убираем <think>...</think> блоки (Qwen3 thinking mode)
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            # Извлекаем JSON даже если есть лишний текст
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                return json.loads(m.group(0))
        except Exception as e:
            if attempt == 2:
                print(f"  [LLM ERROR] {art_a['id']}↔{art_b['id']}: {e}", file=sys.stderr)
            time.sleep(1)
    return None


# ---------------------------------------------------------------------------
# MongoDB-кэш LLM-классификации
# ---------------------------------------------------------------------------

def _cache_key(id_a: str, id_b: str) -> str:
    raw = f"{id_a}|{id_b}|{PROMPT_VERSION}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _lookup_cache(storage, key: str) -> dict | None:
    """Возвращает кэшированный результат или None."""
    try:
        doc = storage.find_one("llm_classification_cache", {"cache_key": key})
        if doc:
            return doc.get("result")
    except Exception:
        pass
    return None


def _save_cache(storage, key: str, id_a: str, id_b: str, result: dict) -> None:
    """Сохраняет результат в кэш; при дубликате — молча пропускает."""
    try:
        from pymongo.errors import DuplicateKeyError
        doc = {
            "cache_key": key,
            "pair_id": f"{id_a}|{id_b}",
            "prompt_version": PROMPT_VERSION,
            "result": result,
        }
        storage.insert("llm_classification_cache", doc)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Построение графа
# ---------------------------------------------------------------------------

def build_graph(
    articles: list[dict],
    llm_classify: bool = True,
    max_pairs: int | None = None,
    storage=None,
) -> nx.DiGraph:
    G = nx.DiGraph()

    # Индекс статей по id
    idx = {a["id"]: a for a in articles}

    # Добавляем ноды
    for a in articles:
        G.add_node(
            a["id"],
            law=a["law_short"],
            article=a["article"],
            title=a["title"],
            valid_from=a["valid_from"],
            text_preview=a["text"][:200],
        )

    print("\n[1/2] Извлечение явных ссылок (regex)...")
    ref_edges = []
    for a in articles:
        for src, tgt in extract_explicit_refs(a):
            if tgt in idx:  # только ссылки на известные статьи
                ref_edges.append((src, tgt))

    # Дедупликация и добавление
    ref_edges = list(set(ref_edges))
    for src, tgt in ref_edges:
        G.add_edge(src, tgt, edge_type="ссылается_на", source="regex")

    print(f"    Найдено {len(ref_edges)} явных ссылок между известными статьями")

    if not llm_classify:
        return G

    client = build_llm_client()

    candidate_edges = ref_edges
    if max_pairs is not None:
        candidate_edges = candidate_edges[:max_pairs]

    print(f"\n[2/2] LLM-классификация рёбер ({len(candidate_edges)} пар)...")
    classified = 0
    cache_hits = 0
    semantic_counts = defaultdict(int)

    for src, tgt in tqdm(candidate_edges, desc="LLM edges", unit="пар"):
        key = _cache_key(src, tgt)
        result = _lookup_cache(storage, key) if storage else None

        if result is not None:
            cache_hits += 1
        else:
            result = classify_edge_llm(client, idx[src], idx[tgt])
            if result and storage:
                _save_cache(storage, key, src, tgt, result)

        if result and result.get("edge_type") and result["edge_type"] in EDGE_TYPES:
            et = result["edge_type"]
            edge_src, edge_tgt = (tgt, src) if result.get("direction") == "B→A" else (src, tgt)
            if not G.has_edge(edge_src, edge_tgt):
                G.add_edge(edge_src, edge_tgt)
            G[edge_src][edge_tgt]["edge_type"] = et
            G[edge_src][edge_tgt]["explanation"] = result.get("explanation", "")
            G[edge_src][edge_tgt]["source"] = "llm"
            semantic_counts[et] += 1
            classified += 1

    print(f"    Классифицировано: {classified}/{len(candidate_edges)} (кэш: {cache_hits})")
    for et, cnt in sorted(semantic_counts.items(), key=lambda x: -x[1]):
        print(f"      {et}: {cnt}")

    return G


# ---------------------------------------------------------------------------
# Сохранение / загрузка
# ---------------------------------------------------------------------------

def save_graph(G: nx.DiGraph, path: Path):
    data = nx.node_link_data(G)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nГраф сохранён → {path}")


def load_graph(path: Path) -> nx.DiGraph:
    data = json.loads(path.read_text(encoding="utf-8"))
    return nx.node_link_graph(data, directed=True)


def print_stats(G: nx.DiGraph):
    print(f"\n=== Статистика графа ===")
    print(f"  Нод:   {G.number_of_nodes()}")
    print(f"  Рёбер: {G.number_of_edges()}")

    edge_type_counts = defaultdict(int)
    for _, _, d in G.edges(data=True):
        edge_type_counts[d.get("edge_type", "unknown")] += 1
    print("  Типы рёбер:")
    for et, cnt in sorted(edge_type_counts.items(), key=lambda x: -x[1]):
        print(f"    {et}: {cnt}")

    # Топ-10 статей по входящим ссылкам
    in_deg = sorted(G.in_degree(), key=lambda x: -x[1])[:10]
    print("\n  Топ-10 по входящим ссылкам:")
    for node, deg in in_deg:
        title = G.nodes[node].get("title", "")[:50]
        print(f"    {node:15s} ← {deg:3d}  ({title})")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    import argparse
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="MHGB: построение графа НПА")
    parser.add_argument("--no-llm", action="store_true",
                        help="Только regex-проход, без LLM-классификации")
    parser.add_argument("--max-pairs", type=int, default=None,
                        help="Максимальное число пар для LLM-классификации (для отладки)")
    parser.add_argument("--out", default=str(DATA_DIR / "graph.json"),
                        help="Путь для сохранения графа (default: data/graph.json)")
    parser.add_argument("--tag", default=None,
                        help="Тег версии графа для MongoDB (default: авто по времени)")
    args = parser.parse_args()

    corpus_path = DATA_DIR / "corpus.jsonl"
    graph_path  = Path(args.out)

    print(f"Загрузка корпуса: {corpus_path}")
    with open(corpus_path, encoding="utf-8") as f:
        articles = [json.loads(l) for l in f]
    print(f"  {len(articles)} статей")

    storage = None
    if not args.no_llm:
        try:
            from mhgb.storage.mongo_client import MongoStorage
            storage = MongoStorage()
            if storage.check_connection():
                storage.ensure_collections()
                print("  [MongoDB] кэш LLM-классификации подключён")
            else:
                storage = None
                print("  [MongoDB] недоступна — кэш отключён", file=sys.stderr)
        except Exception as e:
            storage = None
            print(f"  [MongoDB] ошибка подключения: {e} — кэш отключён", file=sys.stderr)

    G = build_graph(
        articles,
        llm_classify=not args.no_llm,
        max_pairs=args.max_pairs,
        storage=storage,
    )
    print_stats(G)
    save_graph(G, graph_path)

    if storage:
        edge_type_counts: dict[str, int] = defaultdict(int)
        for _, _, d in G.edges(data=True):
            edge_type_counts[d.get("edge_type", "unknown")] += 1

        tag = args.tag or (
            f"graph_{'regex' if args.no_llm else 'llm'}_"
            f"{time.strftime('%Y%m%d_%H%M%S')}"
        )
        params = {
            "no_llm": args.no_llm,
            "max_pairs": args.max_pairs,
            "model_name": GENERATOR_MODEL if not args.no_llm else None,
            "prompt_version": PROMPT_VERSION if not args.no_llm else None,
        }
        try:
            storage.save_graph_version(
                tag=tag,
                node_count=G.number_of_nodes(),
                edge_count=G.number_of_edges(),
                edge_type_counts=dict(edge_type_counts),
                params=params,
                file_path=str(graph_path),
            )
            print(f"  [MongoDB] версия графа сохранена: {tag}")
        except Exception as e:
            print(f"  [MongoDB] не удалось сохранить версию: {e}", file=sys.stderr)
        storage.close()


if __name__ == "__main__":
    main()
