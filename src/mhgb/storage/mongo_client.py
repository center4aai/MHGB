"""
Обёртка над pymongo для хранения версий экспериментов MHGB.

Использование:
    from mhgb.storage.mongo_client import MongoStorage
    storage = MongoStorage()
    storage.ensure_collections()

CLI-проверка подключения:
    uv run python -m mhgb.storage.mongo_client --check
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

# ---------------------------------------------------------------------------
# Конфиг
# ---------------------------------------------------------------------------

from mhgb.config import cfg
from mhgb.logging_setup import logger

DEFAULT_URI = cfg.mongodb_uri
DEFAULT_DB  = "mhgb"

# Названия коллекций
COLLECTIONS = {
    "corpus_versions",
    "graph_versions",
    "task_batches",
    "experiments",
    "model_responses",
    "evaluations",
    "validation_logs",
    "llm_classification_cache",
}

# ---------------------------------------------------------------------------
# Индексы
# ---------------------------------------------------------------------------

# Формат: {collection: [(keys, options), ...]}
INDEXES: dict[str, list[tuple]] = {
    "corpus_versions": [
        (([("tag", ASCENDING)],  {"unique": True})),
        (([("created_at", DESCENDING)], {})),
    ],
    "graph_versions": [
        (([("tag", ASCENDING)],  {"unique": True})),
        (([("corpus_version_id", ASCENDING)], {})),
    ],
    "task_batches": [
        (([("tag", ASCENDING)],  {"unique": True})),
        (([("graph_version_id", ASCENDING)], {})),
    ],
    "experiments": [
        (([("name", ASCENDING)], {"unique": True})),
        (([("model_name", ASCENDING)], {})),
        (([("status", ASCENDING)], {})),
    ],
    "model_responses": [
        (([("experiment_id", ASCENDING), ("task_id", ASCENDING)], {})),
        (([("model_name", ASCENDING)], {})),
        (([("task_id", ASCENDING)], {})),
    ],
    "evaluations": [
        (([("experiment_id", ASCENDING), ("task_id", ASCENDING)], {})),
        (([("model_response_id", ASCENDING)], {})),
    ],
    "validation_logs": [
        (([("task_id", ASCENDING)], {})),
        (([("task_batch_id", ASCENDING)], {})),
        (([("validator", ASCENDING), ("is_valid", ASCENDING)], {})),
    ],
    "llm_classification_cache": [
        (([("cache_key", ASCENDING)], {"unique": True})),
        (([("pair_id", ASCENDING)], {})),
        (([("prompt_version", ASCENDING)], {})),
    ],
}


# ---------------------------------------------------------------------------
# MongoStorage
# ---------------------------------------------------------------------------

class MongoStorage:
    """Обёртка над pymongo с готовыми коллекциями и индексами MHGB."""

    def __init__(self, uri: str = DEFAULT_URI, db_name: str = DEFAULT_DB) -> None:
        self._client: MongoClient = MongoClient(uri, serverSelectionTimeoutMS=5000)
        self.db = self._client[db_name]

    def check_connection(self) -> bool:
        """Возвращает True если MongoDB доступна."""
        try:
            self._client.admin.command("ping")
            return True
        except (ConnectionFailure, ServerSelectionTimeoutError):
            return False

    def ensure_collections(self) -> dict[str, int]:
        """
        Создаёт коллекции и индексы если их ещё нет.
        Возвращает словарь {collection: кол-во созданных индексов}.
        """
        result: dict[str, int] = {}
        existing = set(self.db.list_collection_names())

        for col_name in COLLECTIONS:
            if col_name not in existing:
                self.db.create_collection(col_name)

            collection = self.db[col_name]
            created = 0
            for keys, options in INDEXES.get(col_name, []):
                try:
                    collection.create_index(keys, **options)
                    created += 1
                except Exception:
                    pass
            result[col_name] = created

        return result

    def insert(self, collection_name: str, document: dict[str, Any]) -> str:
        """Вставляет документ, возвращает строку ObjectId."""
        res = self.db[collection_name].insert_one(document)
        return str(res.inserted_id)

    def find_one(self, collection_name: str, query: dict) -> dict | None:
        return self.db[collection_name].find_one(query)

    def find(self, collection_name: str, query: dict) -> list[dict]:
        return list(self.db[collection_name].find(query))

    def latest_corpus_version_id(self) -> str | None:
        """Возвращает doc_id последней версии корпуса или None."""
        doc = self.db["corpus_versions"].find_one(
            {}, sort=[("created_at", DESCENDING)]
        )
        return doc.get("doc_id") if doc else None

    def save_graph_version(
        self,
        tag: str,
        node_count: int,
        edge_count: int,
        edge_type_counts: dict,
        params: dict,
        file_path: str | None = None,
        corpus_version_id: str | None = None,
    ) -> str:
        """
        Сохраняет метаданные версии графа в graph_versions.
        Возвращает строку ObjectId вставленного документа.
        При дубликате тега — поднимает DuplicateKeyError.
        """
        from mhgb.storage.schemas import GraphVersion
        if corpus_version_id is None:
            corpus_version_id = self.latest_corpus_version_id() or "unknown"
        doc = GraphVersion(
            tag=tag,
            corpus_version_id=corpus_version_id,
            node_count=node_count,
            edge_count=edge_count,
            edge_type_counts=edge_type_counts,
            params=params,
            file_path=file_path,
        ).model_dump()
        return self.insert("graph_versions", doc)

    def list_graph_versions(self) -> list[dict]:
        """Список всех версий графа, отсортированных по дате (новые первыми)."""
        return list(
            self.db["graph_versions"].find(
                {}, sort=[("created_at", DESCENDING)]
            )
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "MongoStorage":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(description="MHGB MongoDB утилита")
    parser.add_argument("--check", action="store_true",
                        help="Проверить подключение и создать коллекции")
    parser.add_argument("--uri", default=DEFAULT_URI,
                        help=f"MongoDB URI (default: {DEFAULT_URI})")
    args = parser.parse_args()

    if not args.check:
        parser.print_help()
        return

    print(f"Подключение к {args.uri} ...")
    storage = MongoStorage(uri=args.uri)

    if not storage.check_connection():
        print("  [ОШИБКА] MongoDB недоступна. Убедитесь, что сервер запущен.", file=sys.stderr)
        sys.exit(1)

    print("  [OK] Соединение установлено")
    created = storage.ensure_collections()
    print("\nКоллекции и индексы:")
    for col, n in sorted(created.items()):
        print(f"  {col:25s} — {n} индекс(ов) создано/проверено")

    storage.close()
    print("\nГотово.")


if __name__ == "__main__":
    _cli()
