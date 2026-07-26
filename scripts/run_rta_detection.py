"""
Шаг 4.7 — постфактум прогон RtA-детектора на результатах эксперимента.

Читает:
  reports/<experiment>/results.jsonl      — ответы модели
  data/tasks_raw.jsonl                    — фабулы и вопросы (джойн по task_id)

Пишет:
  reports/<experiment>/results_rta.jsonl  — исходные поля + RtA-поля

Пример запуска (trial на 10 записях):
  uv run python scripts/run_rta_detection.py \\
      --experiment yandexgpt_full600 \\
      --max-records 10 \\
      --dry-run

Полный прогон (только после явного одобрения):
  uv run python scripts/run_rta_detection.py \\
      --experiment yandexgpt_full600 \\
      --judge qwen3-27b-server
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from mhgb.eval.rta_detector import RtADetector
from mhgb.logging_setup import logger
from mhgb.models.llm_client import NativeOllamaClient, OpenAICompatibleClient

# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def load_tasks_index(tasks_path: Path) -> dict[str, dict]:
    """task_id → {fabula, question}"""
    index: dict[str, dict] = {}
    with tasks_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            tid = rec.get("task_id") or rec.get("id", "")
            index[tid] = {"fabula": rec.get("fabula", ""), "question": rec.get("question", "")}
    return index


def load_results(results_path: Path) -> list[dict]:
    records = []
    with results_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_done_set(rta_path: Path) -> set[str]:
    """Читает уже обработанные task_id из results_rta.jsonl (resumability)."""
    done: set[str] = set()
    if not rta_path.exists():
        return done
    with rta_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("is_rta") is not None:
                done.add(rec["task_id"])
    return done


def build_judge_client(judge_name: str):
    """Строит LLM-клиент по имени из models.yaml."""
    from mhgb.experiments.run_main_experiment import load_models_config
    cfg_path = ROOT / "configs" / "models.yaml"
    models = load_models_config(str(cfg_path))  # раскрывает ${VAR} в api_base
    cfg = next((m for m in models if m["name"] == judge_name), None)
    if cfg is None:
        raise ValueError(f"Модель '{judge_name}' не найдена в models.yaml")

    import os
    # Ollama-судья → native /api/chat, чтобы num_ctx из yaml реально применялся:
    # OpenAI-совместимый /v1 игнорирует num_ctx и грузит модель на серверном дефолте.
    if cfg.get("client_type") == "native_ollama":
        return NativeOllamaClient(
            api_base=cfg["api_base"],
            model=cfg["model_id"],
            num_ctx=cfg.get("num_ctx", 16384),
            max_tokens=cfg.get("max_tokens", 512),
            temperature=cfg.get("temperature", 0.0),
            timeout=cfg.get("timeout", 300.0),
            extra=cfg.get("extra_body"),
        )
    api_key = os.environ.get(cfg.get("api_key_env") or "", "dummy")
    return OpenAICompatibleClient(
        api_base=cfg["api_base"],
        api_key=api_key,
        model=cfg["model_id"],
        max_tokens=cfg.get("max_tokens", 512),
        temperature=cfg.get("temperature", 0.0),
        timeout=cfg.get("timeout", 120.0),
    )


# ---------------------------------------------------------------------------
# Главная функция
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Постфактум RtA-детекция.")
    parser.add_argument("--experiment", required=True,
                        help="Имя эксперимента (папка в reports/)")
    parser.add_argument("--tasks", default="data/tasks_public_120.jsonl",
                        help="Путь к tasks_raw.jsonl")
    parser.add_argument("--judge", default="qwen3-27b-server",
                        help="Имя модели-судьи из models.yaml")
    parser.add_argument("--max-records", type=int, default=None,
                        help="Обработать только N записей (для trial-прогона)")
    parser.add_argument("--max-workers", type=int, default=2,
                        help="Параллельные потоки (осторожно с локальным сервером)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Показать план без LLM-вызовов")
    args = parser.parse_args()

    exp_dir = ROOT / "reports" / args.experiment
    results_path = exp_dir / "results.jsonl"
    judge_slug = args.judge.replace("/", "_").replace(":", "-")
    rta_path = exp_dir / f"results_rta_{judge_slug}.jsonl"
    tasks_path = ROOT / args.tasks
    prompt_path = ROOT / "prompts" / "rta_judge.md"

    if not results_path.exists():
        logger.error(f"Не найден {results_path}")
        sys.exit(1)
    if not tasks_path.exists():
        logger.error(f"Не найден {tasks_path}")
        sys.exit(1)

    # Загрузка данных
    logger.info("Загружаю tasks_raw...")
    tasks_index = load_tasks_index(tasks_path)

    logger.info("Загружаю results.jsonl...")
    all_records = load_results(results_path)
    logger.info(f"Всего записей: {len(all_records)}")

    # Фильтрация: пропускаем записи с ошибкой (overflow, timeout и др.)
    # — они не являются реальными ответами модели, RtA-детекция на них некорректна
    valid_records = [r for r in all_records if "error" not in r]
    n_errors = len(all_records) - len(valid_records)
    if n_errors:
        logger.info(f"Пропущено записей с ошибкой (overflow/timeout/др.): {n_errors}")

    # Джойн: оставляем только те, для которых есть фабула
    joined: list[dict] = []
    missing_tasks = 0
    for rec in valid_records:
        tid = rec.get("task_id", "")
        task_info = tasks_index.get(tid)
        if task_info is None:
            missing_tasks += 1
            continue
        joined.append({**rec, **task_info})

    if missing_tasks:
        logger.warning(f"Не найдены фабулы для {missing_tasks} записей — пропущены")
    logger.info(f"После джойна: {len(joined)} записей")

    # Ограничение по --max-records
    if args.max_records:
        joined = joined[: args.max_records]
        logger.info(f"--max-records={args.max_records}: берём первые {len(joined)}")

    # Resumability
    done = load_done_set(rta_path)
    if done:
        logger.info(f"Уже обработано (resumability): {len(done)} записей")
    todo = [r for r in joined if r["task_id"] not in done]
    logger.info(f"К обработке: {len(todo)} записей")

    if args.dry_run:
        print("\n=== DRY-RUN ===")
        print(f"Эксперимент : {args.experiment}")
        print(f"Судья       : {args.judge}")
        print(f"Всего записей в results.jsonl : {len(all_records)}")
        print(f"Из них с ошибкой (пропущено)  : {n_errors}")
        print(f"После фильтра + джойна        : {len(joined)}")
        print(f"Уже в results_rta.jsonl       : {len(done)}")
        print(f"К обработке                   : {len(todo)}")
        print(f"Выходной файл                 : {rta_path}")
        if todo:
            ex = todo[0]
            print(f"\nПример первой записи:")
            print(f"  task_id  : {ex['task_id']}")
            print(f"  mode     : {ex.get('mode')}")
            print(f"  fabula   : {ex['fabula'][:80]}...")
            print(f"  question : {ex['question'][:80]}...")
            print(f"  response : {str(ex.get('raw_response',''))[:80]}...")
        return

    if not todo:
        logger.info("Все записи уже обработаны. Выход.")
        return

    # Построение детектора
    logger.info(f"Инициализирую судью: {args.judge}")
    judge_client = build_judge_client(args.judge)
    detector = RtADetector(judge_client, prompt_path)

    # Чанковый прогон с ИНКРЕМЕНТАЛЬНОЙ записью (resumable mid-run).
    # Системная хрупкость (DESIGN_LOG Урок 5): batch-в-конце терял весь проход
    # при крахе/резюме. Пишем чанками + flush → при обрыве уцелевшие чанки в файле,
    # рестарт пропускает их по done-set (load_done_set по task_id).
    CHUNK_SIZE = 25
    logger.info(f"Запускаю RtA-детекцию ({len(todo)} записей, чанки по {CHUNK_SIZE}, "
                f"{args.max_workers} потока, инкрементальная запись)...")
    rta_results = []
    with rta_path.open("a", encoding="utf-8") as out:
        for start in range(0, len(todo), CHUNK_SIZE):
            chunk = todo[start:start + CHUNK_SIZE]
            batch_records = [
                {
                    "fabula": r["fabula"],
                    "question": r["question"],
                    "model_response": r.get("raw_response", ""),
                    "task_id": r["task_id"],
                }
                for r in chunk
            ]
            chunk_results = detector.detect_batch(batch_records, max_workers=args.max_workers)
            rta_results.extend(chunk_results)
            rta_map = {r.task_id: r for r in chunk_results}
            for rec in chunk:
                rta = rta_map.get(rec["task_id"])
                enriched = {
                    **rec,
                    "is_rta": rta.is_rta if rta else None,
                    "rta_type": rta.refusal_type if rta else None,
                    "rta_topic": rta.refusal_topic if rta else None,
                    "rta_confidence": rta.confidence if rta else None,
                    "rta_explanation": rta.explanation if rta else None,
                }
                # fabula/question не нужны в rta-файле (джойн по task_id при необходимости)
                enriched.pop("fabula", None)
                enriched.pop("question", None)
                out.write(json.dumps(enriched, ensure_ascii=False) + "\n")
            out.flush()  # ЧЕКПОЙНТ после каждого чанка → resumable при крахе/резюме
            logger.info(f"  чанк {start // CHUNK_SIZE + 1}: +{len(chunk)}, "
                        f"всего {start + len(chunk)}/{len(todo)}")

    logger.info(f"Готово. Записано в {rta_path} ({len(rta_results)} записей).")

    # Краткая статистика
    n_rta = sum(1 for r in rta_results if r.is_rta)
    logger.info(f"RtA=True: {n_rta}/{len(rta_results)} ({100*n_rta/max(1,len(rta_results)):.1f}%)")
    if n_rta:
        from collections import Counter
        types = Counter(r.refusal_type for r in rta_results if r.is_rta)
        topics = Counter(r.refusal_topic for r in rta_results if r.is_rta)
        logger.info(f"По типу  : {dict(types)}")
        logger.info(f"По теме  : {dict(topics)}")


if __name__ == "__main__":
    main()
