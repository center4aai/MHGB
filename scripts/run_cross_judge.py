"""
P2-4.4 — постфактум оценка SC и AC вторым судьёй (cross-judge pass).

Читает:
  reports/<experiment>/results.jsonl   — ответы модели (raw_response)
  data/tasks_raw.jsonl                 — gold_chain, gold answer, метаданные

Пишет:
  reports/<experiment>/results_crossjudge_<judge_name>.jsonl
  Поля на запись: task_id, mode, model_name, sc_crossjudge,
                  step_scores_crossjudge, step_explanations_crossjudge,
                  ac_crossjudge, ac_explanation_crossjudge, judge_name.

Логика:
  1. Для каждой успешной записи в results.jsonl берём raw_response.
  2. Из tasks_raw.jsonl берём gold_chain (для SC) и answer (для AC).
  3. Оцениваем SC пошагово — evaluate_step_correctness.
  4. Оцениваем AC — evaluate_answer_correctness.
  5. Пишем в results_crossjudge_<judge>.jsonl (append, resumable).

Пример (trial, 5 записей, без реальных вызовов LLM):
  uv run python scripts/run_cross_judge.py \\
      --experiment tpro_full600 \\
      --max-records 5 \\
      --dry-run

Полный прогон (только после явного одобрения):
  uv run python scripts/run_cross_judge.py \\
      --experiment tpro_full600 \\
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

from mhgb.eval.answer_correctness import evaluate_answer_correctness
from mhgb.eval.step_correctness import evaluate_step_correctness
from mhgb.logging_setup import logger


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def load_tasks_index(tasks_path: Path) -> dict[str, dict]:
    """task_id → {gold_chain, answer, question, type, hop_group, mode}

    Поле 'id' в tasks_raw.jsonl уже содержит суффикс режима
    (например 'uuid_closed', 'uuid_open') — совпадает с task_id в results.jsonl.
    """
    index: dict[str, dict] = {}
    with tasks_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            tid = rec.get("id") or rec.get("task_id", "")
            index[tid] = {
                "gold_chain": rec.get("gold_chain", []),
                "answer":     rec.get("answer", ""),
                "question":   rec.get("question", ""),
                "type":       rec.get("type", "?"),
                "hop_group":  rec.get("hop_group", "?"),
                "mode":       rec.get("mode", "?"),
            }
    return index


def load_results(results_path: Path) -> list[dict]:
    records: list[dict] = []
    with results_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_done_set(out_path: Path) -> set[str]:
    """Читает уже обработанные task_id из results_crossjudge.jsonl (resumability)."""
    done: set[str] = set()
    if not out_path.exists():
        return done
    with out_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if "sc_crossjudge" in rec:
                    done.add(rec["task_id"])
            except json.JSONDecodeError:
                pass
    return done


def build_judge_client(judge_name: str):
    """Строит LLM-клиент по имени из models.yaml."""
    import os

    from mhgb.experiments.run_main_experiment import load_models_config
    from mhgb.models.llm_client import NativeOllamaClient, OpenAICompatibleClient

    cfg_path = ROOT / "configs" / "models.yaml"
    models = load_models_config(str(cfg_path))  # раскрывает ${VAR} в api_base
    cfg = next((m for m in models if m["name"] == judge_name), None)
    if cfg is None:
        raise ValueError(f"Модель '{judge_name}' не найдена в models.yaml")

    api_key = os.environ.get(cfg.get("api_key_env") or "", "dummy")
    client_type = cfg.get("client_type", "openai_compatible")

    if client_type == "native_ollama":
        return NativeOllamaClient(
            api_base=cfg["api_base"],
            model=cfg["model_id"],
            num_ctx=cfg.get("num_ctx", 16384),
            max_tokens=cfg.get("max_tokens", 1024),
            temperature=cfg.get("temperature", 0.0),
            timeout=cfg.get("timeout", 300.0),
        )

    return OpenAICompatibleClient(
        api_base=cfg["api_base"],
        api_key=api_key,
        model=cfg["model_id"],
        max_tokens=cfg.get("max_tokens", 1024),
        temperature=cfg.get("temperature", 0.0),
        timeout=cfg.get("timeout", 300.0),
    )


class _JudgeAdapter:
    """Оборачивает LLMClient в LLMJudge Protocol (метод complete)."""

    def __init__(self, client) -> None:
        self._client = client

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return self._client.generate(system_prompt, user_prompt)


# ---------------------------------------------------------------------------
# Главная функция
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Постфактум cross-judge оценка SC/AC вторым судьёй."
    )
    parser.add_argument("--experiment", required=True,
                        help="Имя эксперимента (папка в reports/)")
    parser.add_argument("--tasks", default="data/tasks_public_120.jsonl",
                        help="Путь к tasks_raw.jsonl")
    parser.add_argument("--judge", default="qwen3-27b-server",
                        help="Имя модели-судьи из models.yaml")
    parser.add_argument("--max-records", type=int, default=None,
                        help="Обработать только N записей (для trial-прогона)")
    parser.add_argument("--max-workers", type=int, default=1,
                        help="Параллельные потоки (для судьи на локальном сервере держи 1)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Показать план без LLM-вызовов")
    args = parser.parse_args()

    _reports_root = ROOT / "reports"
    exp_dir      = _reports_root / args.experiment
    results_path = exp_dir / "results.jsonl"
    tasks_path   = ROOT / args.tasks

    # Имя выходного файла включает имя судьи (позволяет сравнить двух судей)
    judge_slug   = args.judge.replace("/", "_").replace(":", "-")
    out_path     = exp_dir / f"results_crossjudge_{judge_slug}.jsonl"

    if not results_path.exists():
        logger.error(f"Не найден {results_path}")
        sys.exit(1)
    if not tasks_path.exists():
        logger.error(f"Не найден {tasks_path}")
        sys.exit(1)

    # --- Загрузка данных --------------------------------------------------
    logger.info("Загружаю tasks_raw...")
    tasks_index = load_tasks_index(tasks_path)
    logger.info(f"Задач загружено: {len(tasks_index)}")

    logger.info("Загружаю results.jsonl...")
    all_records = load_results(results_path)
    logger.info(f"Всего записей в results.jsonl: {len(all_records)}")

    # Пропускаем записи с ошибкой (overflow, timeout и др.)
    valid_records = [r for r in all_records if "error" not in r]
    n_errors = len(all_records) - len(valid_records)
    if n_errors:
        logger.info(f"Пропущено error-записей: {n_errors}")

    # Для cross-judge обязателен raw_response
    valid_records = [r for r in valid_records if r.get("raw_response")]
    n_no_response = len([r for r in valid_records if not r.get("raw_response")])
    if n_no_response:
        logger.warning(f"Пропущено записей без raw_response: {n_no_response}")

    # --- Джойн с tasks_raw ------------------------------------------------
    # task_id в results.jsonl совпадает с полем 'id' в tasks_raw.jsonl
    # (оба содержат суффикс режима: uuid_closed / uuid_open)
    joined: list[dict] = []
    missing_tasks = 0
    for rec in valid_records:
        tid       = rec.get("task_id", "")
        task_info = tasks_index.get(tid)
        if task_info is None:
            missing_tasks += 1
            continue
        joined.append({**rec, "_task": task_info})

    if missing_tasks:
        logger.warning(f"Не найдены задачи для {missing_tasks} записей — пропущены")
    logger.info(f"После джойна: {len(joined)} записей")

    # --- Ограничение по --max-records -------------------------------------
    if args.max_records:
        joined = joined[: args.max_records]
        logger.info(f"--max-records={args.max_records}: берём первые {len(joined)}")

    # --- Resumability -----------------------------------------------------
    done = load_done_set(out_path)
    if done:
        logger.info(f"Уже обработано (resumability): {len(done)}")
    todo = [r for r in joined if r["task_id"] not in done]
    logger.info(f"К обработке: {len(todo)} записей")

    # --- Dry-run ----------------------------------------------------------
    if args.dry_run:
        print("\n=== DRY-RUN ===")
        print(f"Эксперимент  : {args.experiment}")
        print(f"Судья        : {args.judge}")
        print(f"Записей всего: {len(all_records)}")
        print(f"  с ошибкой  : {n_errors}")
        print(f"  после джойна:{len(joined)}")
        print(f"  в done-set : {len(done)}")
        print(f"  к обработке: {len(todo)}")
        print(f"Выходной файл: {out_path}")
        if todo:
            ex = todo[0]
            task = ex["_task"]
            n_steps = len([s for s in task["gold_chain"] if s.get("norm_id")])
            print(f"\nПример первой записи:")
            print(f"  task_id     : {ex['task_id']}")
            print(f"  mode        : {ex.get('mode')}")
            print(f"  model_name  : {ex.get('model_name')}")
            print(f"  gold_chain  : {n_steps} оцениваемых шагов")
            print(f"  raw_response: {str(ex.get('raw_response',''))[:80]}...")
            print(f"  gold_answer : {task['answer'][:80]}...")
        return

    if not todo:
        logger.info("Все записи уже обработаны. Выход.")
        return

    # --- Инициализация судьи ---------------------------------------------
    logger.info(f"Инициализирую судью: {args.judge}")
    raw_client  = build_judge_client(args.judge)
    judge       = _JudgeAdapter(raw_client)

    # --- Прогон -----------------------------------------------------------
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm

    def _evaluate_one(rec: dict) -> dict:
        tid   = rec["task_id"]
        task  = rec["_task"]
        raw   = rec.get("raw_response", "")
        mode  = rec.get("mode", "?")

        gold_chain   = task["gold_chain"]
        gold_answer  = task["answer"]
        task_meta    = {
            "type":     task["type"],
            "mode":     mode,
            "hop_group": task["hop_group"],
            "question": task["question"],
        }

        # Step Correctness
        sc_res = evaluate_step_correctness(gold_chain, raw, judge)
        # Answer Correctness
        ac_res = evaluate_answer_correctness(gold_answer, raw, task_meta, judge)

        return {
            "task_id":                   tid,
            "mode":                      mode,
            "model_name":                rec.get("model_name"),
            "sc_crossjudge":             sc_res["step_correctness"],
            "step_scores_crossjudge":    sc_res["step_scores"],
            "step_explanations_crossjudge": sc_res["judge_explanations"],
            "ac_crossjudge":             ac_res["score"],
            "ac_explanation_crossjudge": ac_res["explanation"],
            "judge_name":                args.judge,
        }

    n_ok = 0
    n_err = 0

    with out_path.open("a", encoding="utf-8") as fout:
        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            futures = {pool.submit(_evaluate_one, rec): rec["task_id"] for rec in todo}
            with tqdm(total=len(todo), unit="запись", desc="cross-judge") as pbar:
                for fut in as_completed(futures):
                    tid = futures[fut]
                    try:
                        result = fut.result()
                        fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                        fout.flush()
                        n_ok += 1
                    except Exception as exc:
                        logger.error(f"Ошибка {tid}: {exc}")
                        n_err += 1
                    pbar.update(1)

    logger.info(f"Готово. Записано: {n_ok}, ошибок: {n_err} → {out_path}")

    # Краткая статистика
    if n_ok:
        import statistics
        sc_vals: list[float] = []
        ac_vals: list[float] = []
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    if "sc_crossjudge" in r:
                        sc_vals.append(r["sc_crossjudge"])
                        ac_vals.append(r["ac_crossjudge"])
                except json.JSONDecodeError:
                    pass
        logger.info(
            f"SC crossjudge: mean={statistics.mean(sc_vals):.3f}  "
            f"AC crossjudge: mean={statistics.mean(ac_vals):.3f}  "
            f"(n={len(sc_vals)})"
        )


if __name__ == "__main__":
    main()
