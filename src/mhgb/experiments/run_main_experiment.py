"""
Главный экспериментальный пайплайн MHGB (Шаг 7.3).

Прогоняет список моделей из configs/models.yaml на всех задачах
в двух режимах:
  closed          — модель отвечает без контекста (closed-book)
  open_full_graph — модель получает полный граф-контекст (open-book)

Использование:
  uv run python src/mhgb/experiments/run_main_experiment.py \\
    --models-config configs/models.yaml \\
    --tasks data/tasks_public_120.jsonl \\
    --experiment-name main_v1 \\
    [--models qwen3-27b,deepseek-r1-32b] \\
    [--judge qwen3-27b] \\
    [--modes closed,open_full_graph] \\
    [--dry-run] \\
    [--max-tasks 20] \\
    [--save-mongo] \\
    [--skip-rta] \\
    [--rta-workers 2]

Resumability: комбинации (experiment_name, task_id, model_name, mode),
уже сохранённые в MongoDB, пропускаются.
"""

from __future__ import annotations

import json
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()  # загружаем .env до чтения os.environ в _make_client()

from mhgb.config import cfg
from mhgb.eval.answer_correctness import evaluate_answer_correctness
from mhgb.eval.context_builders import (
    build_full_graph_context,
    build_full_graph_edges_context,
)
from mhgb.eval.final_score import compute_final_score
from mhgb.eval.norm_coverage import compute_list_coverage, compute_reasoning_coverage
from mhgb.eval.response_parser import extract_norms_from_text, parse_structured_answer
from mhgb.eval.step_correctness import LLMJudge, evaluate_step_correctness
from mhgb.logging_setup import logger
from mhgb.models.llm_client import GigaChatClient, LLMClient, NativeOllamaClient, OpenAICompatibleClient

VALID_MODES = {
    "closed",
    "open_full_graph",
    "open_full_graph_edges",
}

_SYSTEM_PROMPT = (
    "Ты — эксперт в российском праве. Анализируй фабулы и отвечай строго "
    "в указанном формате."
)

_DRY_RUN_RESPONSE = (
    "ПРИМЕНИМЫЕ СТАТЬИ: dry-run\n"
    "ЦЕПОЧКА РАССУЖДЕНИЙ: dry-run\n"
    "ВЫВОД: dry-run"
)


# ---------------------------------------------------------------------------
# Предохранители: health-check и cap-монитор
# ---------------------------------------------------------------------------

class _HealthMonitor:
    """Отслеживает аномалии ответов модели на ходу.

    health@10  — после 10-го успешного ответа: empty/final0/cap/avg_len.
    cap-окно   — скользящее окно 30: если cap > 20 % → WARNING.
    Для GigaChat cap считается по tokens_output==max_tokens (finish_reason не возвращается).
    """

    _THRESH_EMPTY  = 30   # % пустых ответов для АНОМАЛИИ
    _THRESH_FINAL0 = 70   # % final_score==0.0
    _THRESH_CAP10  = 30   # % cap при health@10
    _THRESH_CAP_WIN = 20  # % cap в скользящем окне

    def __init__(self, model_name: str, max_tokens: int) -> None:
        self.model_name = model_name
        self.max_tokens = max_tokens
        self._ok: list[dict] = []
        self._window: deque[dict] = deque(maxlen=30)
        self._reported10 = False
        self._window_warned = False  # гасим повторные предупреждения пока cap высок

    def record(self, result: dict) -> None:
        if "error" in result:
            return
        self._ok.append(result)
        self._window.append(result)

        n = len(self._ok)
        if n == 10 and not self._reported10:
            self._reported10 = True
            self._report_health10()

        if len(self._window) >= 20:
            self._check_window()

    # ------------------------------------------------------------------
    def _metrics(self, recs: list[dict]) -> dict:
        n = len(recs)
        if n == 0:
            return dict(empty=0.0, final0=0.0, cap=0.0, avg_len=0.0)
        empty  = sum(1 for r in recs if not (r.get("raw_response") or "").strip())
        final0 = sum(1 for r in recs if r.get("final_score") == 0.0)
        cap    = sum(1 for r in recs if r.get("tokens_output", 0) == self.max_tokens)
        avg_len = sum(len(r.get("raw_response") or "") for r in recs) / n
        return dict(
            empty=empty / n * 100,
            final0=final0 / n * 100,
            cap=cap / n * 100,
            avg_len=avg_len,
        )

    def _report_health10(self) -> None:
        m = self._metrics(self._ok)
        sep = "=" * 64
        body = (
            f"[{self.model_name}] health@10: "
            f"empty={m['empty']:.0f}%  "
            f"final0={m['final0']:.0f}%  "
            f"cap={m['cap']:.0f}%  "
            f"avg_len={m['avg_len']:.0f}ch"
        )
        anomaly = (
            m["empty"]  > self._THRESH_EMPTY  or
            m["final0"] > self._THRESH_FINAL0 or
            m["cap"]    > self._THRESH_CAP10
        )
        if anomaly:
            logger.warning(f"\n{sep}\n{body}\n⚠️  АНОМАЛИЯ — проверьте результаты!\n{sep}")
        else:
            logger.info(f"\n{sep}\n{body}\n{sep}")

    def _check_window(self) -> None:
        m = self._metrics(list(self._window))
        if m["cap"] > self._THRESH_CAP_WIN:
            if not self._window_warned:
                logger.warning(
                    f"⚠️  [{self.model_name}] CAP-МОНИТОР (окно {len(self._window)}): "
                    f"cap={m['cap']:.0f}% — возможна обрезка ответов! "
                    f"Проверьте tokens_output vs max_tokens={self.max_tokens}."
                )
                self._window_warned = True
        else:
            self._window_warned = False  # сброс: новый всплеск снова выдаст WARNING

    def final_summary(self, n_tasks_total: int = 0) -> None:
        """Итоговая строка по модели (вызывать после завершения всех задач модели)."""
        n = len(self._ok)
        if n == 0:
            return
        m = self._metrics(self._ok)

        def _num(value: Any) -> float:
            """Клиент может не вернуть usage → поле отсутствует или не число."""
            return float(value) if isinstance(value, (int, float)) else 0.0

        avg_lat = sum(_num(r.get("latency_ms")) for r in self._ok) / n
        avg_out = sum(_num(r.get("tokens_output")) for r in self._ok) / n
        sep = "=" * 64
        logger.info(
            f"\n{sep}\n"
            f"[{self.model_name}] ИТОГ по модели: "
            f"n={n}  empty={m['empty']:.0f}%  final0={m['final0']:.0f}%  "
            f"cap={m['cap']:.0f}%  avg_len={m['avg_len']:.0f}ch  "
            f"avg_out={avg_out:.0f}tok  avg_lat={avg_lat/1000:.1f}s\n"
            f"{sep}"
        )


# ---------------------------------------------------------------------------
# Загрузка конфига моделей
# ---------------------------------------------------------------------------

def load_models_config(path: str) -> list[dict]:
    """Загружает configs/models.yaml → список конфигов моделей.

    В поле `api_base` раскрываются переменные окружения вида ${VAR} и
    ${VAR:-default} — эндпоинты задаются через .env и не хранятся в репозитории.
    """
    import os
    import re

    _ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")

    def _expand(value: str) -> str:
        def sub(match: re.Match) -> str:
            name, default = match.group(1), match.group(2)
            return os.environ.get(name) or (default or "")
        return _ENV_REF.sub(sub, value)

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    models = data["models"]
    for model_cfg in models:
        api_base = model_cfg.get("api_base")
        if isinstance(api_base, str) and "${" in api_base:
            model_cfg["api_base"] = _expand(api_base)
    return models


def instantiate_client(model_cfg: dict) -> LLMClient:
    """Создаёт LLMClient по конфигу из yaml."""
    import os

    client_type = model_cfg.get("client_type", "openai_compatible")

    if client_type == "openai_compatible":
        api_key_env = model_cfg.get("api_key_env")
        api_key = os.environ.get(api_key_env, "dummy") if api_key_env else "dummy"
        return OpenAICompatibleClient(
            api_base=model_cfg["api_base"],
            model=model_cfg["model_id"],
            api_key=api_key,
            max_tokens=model_cfg.get("max_tokens", 2048),
            temperature=model_cfg.get("temperature", 0.0),
            timeout=model_cfg.get("timeout", 300.0),
        )

    if client_type == "gigachat":
        creds_env = model_cfg.get("credentials_env", "GIGACHAT_CREDENTIALS")
        credentials = os.environ.get(creds_env, "")
        if not credentials:
            raise ValueError(f"Env-переменная {creds_env} не задана для GigaChat")
        return GigaChatClient(
            credentials=credentials,
            scope=model_cfg.get("scope", "GIGACHAT_API_PERS"),
            model=model_cfg.get("model_id", "GigaChat-Pro"),
            max_tokens=model_cfg.get("max_tokens", 2048),
            temperature=model_cfg.get("temperature", 0.0),
            timeout=model_cfg.get("timeout", 300.0),
        )

    if client_type == "native_ollama":
        # Ollama через нативный /api/chat — чтобы num_ctx из yaml реально применялся.
        # OpenAI-эндпоинт /v1 игнорирует num_ctx и грузит модель на дефолте 131072.
        # Для судьи (llama3.3-70b): num_ctx=16384 → ~49.6GB; для участников: 131072 (явный дефолт).
        api_key_env = model_cfg.get("api_key_env")
        return NativeOllamaClient(
            api_base=model_cfg["api_base"],
            model=model_cfg["model_id"],
            num_ctx=model_cfg.get("num_ctx", 16384),
            max_tokens=model_cfg.get("max_tokens", 2048),
            temperature=model_cfg.get("temperature", 0.0),
            timeout=model_cfg.get("timeout", 300.0),
            extra=model_cfg.get("extra_body"),
            think=model_cfg.get("think"),   # None → не шлём (дефолт); False → отключить thinking
        )

    raise ValueError(f"Неизвестный client_type: {client_type!r}")


# ---------------------------------------------------------------------------
# Адаптер LLMClient → LLMJudge
# ---------------------------------------------------------------------------

class _JudgeAdapter:
    """Оборачивает LLMClient в интерфейс LLMJudge (метод complete)."""

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return self._client.generate(system_prompt, user_prompt)


# ---------------------------------------------------------------------------
# Resumability
# ---------------------------------------------------------------------------

def _is_done(
    experiment_name: str,
    task_id: str,
    model_name: str,
    mode: str,
    storage,
) -> bool:
    """True если комбинация уже обработана и сохранена в MongoDB."""
    if storage is None:
        return False
    return storage.db["model_responses"].count_documents({
        "experiment_name": experiment_name,
        "task_id":   task_id,
        "model_name": model_name,
        "mode":      mode,
    }) > 0


def _stratified_barrier_sample(tasks: list[dict], n: int) -> list[dict]:
    """Стратифицированная выборка ~n задач для барьера-сметы.

    Покрывает все ячейки матрицы (type × hop_group), выбирая ФАБУЛЫ
    (base_task_id без суффикса _closed/_open) круговым обходом ячеек и
    включая обе строки пары (closed + open). Так avg_in/avg_out
    репрезентативны по режимам и глубине: deep-open даёт максимальный
    графовый контекст (avg_in ×4+ против shallow-closed), а барьер по
    первым N задачам берёт только лёгкий край выборки → недооценка сметы
    (см. DESIGN_LOG Урок про барьер-30).

    Детерминирована: без random, берёт фабулы в порядке появления в файле.
    Resume не ломается — done-set пропускает по task_id, не по позиции,
    поэтому полный прогон доделает оставшиеся задачи без повторов.
    """
    by_base: dict[str, list[dict]] = {}
    for t in tasks:
        base = t["id"].rsplit("_", 1)[0]
        by_base.setdefault(base, []).append(t)

    cells: dict[tuple, list[str]] = {}
    for base, rows in by_base.items():
        key = (rows[0].get("type"), rows[0].get("hop_group"))
        cells.setdefault(key, []).append(base)

    cell_keys = sorted(cells.keys(), key=lambda k: (str(k[0]), str(k[1])))
    idx = {k: 0 for k in cell_keys}
    selected_bases: list[str] = []
    n_tasks = 0
    while n_tasks < n:
        progressed = False
        for k in cell_keys:
            if idx[k] < len(cells[k]):
                base = cells[k][idx[k]]
                idx[k] += 1
                selected_bases.append(base)
                n_tasks += len(by_base[base])
                progressed = True
                if n_tasks >= n:
                    break
        if not progressed:
            break  # фабулы исчерпаны

    selected = set(selected_bases)
    return [t for t in tasks if t["id"].rsplit("_", 1)[0] in selected]


def _load_done_set(output_path: Path) -> set[str]:
    """Читает существующий JSONL и возвращает set task_id успешных результатов.
    Задачи с ошибкой (поле 'error') НЕ добавляются — они будут перезапущены.
    """
    done: set[str] = set()
    if not output_path.exists():
        return done
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if "error" not in r and "task_id" in r:
                    done.add(r["task_id"])
            except json.JSONDecodeError:
                pass
    return done


# ---------------------------------------------------------------------------
# Построение контекста
# ---------------------------------------------------------------------------

def _build_context(task: dict, mode: str, graph, corpus: dict, token_budget: int | None = None) -> list | None:
    if mode == "closed":
        return None
    kwargs: dict = {} if token_budget is None else {"token_budget": token_budget}
    norm_ids = task.get("norm_ids", [])
    task_id = task.get("id")  # детерминированное перемешивание чанков per-task
    if mode == "open_full_graph":
        return build_full_graph_context(norm_ids, graph, corpus, **kwargs, task_id=task_id)
    if mode == "open_full_graph_edges":
        return build_full_graph_edges_context(norm_ids, graph, corpus, **kwargs, task_id=task_id)
    raise ValueError(f"Неизвестный режим: {mode!r}")


# ---------------------------------------------------------------------------
# Формирование промпта
# ---------------------------------------------------------------------------

def _render_chunk_with_meta(c) -> str:
    """Рендер чанка с датами и типом связи для конфига full_graph_edges ((3))."""
    date_parts = []
    if c.valid_from:
        date_parts.append(f"валидна с {c.valid_from}")
    if c.valid_to:
        date_parts.append(f"до {c.valid_to}")
    if c.is_repealed:
        date_parts.append("утратила силу")
    header = f"[{c.norm_id}] {c.title}"
    if date_parts:
        header += f" ({'; '.join(date_parts)})"
    lines = [header, c.text]
    if c.edge_type and c.related_to:
        lines.append(f"— между {c.related_to} и {c.norm_id} есть отношение: {c.edge_type}")
    return "\n".join(line for line in lines if line.strip())


def _make_prompt(task: dict, context_chunks: list | None, render_meta: bool = False) -> str:
    fabula   = task.get("fabula", "")
    question = task.get("question", "")
    fmt      = task.get("expected_answer_format", "")

    parts = [f"Фабула:\n{fabula}"]
    if context_chunks:
        if render_meta:
            ctx_lines = [_render_chunk_with_meta(c) for c in context_chunks]
        else:
            ctx_lines = [
                f"[{c.norm_id}] {c.title}\n{c.text}".strip()
                for c in context_chunks
            ]
        parts.append("Контекст:\n" + "\n\n".join(ctx_lines))
    parts.append(f"Вопрос:\n{question}")
    if fmt:
        parts.append(f"Формат ответа:\n{fmt}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Eval pipeline
# ---------------------------------------------------------------------------

def _evaluate(task: dict, model_response: str, judge: LLMJudge, corpus: dict,
              skip_judge: bool = False) -> dict[str, Any]:
    """Parse → norm_coverage → step → answer → final_score.

    skip_judge=True: только NC (детерминированный, без LLM); SC/AC/final_score = None
    (заполняются post-hoc отдельным судьёй). Нужен для тяжёлых моделей, которые не
    помещаются на одну карту вместе с llama-судьёй (deepseek-r1-32b 62.5ГБ + llama
    49.6ГБ > 96ГБ .64 → своп). Схема: генерация+NC сейчас → llama-судья post-hoc.
    """
    parsed = parse_structured_answer(model_response)
    articles_text = (parsed.get("applicable_articles") or "") + " " + (parsed.get("reasoning") or "")
    predicted_articles = extract_norms_from_text(articles_text, corpus)
    gold_articles = set(task.get("norm_ids", []))

    list_cov      = compute_list_coverage(predicted_articles, gold_articles)
    reasoning_cov = compute_reasoning_coverage(parsed.get("reasoning") or "", gold_articles, corpus)

    if skip_judge:
        return {
            "parsed":                   parsed,
            "norm_coverage_list":       list_cov["f1"],
            "norm_coverage_reasoning":  reasoning_cov["f1"],
            "step_correctness":         None,
            "step_scores":              [],
            "answer_correctness":       None,
            "final_score":              None,
            "judge_logs":               [{"note": "SKIP_JUDGE_POSTHOC"}],
        }

    gold_chain = task.get("gold_chain", [])
    step_res   = evaluate_step_correctness(gold_chain, model_response, judge)
    answer_res = evaluate_answer_correctness(
        gold_answer=task.get("answer", ""),
        model_answer=model_response,
        task_meta={"type": task.get("type"), "fabula": (task.get("fabula") or "")[:200]},
        judge_client=judge,
    )
    final = compute_final_score(
        norm_coverage=list_cov["f1"],
        step_correctness=step_res["step_correctness"],
        answer_correctness=answer_res["score"],
    )

    return {
        "parsed":                   parsed,
        "norm_coverage_list":       list_cov["f1"],
        "norm_coverage_reasoning":  reasoning_cov["f1"],
        "step_correctness":         step_res["step_correctness"],
        "step_scores":              step_res["step_scores"],
        "answer_correctness":       answer_res["score"],
        "final_score":              final["final_score"],
        "judge_logs": [
            {"step_explanations": step_res["judge_explanations"]},
            {"answer_explanation": answer_res["explanation"]},
        ],
    }


# ---------------------------------------------------------------------------
# Прогон: одна задача × одна модель × один режим
# ---------------------------------------------------------------------------

def run_one(
    task: dict,
    model_name: str,
    client: LLMClient,
    judge: LLMJudge,
    mode: str,
    graph,
    corpus: dict,
    experiment_name: str,
    storage=None,
    dry_run: bool = False,
    token_budget: int | None = None,
    skip_judge: bool = False,
) -> dict[str, Any]:
    """
    Генерирует ответ, вычисляет все метрики, опционально сохраняет в MongoDB.
    Возвращает dict с результатами (или {"error": ...} при падении).
    skip_judge: генерация + NC, без SC/AC/final (post-hoc судья) — см. _evaluate.
    """
    task_id = task.get("id", "")
    try:
        context_chunks = _build_context(task, mode, graph, corpus, token_budget=token_budget)
        render_meta = mode == "open_full_graph_edges"
        prompt = _make_prompt(task, context_chunks, render_meta=render_meta)

        context_chunks_n   = len(context_chunks) if context_chunks else 0
        context_tokens_est = sum(max(1, len(c.text) // 3) for c in context_chunks) if context_chunks else 0

        t0 = time.time()
        raw_response = (
            _DRY_RUN_RESPONSE if dry_run
            else client.generate(_SYSTEM_PROMPT, prompt)
        )
        latency_ms = int((time.time() - t0) * 1000)
        usage = getattr(client, "last_usage", {})

        eval_result = _evaluate(task, raw_response, judge, corpus, skip_judge=skip_judge)

        result: dict[str, Any] = {
            "experiment_name": experiment_name,
            "task_id":   task_id,
            "model_name": model_name,
            "mode":      mode,
            "task_type": task.get("type"),
            "hop_group": task.get("hop_group"),
            "raw_response": raw_response,
            "latency_ms": latency_ms,
            "tokens_input":       usage.get("tokens_input", 0),
            "tokens_output":      usage.get("tokens_output", 0),
            "context_chunks_n":   context_chunks_n,
            "context_tokens_est": context_tokens_est,
            **eval_result,
        }

        if storage is not None and not dry_run:
            _save_one(result, storage)

        _fs = eval_result.get("final_score")
        logger.info(
            f"  {model_name} | task={task_id[:16]} | mode={mode}"
            f" | final={f'{_fs:.3f}' if _fs is not None else 'SKIP_JUDGE (NC только)'}"
        )
        return result

    except Exception as exc:
        logger.error(f"Ошибка {model_name} | task={task_id} | mode={mode}: {exc}")
        return {
            "experiment_name": experiment_name,
            "task_id":   task_id,
            "model_name": model_name,
            "mode":      mode,
            "error":     str(exc),
        }


def _save_one(result: dict, storage) -> None:
    try:
        storage.db["model_responses"].insert_one({
            "experiment_name":  result["experiment_name"],
            "task_id":          result["task_id"],
            "model_name":       result["model_name"],
            "mode":             result["mode"],
            "raw_response":     result["raw_response"],
            "parsed_articles":  result["parsed"].get("applicable_articles"),
            "parsed_reasoning": result["parsed"].get("reasoning"),
            "parsed_conclusion": result["parsed"].get("conclusion"),
            "latency_ms":       result["latency_ms"],
        })
        storage.db["evaluations"].insert_one({
            "experiment_name":         result["experiment_name"],
            "task_id":                 result["task_id"],
            "model_name":              result["model_name"],
            "mode":                    result["mode"],
            "norm_coverage_list":      result["norm_coverage_list"],
            "norm_coverage_reasoning": result["norm_coverage_reasoning"],
            "step_correctness":        result["step_correctness"],
            "answer_correctness":      result["answer_correctness"],
            "final_score":             result["final_score"],
            "judge_logs":              result["judge_logs"],
        })
    except Exception as exc:
        logger.error(f"MongoDB save error: {exc}")


# ---------------------------------------------------------------------------
# Главный цикл
# ---------------------------------------------------------------------------

def run_experiment(
    tasks: list[dict],
    models_cfg: list[dict],
    judge_client: LLMClient,
    graph,
    corpus: dict,
    modes: list[str],
    experiment_name: str,
    storage=None,
    dry_run: bool = False,
    models_filter: list[str] | None = None,
    output_path: Path | None = None,
    match_task_mode: bool = False,
    skip_judge: bool = False,
) -> list[dict[str, Any]]:
    """
    Внешний цикл: модели × задачи × режимы.
    models_filter — список name из yaml; None = все модели.
    output_path — если задан, результаты пишутся в JSONL построчно (для анализа).
    match_task_mode — запускать задачу только в её родном task.mode (игнорирует кросс-комбинации).
    """
    if models_filter:
        models_cfg = [m for m in models_cfg if m["name"] in models_filter]

    judge = _JudgeAdapter(judge_client)
    all_results: list[dict[str, Any]] = []
    tasks_per_model = sum(
        1 for t in tasks for m in modes
        if not match_task_mode or t.get("mode") == m
    )
    total = len(models_cfg) * tasks_per_model

    out_fh = None
    done_from_jsonl: set[str] = set()
    if output_path and not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        done_from_jsonl = _load_done_set(output_path)
        if done_from_jsonl:
            logger.info(
                f"Resumability (JSONL): пропускаем {len(done_from_jsonl)} успешных задач, "
                f"ошибочные будут перезапущены — {output_path}"
            )
        out_fh = open(output_path, "a", encoding="utf-8")

    logger.info(
        f"Эксперимент {experiment_name!r}: {len(models_cfg)} моделей × "
        f"{tasks_per_model} вызовов = {total} вызовов"
        + (" (--match-task-mode)" if match_task_mode else "")
    )

    try:
        with tqdm(total=total, desc=experiment_name, unit="вызов") as pbar:
            for model_cfg in models_cfg:
                model_name = model_cfg["name"]
                logger.info(f"Модель: {model_name}")
                health = _HealthMonitor(model_name, model_cfg.get("max_tokens", 2048))

                try:
                    client = instantiate_client(model_cfg)
                except Exception as exc:
                    logger.error(f"Не удалось создать клиент {model_name}: {exc}")
                    pbar.update(tasks_per_model)
                    continue

                # Порог 8 ПОДРЯД (не суммарно): одиночные разбросанные ошибки сбрасывают
                # счётчик (см. else ниже) и не копятся. 8 фильтрует случайные серии медленных
                # задач / короткое моргание сервера, но ловит реальное падение быстро.
                _CIRCUIT_LIMIT = 8
                consecutive_errors = 0

                for task in tasks:
                    for mode in modes:
                        if match_task_mode and task.get("mode") != mode:
                            continue
                        pbar.update(1)
                        task_id = task.get("id", "")

                        if _is_done(experiment_name, task_id, model_name, mode, storage):
                            continue
                        if task_id in done_from_jsonl:
                            continue

                        result = run_one(
                            task, model_name, client, judge, mode,
                            graph, corpus, experiment_name, storage, dry_run,
                            token_budget=model_cfg.get("token_budget"),
                            skip_judge=skip_judge,
                        )
                        all_results.append(result)
                        health.record(result)
                        if out_fh is not None:
                            _writeable = {k: v for k, v in result.items()
                                          if k not in ("parsed", "judge_logs", "step_scores")}
                            out_fh.write(json.dumps(_writeable, ensure_ascii=False) + "\n")
                            out_fh.flush()

                        if "error" in result:
                            consecutive_errors += 1
                            if consecutive_errors >= _CIRCUIT_LIMIT:
                                logger.error(
                                    f"CIRCUIT BREAKER [{model_name}]: "
                                    f"{consecutive_errors} ошибок подряд — эндпоинт, вероятно, недоступен. "
                                    f"Проверьте /api/ps на сервере и перезапустите ту же команду "
                                    f"для возобновления (успешные задачи будут пропущены, "
                                    f"упавшие — переделаны)."
                                )
                                break  # выходим из цикла задач для этой модели
                        else:
                            consecutive_errors = 0
                    else:
                        continue
                    break  # propagate inner break через внешний for
                health.final_summary()
    finally:
        if out_fh is not None:
            out_fh.close()

    ok  = sum(1 for r in all_results if "error" not in r)
    err = len(all_results) - ok
    logger.info(f"Готово: {ok} успешно, {err} ошибок")
    return all_results


# ---------------------------------------------------------------------------
# Постфактум RtA-прогон (вызывается в конце main, если не --skip-rta)
# ---------------------------------------------------------------------------

def _run_rta_pass(
    all_results: list[dict],
    tasks: list[dict],
    output_path: Path,
    judge_client,
    rta_workers: int = 2,
) -> None:
    """Запускает RtA-детекцию на результатах и пишет results_rta.jsonl."""
    from mhgb.eval.rta_detector import RtADetector

    prompt_path = Path(__file__).parents[3] / "prompts" / "rta_judge.md"
    if not prompt_path.exists():
        logger.warning(f"RtA промпт не найден: {prompt_path} — пропускаем RtA-прогон")
        return

    tasks_index: dict[str, dict] = {
        (t.get("task_id") or t.get("id", "")): t
        for t in tasks
    }

    # Уже обработанные (resumability)
    done: set[str] = set()
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line).get("task_id", ""))

    batch = []
    for r in all_results:
        if "error" in r:
            continue
        tid = r.get("task_id", "")
        if tid in done:
            continue
        task = tasks_index.get(tid, {})
        if not task.get("fabula"):
            continue
        batch.append({
            "fabula":         task["fabula"],
            "question":       task["question"],
            "model_response": r.get("raw_response", ""),
            "task_id":        tid,
        })

    if not batch:
        logger.info("RtA: все записи уже обработаны или нет данных — пропускаем")
        return

    logger.info(f"RtA: запускаю детекцию для {len(batch)} ответов ({rta_workers} потока)...")
    detector = RtADetector(judge_client, prompt_path)
    rta_results = detector.detect_batch(batch, max_workers=rta_workers)
    rta_map = {r.task_id: r for r in rta_results}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as out:
        for r in all_results:
            tid = r.get("task_id", "")
            if "error" in r or tid in done or tid not in rta_map:
                continue
            rta = rta_map[tid]
            enriched = {
                k: v for k, v in r.items()
                if k not in ("parsed", "judge_logs")
            }
            enriched.update({
                "is_rta":          rta.is_rta,
                "rta_type":        rta.refusal_type,
                "rta_topic":       rta.refusal_topic,
                "rta_confidence":  rta.confidence,
                "rta_explanation": rta.explanation,
            })
            out.write(json.dumps(enriched, ensure_ascii=False) + "\n")

    n_rta = sum(1 for r in rta_results if r.is_rta)
    logger.info(
        f"RtA готово → {output_path.name}: "
        f"{n_rta}/{len(rta_results)} отказов "
        f"({100*n_rta//max(1,len(rta_results))}%)"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    import networkx as nx

    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="MHGB Main Experiment Runner")
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--tasks",         default="data/tasks_public_120.jsonl")
    parser.add_argument("--experiment-name", default="main_v1")
    parser.add_argument("--modes",  default="closed,open_full_graph")
    parser.add_argument("--models", default=None,
                        help="Фильтр: имена моделей через запятую")
    parser.add_argument("--judge",  default=None,
                        help="Имя модели-судьи из yaml (по умолчанию — первая)")
    parser.add_argument("--dry-run",   action="store_true")
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument(
        "--barrier-stratified",
        action="store_true",
        help="Выбрать --max-tasks задач стратифицированно по матрице type×hop_group "
             "(вместо первых N по порядку) — для честной сметы стоимости перед дорогими "
             "прогонами. Resume доделает остальные без повторов (done-set по task_id).",
    )
    parser.add_argument("--save-mongo", action="store_true")
    parser.add_argument("--output", default=None,
                        help="Путь к JSONL-файлу для записи результатов (append)")
    parser.add_argument("--match-task-mode", action="store_true",
                        help="Запускать каждую задачу только в её родном режиме (task.mode)")
    parser.add_argument("--skip-rta", action="store_true",
                        help="Пропустить RtA-детекцию после сбора ответов (по умолчанию включена)")
    parser.add_argument("--rta-workers", type=int, default=2,
                        help="Параллельные потоки для RtA-детекции (default: 2)")
    parser.add_argument("--skip-judge", action="store_true",
                        help="Только генерация + NC (без SC/AC/final); судья post-hoc. "
                             "Для тяжёлых моделей, не влезающих на карту вместе с llama-судьёй "
                             "(deepseek-r1-32b). Подразумевает пропуск инлайн-RtA.")
    args = parser.parse_args()

    modes = [m.strip() for m in args.modes.split(",")]
    unknown = set(modes) - VALID_MODES
    if unknown:
        parser.error(f"Неизвестные режимы: {unknown}. Допустимые: {VALID_MODES}")

    models_cfg = load_models_config(args.models_config)
    models_filter = [m.strip() for m in args.models.split(",")] if args.models else None

    judge_name  = args.judge or models_cfg[0]["name"]
    judge_cfgs  = [m for m in models_cfg if m["name"] == judge_name]
    if not judge_cfgs:
        parser.error(f"Модель-судья {judge_name!r} не найдена в конфиге")
    judge_client = instantiate_client(judge_cfgs[0])

    tasks: list[dict] = []
    with open(args.tasks, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))

    # Нормализуем mode: "open" → "open_full_graph_edges" (P2-4: основной open-конфиг).
    # Phase 1 использовала "open_full_graph"; для P2-4 используем ((3)) full_graph_edges.
    for t in tasks:
        if t.get("mode") == "open":
            t["mode"] = "open_full_graph_edges"

    if args.max_tasks:
        if args.barrier_stratified:
            tasks = _stratified_barrier_sample(tasks, args.max_tasks)
            _cells = sorted({(t.get("type"), t.get("hop_group")) for t in tasks})
            logger.info(
                f"Стратифицированный барьер: {len(tasks)} задач, "
                f"покрыто ячеек матрицы type×hop_group: {len(_cells)}/12"
            )
        else:
            tasks = tasks[: args.max_tasks]
    logger.info(f"Загружено {len(tasks)} задач из {args.tasks}")

    with open(cfg.data_dir / "graph.json", encoding="utf-8") as f:
        graph = nx.node_link_graph(json.load(f), edges="edges")
    corpus: dict = {}
    with open(cfg.data_dir / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            a = json.loads(line)
            corpus[a["id"]] = a

    storage = None
    if args.save_mongo:
        from mhgb.storage.mongo_client import MongoStorage
        storage = MongoStorage()

    output_path = Path(args.output) if args.output else None
    if output_path is None and not args.dry_run:
        output_path = Path(f"reports/{args.experiment_name}/results.jsonl")
        logger.info(f"--output не задан, результаты → {output_path}")

    all_results = run_experiment(
        tasks=tasks,
        models_cfg=models_cfg,
        judge_client=judge_client,
        graph=graph,
        corpus=corpus,
        modes=modes,
        experiment_name=args.experiment_name,
        storage=storage,
        dry_run=args.dry_run,
        models_filter=models_filter,
        output_path=output_path,
        match_task_mode=args.match_task_mode,
        skip_judge=args.skip_judge,
    )

    # Автоматический RtA-прогон (если не --skip-rta/--skip-judge и не --dry-run).
    # При --skip-judge инлайн-RtA НЕ запускаем: RtA-судья (llama) загрузился бы рядом
    # с тяжёлой моделью на .64 → своп. RtA делается post-hoc после выгрузки модели.
    if not args.skip_rta and not args.skip_judge and not args.dry_run and output_path and all_results:
        rta_output = output_path.parent / "results_rta.jsonl"
        logger.info(f"Запускаю RtA-детекцию → {rta_output}")
        _run_rta_pass(
            all_results=all_results,
            tasks=tasks,
            output_path=rta_output,
            judge_client=judge_client,
            rta_workers=args.rta_workers,
        )
    elif args.skip_rta:
        logger.info("--skip-rta: RtA-детекция пропущена. Запусти вручную: scripts/run_rta_detection.py")


if __name__ == "__main__":
    main()
