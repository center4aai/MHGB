"""
MHGB Graph Explorer — interactive interface to the knowledge graph and the tasks.

Run:  uv run streamlit run graph_explorer.py

Optional environment variables (defaults suit a plain checkout):
  MHGB_DATA_DIR      directory holding graph.json / corpus.jsonl / the task file
  MHGB_REPORTS_DIR   directory holding evaluation results (Analytics, Leaderboard)
  MHGB_CONFIGS_DIR   directory holding models.yaml
  MHGB_ALLOW_EXPORT  "1" re-enables the table export button (disabled by default)
"""

import html as _html
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import pandas as pd
import streamlit as st
import yaml
from pyvis.network import Network

try:
    import plotly.graph_objects as go
    _PLOTLY_OK = True
except ImportError:
    _PLOTLY_OK = False

# ---------------------------------------------------------------------------
# Конфиг
# ---------------------------------------------------------------------------

# Paths resolve relative to this file rather than the current directory: under a
# service manager the working directory is set separately and cannot be relied on.
# Each directory can be overridden, so data may live outside the checkout.
_APP_DIR = Path(__file__).resolve().parent


def _dir_from_env(var: str, default: Path) -> Path:
    value = os.environ.get(var)
    return Path(value).expanduser().resolve() if value else default


DATA_DIR          = _dir_from_env("MHGB_DATA_DIR",    _APP_DIR / "data")
REPORTS_DIR       = _dir_from_env("MHGB_REPORTS_DIR", _APP_DIR / "reports")
CONFIGS_DIR       = _dir_from_env("MHGB_CONFIGS_DIR", _APP_DIR / "configs")

GRAPH_PATH        = DATA_DIR / "graph.json"
CORPUS_PATH       = DATA_DIR / "corpus.jsonl"
# The full 600-task set when present, otherwise the public 120-task subset.
_TASKS_FULL       = DATA_DIR / "tasks_raw.jsonl"
TASKS_PATH        = _TASKS_FULL if _TASKS_FULL.exists() else DATA_DIR / "tasks_public_120.jsonl"
PHASE2_DIR        = REPORTS_DIR / "phase2"
MODELS_YAML_PATH  = CONFIGS_DIR / "models.yaml"

# Table export is off by default. The st.dataframe toolbar carries a
# "Download as CSV" button that would dump every filtered task, seed norms
# included; browsing tasks one by one stays available.
ALLOW_EXPORT = os.environ.get("MHGB_ALLOW_EXPORT", "") == "1"

# ---------------------------------------------------------------------------
# Версионирование экспериментов (P2-4.7): MVP (май 2026) vs Phase-2 (июнь-июль)
# ---------------------------------------------------------------------------
# MVP-данные заморожены в reports/<exp>; Phase-2 пишется в reports/phase2/<exp>.
# Нельзя смешивать на графиках — другой судья, контекст с метаданными рёбер,
# рандомизация порядка чанков → методологически несравнимо.
MVP_FULL_EXPERIMENTS = {
    "o3_full600", "deepseek_full600", "gigachat2max_full600",
    "gigachat2_full600", "qwen3_full600", "yandexgpt_full600",
}
# Phase-2-реперные прогоны, лежащие плоско в reports/ (rehearsal-прогоны без
# gap_records.jsonl → в UI не отображаются, но классифицируются правильно).
# tpro_full600 и gemma4_full600 перенесены в reports/phase2/ (29.06).
PHASE2_FLAT_PENDING = {
    "tpro_rehearsal", "tpro_rehearsal2",
}

EXPERIMENT_VERSION_LABELS = {
    "phase2": "Phase-2 эксперименты (июнь-июль 2026)",
    "mvp":    "MVP-эксперименты (май 2026)",
}


def resolve_exp_dir(exp_name: str, version: str | None = None) -> Path:
    """Папка эксперимента — резолвинг по ВЕРСИИ (изоляция данных по пути).

    Данные MVP и Phase-2 физически разделены по корню (reports/ vs
    reports/phase2/). Одноимённые эксперименты (yandexgpt_full600,
    o3_full600, gigachat2max_full600, qwen3_full600) существуют в ОБОИХ —
    поэтому резолвинг обязан учитывать версию, а не только имя.

    version="mvp"    → reports/<exp>  (майский, плоский корень)
    version="phase2" → reports/phase2/<exp> если есть, иначе reports/<exp>
                       (PHASE2_FLAT_PENDING: переехавшие в phase2/ берутся
                        оттуда, не переехавшие — из плоского reports/)
    version=None     → обратная совместимость (phase2 если есть, иначе
                       reports) — для вкладок без выбора версии.
    """
    if version == "mvp":
        return REPORTS_DIR / exp_name
    p2 = PHASE2_DIR / exp_name
    if version == "phase2":
        return p2 if p2.is_dir() else REPORTS_DIR / exp_name
    # version is None → прежнее поведение
    if p2.is_dir():
        return p2
    return REPORTS_DIR / exp_name

ACCESS_MAP = {
    "open_source":  "Открытый",
    "proprietary":  "Проприетарный",
    "russian":      "Проприетарный",
}

EDGE_COLORS = {
    "исключает":    "#e74c3c",
    "дополняет":    "#2ecc71",
    "приоритет":    "#f39c12",
    "применяется_к":"#3498db",
    "ссылается_на": "#bdc3c7",
}

NODE_COLORS = {
    "ТК":    "#e8f4f8",
    "ГК":    "#fef9e7",
    "КоАП":  "#fdebd0",
    "СК":    "#eafaf1",
    "ЖК":    "#f4ecf7",
}

TYPE_COLORS = {
    "conflict_resolution": "#e74c3c",
    "issue_spotting":      "#3498db",
    "rule_selection":      "#27ae60",
    "temporal_validity":   "#f39c12",
}

TYPE_RU = {
    "conflict_resolution": "Conflict Resolution",
    "issue_spotting":      "Issue Spotting",
    "rule_selection":      "Rule Selection",
    "temporal_validity":   "Temporal Validity",
}

TASK_TYPES = ["issue_spotting", "rule_selection", "conflict_resolution", "temporal_validity"]

def _task_scope(task: dict) -> str:
    """Охват задачи: одна норма / внутри НПА / межкодексная."""
    norm_ids = task.get("norm_ids", [])
    if len(norm_ids) <= 1:
        return "одна норма"
    codes = {nid.split("_")[0] for nid in norm_ids}
    return "межкодексная" if len(codes) > 1 else "внутри НПА"

# ---------------------------------------------------------------------------
# Загрузка данных
# ---------------------------------------------------------------------------

@st.cache_data
def load_graph(path: str = str(GRAPH_PATH)) -> nx.DiGraph:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return nx.node_link_graph(data, directed=True)


@st.cache_data(ttl=60)
def load_graph_versions() -> list[dict]:
    try:
        from mhgb.storage.mongo_client import MongoStorage
        storage = MongoStorage()
        if not storage.check_connection():
            return []
        versions = storage.list_graph_versions()
        storage.close()
        return versions
    except Exception:
        return []


@st.cache_data
def load_corpus() -> dict:
    corpus = {}
    with open(CORPUS_PATH, encoding="utf-8") as f:
        for line in f:
            a = json.loads(line)
            corpus[a["id"]] = a
    return corpus


@st.cache_data
def load_tasks() -> list[dict]:
    if not TASKS_PATH.exists():
        return []
    tasks = []
    with open(TASKS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                t = json.loads(line)
                if not t.get("branch_of_law"):
                    norms = t.get("norm_ids") or []
                    if norms:
                        code = norms[0].split("_")[0]
                        t["branch_of_law"] = _BRANCH_MAP.get(code, "")
                tasks.append(t)
    return tasks


@st.cache_data
def list_experiments() -> list[str]:
    if not REPORTS_DIR.exists():
        return []
    return sorted(
        d.name for d in REPORTS_DIR.iterdir()
        if d.is_dir() and (d / "gap_records.jsonl").exists()
    )


@st.cache_data
def list_experiments_versioned() -> list[dict]:
    """Все эксперименты с gap_records.jsonl из обоих корней + версия (mvp|phase2).

    Phase-2 = reports/phase2/* (канон) ∪ транзитные плоские (PHASE2_FLAT_PENDING).
    MVP     = всё остальное плоское в reports/ (6 _full + майские smoke).
    """
    out: list[dict] = []
    phase2_names: set[str] = set()

    # Канонический Phase-2: reports/phase2/*
    if PHASE2_DIR.exists():
        for d in sorted(PHASE2_DIR.iterdir()):
            if d.is_dir() and (d / "gap_records.jsonl").exists():
                out.append({"name": d.name, "version": "phase2"})
                phase2_names.add(d.name)

    # Плоский reports/* (исключая подпапку phase2)
    if REPORTS_DIR.exists():
        for d in sorted(REPORTS_DIR.iterdir()):
            if not d.is_dir() or d.name == "phase2":
                continue
            if not (d / "gap_records.jsonl").exists():
                continue
            if d.name in PHASE2_FLAT_PENDING:
                # транзитный flat Phase-2 (t-pro/gemma до переезда):
                # добавить как phase2, но НЕ задваивать, если уже учтён
                # из канонического reports/phase2/
                if d.name not in phase2_names:
                    out.append({"name": d.name, "version": "phase2"})
                    phase2_names.add(d.name)
            else:
                # MVP — добавляем ВСЕГДА, даже если есть одноимённый phase2.
                # Это РАЗНЫЕ версии (reports/<exp> vs reports/phase2/<exp>);
                # дедуп по имени скрывал бы MVP-двойник перепрогнанных моделей.
                out.append({"name": d.name, "version": "mvp"})

    return out


@st.cache_data
def load_model_metadata() -> dict:
    """Загружает метаданные моделей из models.yaml."""
    if not MODELS_YAML_PATH.exists():
        return {}
    with open(MODELS_YAML_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return {
        m["name"]: {
            "display_name": m.get("display_name", m["name"]),
            "size":         m.get("size", "—"),
            "type":         m.get("type", "general"),
            "slice":        m.get("slice", "—"),
            "adapted_to_ru": bool(m.get("adapted_to_ru", False)),
        }
        for m in cfg.get("models", [])
    }


@st.cache_data
def load_leaderboard_raw(version: str = "phase2") -> list[dict]:
    """Загружает gap_records из всех _full экспериментов версии.

    version="phase2" → reports/phase2/*_full600; version="mvp" → плоский reports/*_full.
    Версии НЕ смешиваются (несравнимы). Одноимённые (o3/yandex/gigachat-max) разведены
    по корню, как в resolve_exp_dir.
    """
    if not REPORTS_DIR.exists():
        return []

    # Build task_id → branch_of_law lookup from tasks_raw.jsonl.
    # gap_records.jsonl stores branch_of_law="unknown" because results.jsonl
    # doesn't carry this field; we derive it here from norm_ids.
    task_branch: dict[str, str] = {}
    if TASKS_PATH.exists():
        with open(TASKS_PATH, encoding="utf-8") as _tf:
            for _line in _tf:
                _line = _line.strip()
                if not _line:
                    continue
                _t = json.loads(_line)
                _raw_id = _t.get("id", "")
                _base_id = _raw_id.removesuffix("_closed").removesuffix("_open")
                if _base_id and _base_id not in task_branch:
                    _norms = _t.get("norm_ids") or []
                    _code  = _norms[0].split("_")[0] if _norms else ""
                    task_branch[_base_id] = _BRANCH_MAP.get(_code, "unknown") if _code else "unknown"

    records = []
    for d in _full_dirs(version):
        gap_path = d / "gap_records.jsonl"
        if not gap_path.exists():
            continue
        with open(gap_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    r["_experiment"] = d.name
                    if r.get("branch_of_law", "unknown") in ("unknown", "", None):
                        r["branch_of_law"] = task_branch.get(r.get("task_id", ""), "unknown")
                    records.append(r)
    return records


def _aggregate_leaderboard(records: list[dict], models_meta: dict) -> pd.DataFrame:
    """Агрегирует gap_records по моделям → строки лидерборда."""
    grouped: dict[str, list] = defaultdict(list)
    for r in records:
        grouped[r.get("model_name", "unknown")].append(r)

    rows = []
    for model_name, recs in grouped.items():
        meta = models_meta.get(model_name, {})
        n = len(recs)
        quad_counts = Counter(r.get("quadrant", "incompetent") for r in recs)
        rows.append({
            "_model_name":       model_name,
            "_slice":            meta.get("slice", ""),
            "Модель":            meta.get("display_name", model_name),
            "Параметры":         meta.get("size", "—"),
            "Тип":               "Reasoning" if meta.get("type") == "reasoning" else "General",
            "Доступ":            ACCESS_MAP.get(meta.get("slice", ""), "—"),
            "Адаптация":         "рус." if meta.get("adapted_to_ru") else "—",
            "Знает %":           round(quad_counts.get("knows", 0) / n * 100, 1),
            "Рассуждает %":      round(quad_counts.get("reasons", 0) / n * 100, 1),
            "Не справляется %":  round(quad_counts.get("incompetent", 0) / n * 100, 1),
            "Галлюцинирует %":   round(quad_counts.get("hallucinates", 0) / n * 100, 1),
            "Mean Closed":       round(sum(r.get("closed_final_score", 0) for r in recs) / n, 3),
            "Mean Open":         round(sum(r.get("open_final_score", 0) for r in recs) / n, 3),
            "GAP":               round(sum(r.get("final_score_gap", 0) for r in recs) / n, 3),
            "N пар":             n,
        })

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("Mean Open", ascending=False).reset_index(drop=True)
    df.insert(0, "#", range(1, len(df) + 1))
    return df


def _render_leaderboard_html(df: "pd.DataFrame") -> str:
    """HTML-таблица лидерборда: групповые заголовки, цвета, сортировка."""

    # ── color interpolation: green #44944A ↔ white ↔ red #AF2B1E ─────────────
    # green (68, 148, 74)  |  white (255, 255, 255)  |  red (175, 43, 30)
    # final result mixed with 30% white to reduce saturation
    def _lerp_bg(v: float, low_good: bool = False) -> str:
        t = (1.0 - v) if low_good else v
        if t >= 0.5:                       # white → green
            tp = (t - 0.5) * 2.0
            r = int(255 + tp * (68  - 255))
            g = int(255 + tp * (148 - 255))
            b = int(255 + tp * (74  - 255))
        else:                              # red → white
            tp = t * 2.0
            r = int(175 + tp * (255 - 175))
            g = int(43  + tp * (255 - 43))
            b = int(30  + tp * (255 - 30))
        # +30% white → softer tones, better text readability
        r = int(r * 0.7 + 255 * 0.3)
        g = int(g * 0.7 + 255 * 0.3)
        b = int(b * 0.7 + 255 * 0.3)
        return f"rgb({r},{g},{b})"

    def _norm(series):
        mn, mx = float(series.min()), float(series.max())
        if mx == mn:
            return [0.5] * len(series)
        return [(float(v) - mn) / (mx - mn) for v in series]

    _COLOR_CFG = {
        "Знает %":           False,
        "Рассуждает %":      False,
        "Не справляется %":  True,
        "Галлюцинирует %":   True,
        "Mean Closed":       False,
        "Mean Open":         False,
        "GAP":               False,
    }
    col_colors: dict[str, list[str]] = {}
    for col, lg in _COLOR_CFG.items():
        if col in df.columns:
            col_colors[col] = [_lerp_bg(v, lg) for v in _norm(df[col])]

    # (df col name, display label, width_px or None, text-align)
    _COL_CFG: list[tuple[str, str, int | None, str]] = [
        ("#",                 "№",                   40,   "center"),
        ("Модель",            "Модель",               160,  "left"),
        ("Параметры",         "Параметры",            80,   "center"),
        ("Тип",               "Тип",                  75,   "center"),
        ("Доступ",            "Доступ",               110,  "center"),
        ("Адаптация",         "Адаптация",            85,   "center"),
        ("Знает %",           "Знает, %",             130,  "center"),
        ("Рассуждает %",      "Рассуждает, %",        130,  "center"),
        ("Не справляется %",  "Не справляется, %",    130,  "center"),
        ("Галлюцинирует %",   "Галлюцинирует, %",     130,  "center"),
        ("Mean Closed",       "Mean Closed",          110,  "center"),
        ("Mean Open",         "Mean Open",            110,  "center"),
        ("GAP",               "GAP",                  80,   "center"),
        ("N пар",             "N пар",                60,   "center"),
    ]

    tbl_id = "lb_tbl_main"

    css = """
<style>
body { margin: 0; font-family: sans-serif; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { border: 1px solid #d8d8d8; padding: 5px 8px; white-space: nowrap; }
th { background: #f5f5f5; text-align: center; font-weight: 600; }
.grp-quad  { background: #dbeafe; color: #1e40af; }
.grp-score { background: #fef3c7; color: #92400e; }
tbody tr:hover td { filter: brightness(0.95); }
th.sortable { cursor: pointer; user-select: none; }
th.sortable:hover { background: #e0e0e0; }
.si { font-size: 10px; margin-left: 3px; color: #999; }
</style>"""

    js = """
<script>
var _dir = {};
function lbSort(idx, el) {
  var tb = document.querySelector('tbody');
  var rows = Array.from(tb.querySelectorAll('tr'));
  var asc = _dir[idx] !== 'asc';
  _dir[idx] = asc ? 'asc' : 'desc';
  document.querySelectorAll('.si').forEach(function(s){ s.textContent = '⇅'; });
  el.querySelector('.si').textContent = asc ? '↑' : '↓';
  rows.sort(function(a, b) {
    var va = a.cells[idx].textContent.trim().replace(/[+% ]/g, '');
    var vb = b.cells[idx].textContent.trim().replace(/[+% ]/g, '');
    var na = parseFloat(va.replace(',', '.'));
    var nb = parseFloat(vb.replace(',', '.'));
    if (!isNaN(na) && !isNaN(nb)) return asc ? na - nb : nb - na;
    return asc ? va.localeCompare(vb, 'ru') : vb.localeCompare(va, 'ru');
  });
  rows.forEach(function(r){ tb.appendChild(r); });
}
</script>"""

    # thead row 1: grouped headers
    row1 = (
        '<th colspan="6" style="border:1px solid #d8d8d8;background:#f5f5f5"></th>'
        '<th colspan="4" class="grp-quad" style="border:1px solid #d8d8d8">Квадранты</th>'
        '<th colspan="3" class="grp-score" style="border:1px solid #d8d8d8">Метрики</th>'
        '<th colspan="1" style="border:1px solid #d8d8d8;background:#f5f5f5"></th>'
    )

    # thead row 2: sortable individual headers
    row2_cells = []
    for col_idx, (col, label, w, _) in enumerate(_COL_CFG):
        w_style = f"width:{w}px;" if w else ""
        if col == "#":
            row2_cells.append(
                f'<th style="{w_style}text-align:center;border:1px solid #d8d8d8">'
                f"{_html.escape(label)}</th>"
            )
        else:
            row2_cells.append(
                f'<th class="sortable" onclick="lbSort({col_idx}, this)" '
                f'style="{w_style}text-align:center;border:1px solid #d8d8d8">'
                f'{_html.escape(label)}<span class="si">&#x21C5;</span></th>'
            )
    row2 = "".join(row2_cells)
    thead = f"<thead><tr>{row1}</tr><tr>{row2}</tr></thead>"

    # tbody
    def _fmt(col, val):
        if col in ("Mean Closed", "Mean Open"):
            return f"{val:.3f}"
        if col == "GAP":
            return f"{val:+.3f}"
        if col in ("Знает %", "Рассуждает %", "Не справляется %", "Галлюцинирует %"):
            return f"{val:.1f}"
        return _html.escape(str(val)) if val is not None else "—"

    rows_html = ""
    for row_pos, (_, row) in enumerate(df.iterrows()):
        cells = ""
        for col, _, w, align in _COL_CFG:
            w_style  = f"width:{w}px;" if w else ""
            bg_style = f"background-color:{col_colors[col][row_pos]};" if col in col_colors else ""
            cells += (
                f'<td style="text-align:{align};{w_style}{bg_style}">'
                f"{_fmt(col, row[col])}</td>"
            )
        rows_html += f"<tr>{cells}</tr>"

    return f'{css}{js}<table id="{tbl_id}">{thead}<tbody>{rows_html}</tbody></table>'


@st.cache_data
def load_analytics_data(exp_name: str, version: str | None = None) -> list[dict]:
    """Загружает и объединяет gap_records + results + tasks для вкладки Аналитика."""
    exp_dir = resolve_exp_dir(exp_name, version)
    gap_path     = exp_dir / "gap_records.jsonl"
    results_path = exp_dir / "results.jsonl"
    if not gap_path.exists():
        return []

    gap_records: dict[tuple, dict] = {}
    with open(gap_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                gap_records[(r.get("model_name", ""), r.get("task_id", ""))] = r

    results: dict[str, dict] = {}
    if results_path.exists():
        with open(results_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    results[r.get("task_id", "")] = r

    tasks_index: dict[str, dict] = {}
    if TASKS_PATH.exists():
        with open(TASKS_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    t = json.loads(line)
                    tid = t.get("id") or t.get("task_id", "")
                    if tid:
                        base = tid.removesuffix("_closed").removesuffix("_open")
                        tasks_index[base] = t

    def _fv(v) -> float:
        try:
            return round(float(v), 3) if v is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    combined: list[dict] = []
    for (model_name, task_id), g in gap_records.items():
        c = results.get(task_id + "_closed", {})
        o = results.get(task_id + "_open", {})
        t = tasks_index.get(task_id, {})
        combined.append({
            "task_id":       task_id,
            "model_name":    model_name,
            "task_type":     g.get("task_type") or t.get("type", "unknown"),
            "hop_group":     g.get("hop_group") or t.get("hop_group", "unknown"),
            "branch_of_law": g.get("branch_of_law", "unknown"),
            "quadrant":      g.get("quadrant", "incompetent"),
            "closed_final":  _fv(g.get("closed_final_score")),
            "open_final":    _fv(g.get("open_final_score")),
            "gap_final":     _fv(g.get("final_score_gap")),
            "gap_norm_cov":  _fv(g.get("norm_coverage_gap")),
            "gap_step":      _fv(g.get("step_correctness_gap")),
            "gap_answer":    _fv(g.get("answer_correctness_gap")),
            "norm_cov_c":    _fv(c.get("norm_coverage_list")),
            "step_c":        _fv(c.get("step_correctness")),
            "ans_c":         _fv(c.get("answer_correctness")),
            "final_c":       _fv(c.get("final_score")),
            "norm_cov_o":    _fv(o.get("norm_coverage_list")),
            "step_o":        _fv(o.get("step_correctness")),
            "ans_o":         _fv(o.get("answer_correctness")),
            "final_o":       _fv(o.get("final_score")),
            "fabula":        t.get("fabula", ""),
            "question":      t.get("question", ""),
            "answer":        t.get("answer", ""),
            "raw_resp_c":    c.get("raw_response", ""),
            "raw_resp_o":    o.get("raw_response", ""),
            "norm_ids":      t.get("norm_ids", []),
            "gold_chain":    t.get("gold_chain", []),
        })
    return combined


@st.cache_data
def count_successful_records(exp_name: str, version: str | None = None) -> int:
    """Считает записи в results.jsonl без поля 'error' — реальные ответы модели."""
    path = resolve_exp_dir(exp_name, version) / "results.jsonl"
    if not path.exists():
        return 0
    count = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                    if "error" not in r:
                        count += 1
                except json.JSONDecodeError:
                    pass
    return count


@st.cache_data
def load_rta_index(exp_name: str, version: str | None = None) -> dict:
    """
    Загружает results_rta.jsonl → dict[base_task_id → RtA-поля].

    Ключи в возвращаемом dict:
      is_rta_c, rta_type_c, rta_topic_c  — для _closed ответа
      is_rta_o, rta_type_o, rta_topic_o  — для _open ответа
    """
    rta_path = resolve_exp_dir(exp_name, version) / "results_rta.jsonl"
    if not rta_path.exists():
        return {}
    index: dict[str, dict] = {}
    with open(rta_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            tid = r.get("task_id", "")
            base = tid.removesuffix("_closed").removesuffix("_open")
            if base not in index:
                index[base] = {}
            sfx = "_c" if tid.endswith("_closed") else "_o"
            index[base][f"is_rta{sfx}"]   = bool(r.get("is_rta", False))
            index[base][f"rta_type{sfx}"]  = r.get("rta_type")
            index[base][f"rta_topic{sfx}"] = r.get("rta_topic")
    return index


def load_error_analysis(exp_name: str, version: str | None = None) -> dict | None:
    path = resolve_exp_dir(exp_name, version) / "error_analysis.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


_TASK_TYPE_RU = {
    "issue_spotting":    "выявление нарушения",
    "rule_selection":    "выбор нормы",
    "conflict_resolution": "разрешение коллизии",
    "temporal_validity": "действие нормы во времени",
}
_ERROR_CAT_RU = {
    "norm_miss":      "не нашла нормы",
    "reasoning_fail": "ошибка рассуждения",
    "answer_fail":    "неверный вывод",
    "multi_fail":     "ошибка на всех этапах",
    "hallucination":  "галлюцинация",
}
_TASK_TYPE_ORDER = ["issue_spotting", "rule_selection", "conflict_resolution", "temporal_validity"]


def _render_error_analysis_panel(ea: dict, model_label: str = "") -> None:
    pct = ea.get("overall_pct", {})
    n   = ea.get("total", 0)

    name_html = f" <u>{model_label}</u>" if model_label else ""
    st.markdown(
        f'<h3 style="margin-bottom:0;">Общая оценка ответов{name_html}</h3>'
        f'<p style="margin-top:0.15rem;color:#888;font-size:0.85rem;">'
        f'вся выборка, n={n}, порог Final Score ≥ 0.5</p>',
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Верно",                    f"{pct.get('ok', 0)*100:.1f}%")
    c2.metric("Не нашла нормы",           f"{pct.get('norm_miss', 0)*100:.1f}%")
    c3.metric("Ошибка рассуждения",       f"{pct.get('reasoning_fail', 0)*100:.1f}%")
    c4.metric("Неверный вывод",           f"{pct.get('answer_fail', 0)*100:.1f}%")
    c5.metric("Ошибка на всех этапах",    f"{pct.get('multi_fail', 0)*100:.1f}%")

    st.markdown("**Доминирующая ошибка по типу задачи**")
    btt = ea.get("by_task_type", {})
    rows = []
    for tt in _TASK_TYPE_ORDER:
        rec   = btt.get(tt, {})
        total = sum(rec.values()) or 1
        errs  = {k: v for k, v in rec.items() if k != "ok"}
        if not errs:
            top_label = "—"
        else:
            top = max(errs, key=errs.get)
            top_pct = errs[top] / total * 100
            ru = _ERROR_CAT_RU.get(top, top)
            top_label = f"{top} ({ru}) — {top_pct:.0f}%"
        tt_ru = _TASK_TYPE_RU.get(tt, tt)
        rows.append({"Тип задачи": f"{tt} ({tt_ru})", "Доминирующая ошибка": top_label})
    st.dataframe(rows, use_container_width=True, hide_index=True)

    with st.expander("ℹ️ Как считается классификация ошибок"):
        st.markdown("""
**Final Score = α · Norm Coverage + β · Step Correctness + γ · Answer Correctness**

где α = β = γ = ⅓ — все три компонента равнозначны.

Каждый ответ оценивается по трём слагаемым формулы, после чего получает категорию ошибки.

**Компоненты оценки:**

| Компонент | Что измеряет | Шкала | Порог «провала» |
|---|---|---|---|
| Norm Coverage | Нашла ли модель нужные нормы — F1 относительно эталонного списка `gold_articles` | [0, 1] | < 0.3 |
| Step Correctness | Правильно ли выстроена цепочка рассуждений — оценка каждого шага `gold_chain` | {0, 0.5, 1} → среднее | < 0.4 |
| Answer Correctness | Верен ли финальный вывод | {0, 0.33, 0.67, 1} | < 0.33 |

Ответы с Final Score ≥ 0.5 считаются корректными («Верно»). Для остальных определяется категория ошибки по тому, какой компонент формулы провалился:

**Категории ошибок** (применяются к ответам с Final Score < 0.5):

| Категория | Условие | Смысл |
|---|---|---|
| Не нашла нормы | Norm Coverage < 0.3 | Менее 30% нужных норм найдено; рассуждение и вывод могут быть частично верны |
| Ошибка рассуждения | Step Correctness < 0.4 | Нормы нашла, но логика рассуждения по шагам `gold_chain` содержит ошибки |
| Неверный вывод | Answer Correctness < 0.33 | Рассуждение шло правильно, но финальный вывод неверен |
| Ошибка на всех этапах | Одновременный провал: norm_coverage_list < 0.3 **И** step_correctness < 0.4 **И** answer_correctness < 0.33 | Системный сбой — нет одного слабого места; проверяется в первую очередь |

**Важно:** статистика считается по всем 600 записям (300 closed-book + 300 open-book вместе), без разделения по режиму.
""")  # noqa: E501


# ---------------------------------------------------------------------------
# Построение подграфа для визуализации
# ---------------------------------------------------------------------------

def get_subgraph(
    G: nx.DiGraph,
    corpus: dict,
    selected_laws: list[str],
    selected_edge_types: list[str],
    include_ref_edges: bool,
    max_nodes: int,
    hide_repealed: bool = True,
) -> nx.DiGraph:
    active_types = set(selected_edge_types)
    if include_ref_edges:
        active_types.add("ссылается_на")

    repealed_nodes: set[str] = set()
    if hide_repealed:
        repealed_nodes = {nid for nid, nd in G.nodes(data=True) if nd.get("is_repealed")}
        repealed_nodes |= {nid for nid, art in corpus.items() if art.get("is_repealed")}

    sub_edges = []
    for u, v, d in G.edges(data=True):
        if hide_repealed and (u in repealed_nodes or v in repealed_nodes):
            continue
        et = d.get("edge_type", "ссылается_на")
        if et not in active_types:
            continue
        u_law = u.split("_")[0] if "_" in u else ""
        v_law = v.split("_")[0] if "_" in v else ""
        if selected_laws and not (u_law in selected_laws or v_law in selected_laws):
            continue
        sub_edges.append((u, v, d))

    laws_in_use = selected_laws if selected_laws else list({
        u.split("_")[0] for u, _, _ in sub_edges if "_" in u
    })
    per_law_limit = max(max_nodes // max(len(laws_in_use), 1), 10)

    law_node_counts: dict[str, int] = defaultdict(int)
    seen_nodes: set[str] = set()
    filtered_edges = []

    for u, v, d in sub_edges:
        u_law = u.split("_")[0] if "_" in u else ""
        v_law = v.split("_")[0] if "_" in v else ""
        primary = u_law if u_law in laws_in_use else v_law

        if law_node_counts[primary] >= per_law_limit:
            continue

        new_u = u not in seen_nodes
        new_v = v not in seen_nodes
        seen_nodes.add(u)
        seen_nodes.add(v)
        filtered_edges.append((u, v, d))
        if new_u:
            law_node_counts[u_law] += 1
        if new_v:
            law_node_counts[v_law] += 1

    SG = nx.DiGraph()
    for node in seen_nodes:
        law = node.split("_")[0] if "_" in node else ""
        art = corpus.get(node, {})
        SG.add_node(
            node,
            law=law,
            article=art.get("article", node),
            title=art.get("title", ""),
            valid_from=art.get("valid_from") or "—",
            valid_to=art.get("valid_to") or "—",
            is_repealed=bool(art.get("is_repealed")),
            has_repealed_parts=bool(art.get("has_repealed_parts")),
            version=art.get("version") or "—",
            text_preview=art.get("text", ""),
        )
    for u, v, d in filtered_edges:
        SG.add_edge(u, v, **d)

    return SG


# ---------------------------------------------------------------------------
# Построение PyVis сети
# ---------------------------------------------------------------------------

def build_pyvis(SG: nx.DiGraph, physics: bool) -> Network:
    net = Network(
        height="700px",
        width="100%",
        directed=True,
        bgcolor="#1a1a2e",
        font_color="#ffffff",
        notebook=False,
        cdn_resources="in_line",
    )

    net.set_options("""
    {
      "nodes": {
        "borderWidth": 2,
        "font": { "size": 14, "face": "Arial" },
        "shadow": true
      },
      "edges": {
        "arrows": { "to": { "enabled": true, "scaleFactor": 0.8 } },
        "smooth": { "type": "dynamic" },
        "font": { "size": 11, "align": "middle" },
        "shadow": true
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 9999999,
        "navigationButtons": true,
        "keyboard": true,
        "multiselect": true
      },
      "physics": {
        "enabled": """ + str(physics).lower() + """,
        "barnesHut": {
          "gravitationalConstant": -8000,
          "centralGravity": 0.3,
          "springLength": 150,
          "springConstant": 0.04,
          "damping": 0.09
        },
        "stabilization": { "iterations": 200 }
      }
    }
    """)

    for node, d in SG.nodes(data=True):
        law = d.get("law", "")
        is_repealed = d.get("is_repealed", False)
        has_repealed_parts = d.get("has_repealed_parts", False)

        if is_repealed:
            color = "#95a5a6"
            border_color = "#7f8c8d"
            border_width = 3
        elif has_repealed_parts:
            color = NODE_COLORS.get(law, "#ecf0f1")
            border_color = "#e74c3c"
            border_width = 2
        else:
            color = NODE_COLORS.get(law, "#ecf0f1")
            border_color = "#2c3e50"
            border_width = 2

        status_line = ""
        if is_repealed:
            vto = d.get("valid_to", "—")
            status_line = f"<br><b style='color:#e74c3c'>⚠ Утратила силу{(' с ' + vto) if vto != '—' else ''}</b>"
        elif has_repealed_parts:
            status_line = "<br><span style='color:#f39c12'>⚠ Частичная утрата силы</span>"

        text_html = d.get("text_preview", "").replace("\n", "<br>")
        tooltip = (
            f"<b>{node}</b>{status_line}<br>"
            f"Кодекс: {law}<br>"
            f"Статья: {d.get('article', '—')}<br>"
            f"Название: {d.get('title', '—')}<br>"
            f"Редакция: {d.get('version', '—')}<br>"
            f"Дата редакции: {d.get('valid_from', '—')}<br><br>"
            f"<i>{text_html}</i>"
        )
        net.add_node(
            node,
            label=node,
            title=tooltip,
            color={"background": color, "border": border_color,
                   "highlight": {"background": color, "border": border_color}},
            size=20 + SG.in_degree(node) * 3,
            borderWidth=border_width,
        )

    for u, v, d in SG.edges(data=True):
        et = d.get("edge_type", "ссылается_на")
        color = EDGE_COLORS.get(et, "#bdc3c7")
        explanation = d.get("explanation", "")
        tooltip = f"<b>{et}</b><br>{explanation}" if explanation else et
        net.add_edge(
            u, v,
            title=tooltip,
            label=et if et != "ссылается_на" else "",
            color=color,
            width=3 if et != "ссылается_на" else 1,
            arrows="to",
        )

    return net


# ---------------------------------------------------------------------------
# Рендер задачи (вкладка Задачи)
# ---------------------------------------------------------------------------

def _badge(text: str, bg: str, fg: str = "#fff") -> str:
    return (
        f'<span style="background:{bg};color:{fg};padding:3px 10px;'
        f'border-radius:12px;font-size:12px;font-weight:600;'
        f'margin-right:6px;display:inline-block">{text}</span>'
    )


def render_task_detail(task: dict, corpus: dict) -> None:
    task_type = task.get("type", "")
    mode      = task.get("mode", "")
    hop_group = task.get("hop_group", "")
    sens      = task.get("sensitivity")
    norm_ids  = task.get("norm_ids", [])

    HOP_COLORS = {"shallow": "#27ae60", "medium": "#e67e22", "deep": "#e74c3c"}

    # Бейджи
    badges = ""
    badges += _badge(TYPE_RU.get(task_type, task_type), TYPE_COLORS.get(task_type, "#888"))
    badges += _badge(
        "📖 Open-book" if mode == "open" else "🔒 Closed-book",
        "#2980b9" if mode == "open" else "#7f8c8d",
    )
    badges += _badge(hop_group.capitalize(), HOP_COLORS.get(hop_group, "#888"))
    if sens is not None:
        badges += _badge(f"Sensitivity {sens}", "#8e44ad")
    st.markdown(badges, unsafe_allow_html=True)

    # Нормы
    if norm_ids:
        norm_tags = " ".join(
            f'<code style="background:#2c3e50;color:#ecf0f1;padding:2px 7px;border-radius:4px;font-size:12px">{n}</code>'
            for n in norm_ids
        )
        st.markdown(f"**Нормы:** {norm_tags}", unsafe_allow_html=True)

    st.markdown("---")

    # Фабула
    st.markdown("**Фабула**")
    st.markdown(
        f'<div style="background:#f8f9fa;border-left:4px solid #3498db;'
        f'padding:12px 16px;border-radius:4px;font-size:14px;line-height:1.7">'
        f'{task.get("fabula","")}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("")

    # Вопрос
    st.markdown("**Вопрос**")
    st.markdown(
        f'<div style="background:#f8f9fa;border-left:4px solid #2ecc71;'
        f'padding:12px 16px;border-radius:4px;font-size:14px;line-height:1.7">'
        f'{task.get("question","")}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("")

    # Ответ
    st.markdown("**Эталонный ответ**")
    st.markdown(
        f'<div style="background:#f8f9fa;border-left:4px solid #e67e22;'
        f'padding:12px 16px;border-radius:4px;font-size:14px;line-height:1.7">'
        f'{task.get("answer","")}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("")

    # Gold Chain
    gold_chain = task.get("gold_chain") or []
    if gold_chain:
        with st.expander("🔗 Gold Chain (цепочка рассуждений)", expanded=True):
            for step in gold_chain:
                step_n   = step.get("step", "?")
                norm_id  = step.get("norm_id")
                reasoning = step.get("reasoning") or step.get("conclusion", "")
                if norm_id:
                    header = f"Шаг {step_n} — `{norm_id}`"
                else:
                    header = f"Шаг {step_n} — **Вывод**"
                st.markdown(
                    f'<div style="background:#eaf4fb;border:1px solid #aed6f1;'
                    f'border-radius:6px;padding:10px 14px;margin-bottom:8px;font-size:13px">'
                    f'<b>{header}</b><br>{reasoning}</div>',
                    unsafe_allow_html=True,
                )

    # Context chunks (только open-book)
    chunks = task.get("context_chunks") or []
    if chunks:
        with st.expander(f"📄 Context Chunks ({len(chunks)} норм)", expanded=False):
            for chunk in chunks:
                nid  = chunk.get("norm_id", "")
                text = chunk.get("text", "")
                art  = corpus.get(nid, {})
                title    = art.get("title", "")
                header_  = f"{nid}" + (f" — {title}" if title else "")
                st.markdown(f"**{header_}**")
                st.markdown(
                    f'<div style="background:#f9f9f9;border:1px solid #ddd;'
                    f'border-radius:4px;padding:10px 14px;font-size:13px;'
                    f'line-height:1.7;max-height:300px;overflow-y:auto;'
                    f'white-space:pre-wrap">{text}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown("")
    elif mode == "open":
        st.info("Context chunks отсутствуют (open-book задача без чанков).")

    # edge_meta (опционально)
    edge_meta = task.get("edge_meta") or []
    if edge_meta:
        with st.expander("🔍 Edge meta (рёбра графа)", expanded=False):
            for e in edge_meta:
                st.markdown(
                    f'`{e.get("source","")}` → `{e.get("target","")}` '
                    f'**{e.get("edge_type","")}** — {e.get("explanation","")}',
                )


# ---------------------------------------------------------------------------
# Quadrant figure (вкладка Аналитика)
# ---------------------------------------------------------------------------

def _build_radar_figure(df: pd.DataFrame, selected_names: list[str]) -> "go.Figure":
    """Spider/radar chart для сравнения выбранных моделей."""
    theta = ["Mean Closed", "Mean Open", "Знает", "Рассуждает",
             "1−Не справ.", "1−Галлюц."]
    theta_closed = theta + [theta[0]]
    colors = ["#3498db", "#e74c3c", "#27ae60", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22"]
    fig = go.Figure()
    for i, mname in enumerate(selected_names):
        row = df[df["_model_name"] == mname]
        if row.empty:
            continue
        r = row.iloc[0]
        vals = [
            r["Mean Closed"],
            r["Mean Open"],
            r["Знает %"] / 100,
            r["Рассуждает %"] / 100,
            1 - r["Не справляется %"] / 100,
            1 - r["Галлюцинирует %"] / 100,
        ]
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=theta_closed,
            fill="toself",
            name=r["Модель"],
            line_color=colors[i % len(colors)],
            opacity=0.65,
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickfont_size=10)),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
        height=430,
        margin=dict(t=40, b=60, l=40, r=40),
        title=dict(text="Сравнение моделей", font_size=14),
    )
    return fig


_BRANCH_MAP: dict[str, str] = {
    "КРФ": "конституционное", "ГК": "гражданское",   "ТК": "трудовое",
    "КоАП": "административное", "УК": "уголовное",   "СК": "семейное",
    "НК": "налоговое",          "ЗК": "земельное",   "ЖК": "жилищное",
    "ГрК": "градостроительное",
}

_QUAD_COLORS = {
    "reasons":      "#2ecc71",
    "knows":        "#3498db",
    "incompetent":  "#95a5a6",
    "hallucinates": "#e74c3c",
}
# Русские названия квадрантов (open-book / closed-book не переводятся)
_QUAD_RU = {
    "knows":        "Знает",
    "reasons":      "Рассуждает",
    "hallucinates": "Галлюцинирует",
    "incompetent":  "Не справляется",
}
# Угловые позиции подписей внутри квадрантов (крайние углы — не перекрывают точки)
_QUAD_CORNER = {
    "knows":        (0.98, 0.98, "right", "top"),
    "reasons":      (0.02, 0.98, "left",  "top"),
    "hallucinates": (0.98, 0.02, "right", "bottom"),
    "incompetent":  (0.02, 0.02, "left",  "bottom"),
}


def _build_quadrant_figure(records: list[dict], rta_task_ids: set | None = None, n_rta_total: int | None = None):
    fig = go.Figure()

    # Фон квадрантов (x = closed-book, y = open-book) — слабый, чтобы не перекрывать точки
    for (x0, y0, x1, y1), qname in [
        ((0.5, 0.5, 1.0, 1.0), "knows"),
        ((0.0, 0.5, 0.5, 1.0), "reasons"),
        ((0.5, 0.0, 1.0, 0.5), "hallucinates"),
        ((0.0, 0.0, 0.5, 0.5), "incompetent"),
    ]:
        fig.add_shape(
            type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
            fillcolor=_QUAD_COLORS[qname], opacity=0.06,
            line_width=0, layer="below",
        )

    # Диагональ и разделители
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                  line=dict(color="rgba(150,150,150,0.4)", width=1, dash="dash"))
    fig.add_shape(type="line", x0=0.5, y0=0, x1=0.5, y1=1,
                  line=dict(color="rgba(150,150,150,0.35)", width=1))
    fig.add_shape(type="line", x0=0, y0=0.5, x1=1, y1=0.5,
                  line=dict(color="rgba(150,150,150,0.35)", width=1))

    # Группировка точек по квадрантам
    by_quad: dict[str, list] = {q: [] for q in _QUAD_COLORS}
    for r in records:
        q = r.get("quadrant", "incompetent")
        by_quad.setdefault(q, []).append(r)

    n_total = len(records)

    # Добавляем трассы — все квадранты, даже пустые (чтобы легенда была полной)
    for qname in _QUAD_COLORS:
        pts = by_quad.get(qname, [])
        n   = len(pts)
        pct = f"{100*n/n_total:.0f}%" if n_total > 0 else "0%"
        # Название в легенде содержит все: имя, количество, процент
        trace_name = f"<b>{_QUAD_RU[qname]}</b>  {n} зад. ({pct})"

        customdata = [
            [
                p["task_id"], p["model_name"], p["task_type"], p["hop_group"],
                p["fabula"], p["question"], p["answer"],
                p["raw_resp_c"], p["raw_resp_o"],
                p["norm_cov_c"], p["step_c"], p["ans_c"], p["final_c"],
                p["norm_cov_o"], p["step_o"], p["ans_o"], p["final_o"],
                p["gap_final"], p["gap_norm_cov"], p["gap_step"], p["gap_answer"],
                p["quadrant"],
                "|".join(p.get("norm_ids", [])),
            ]
            for p in pts
        ] if pts else []

        fig.add_trace(go.Scatter(
            x=[p["closed_final"] for p in pts] if pts else [None],
            y=[p["open_final"]   for p in pts] if pts else [None],
            mode="markers",
            name=trace_name,
            marker=dict(
                color=_QUAD_COLORS[qname],
                size=14,
                line=dict(width=2, color="white"),
                opacity=1.0,
            ),
            customdata=customdata if customdata else [[None] * 23],
            hovertemplate=(
                "<b>%{customdata[1]}</b><br>"
                "Тип: %{customdata[2]} · %{customdata[3]}<br>"
                "Closed-book: %{x:.3f}  |  Open-book: %{y:.3f}<br>"
                "GAP: %{customdata[17]:+.3f}<br>"
                "<i>Нажми для деталей</i><extra></extra>"
            ) if pts else "<extra></extra>",
            showlegend=True,
        ))

    # RtA-оверлей: X-маркеры поверх обычных точек
    # customdata — те же 23 поля, что и в основных трейсах, чтобы обработчик клика работал
    if rta_task_ids:
        rta_pts = [r for r in records if r.get("task_id") in rta_task_ids]
        if rta_pts:
            rta_cd = [
                [
                    p["task_id"], p["model_name"], p["task_type"], p["hop_group"],
                    p["fabula"], p["question"], p["answer"],
                    p["raw_resp_c"], p["raw_resp_o"],
                    p["norm_cov_c"], p["step_c"], p["ans_c"], p["final_c"],
                    p["norm_cov_o"], p["step_o"], p["ans_o"], p["final_o"],
                    p["gap_final"], p["gap_norm_cov"], p["gap_step"], p["gap_answer"],
                    p["quadrant"],
                    "|".join(p.get("norm_ids", [])),
                ]
                for p in rta_pts
            ]
            fig.add_trace(go.Scatter(
                x=[p["closed_final"] for p in rta_pts],
                y=[p["open_final"]   for p in rta_pts],
                mode="markers",
                name=f"⚠️ RtA-отказ ({n_rta_total if n_rta_total is not None else len(rta_pts)} шт / {len(rta_pts)} меток)",
                marker=dict(
                    symbol="x", size=14, color="rgba(0,0,0,0)",
                    line=dict(width=2.5, color="#E32636"),
                ),
                hovertemplate=(
                    "<b>⚠️ RtA: %{customdata[1]}</b><br>"
                    "Тип: %{customdata[2]} · %{customdata[3]}<br>"
                    "Closed: %{x:.3f} | Open: %{y:.3f}<br>"
                    "<i>Нажми для деталей</i><extra></extra>"
                ),
                customdata=rta_cd,
                showlegend=True,
            ))

    # Подписи квадрантов — в УГЛАХ области, не перекрывают центральные точки
    for qname, (px, py, xanch, yanch) in _QUAD_CORNER.items():
        fig.add_annotation(
            x=px, y=py,
            xref="x domain", yref="y domain",
            text=f"<i>{_QUAD_RU[qname]}</i>",
            showarrow=False,
            xanchor=xanch, yanchor=yanch,
            font=dict(size=10, color=_QUAD_COLORS[qname]),
            opacity=0.75,
        )

    fig.update_layout(
        xaxis=dict(
            title="Closed-book (балл)", range=[-0.05, 1.05], dtick=0.25,
            gridcolor="#ebebeb", zeroline=False,
        ),
        yaxis=dict(
            title="Open-book (балл)", range=[-0.05, 1.05], dtick=0.25,
            gridcolor="#ebebeb", zeroline=False,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=60, r=20, t=30, b=110),
        height=540,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top", y=-0.17,
            xanchor="center", x=0.5,
            font=dict(size=12),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#ddd", borderwidth=1,
        ),
    )
    return fig


def _build_gap_by_type_figure(records: list[dict]):
    """Сгруппированный bar: mean Closed / Open / GAP по типу задачи."""
    agg: dict[str, dict[str, list]] = defaultdict(lambda: {"closed": [], "open": [], "gap": []})
    for r in records:
        tt = r.get("task_type", "unknown")
        agg[tt]["closed"].append(r.get("closed_final", 0.0))
        agg[tt]["open"].append(r.get("open_final", 0.0))
        agg[tt]["gap"].append(r.get("gap_final", 0.0))
    types = [t for t in ["issue_spotting", "rule_selection", "conflict_resolution", "temporal_validity"] if t in agg]
    labels = [TYPE_RU.get(t, t) for t in types]
    fig = go.Figure()
    for key, color, name in [("closed", "#3498db", "Closed"), ("open", "#2ecc71", "Open"), ("gap", "#E32636", "GAP")]:
        fig.add_trace(go.Bar(
            name=name, x=labels,
            y=[round(sum(agg[t][key]) / len(agg[t][key]), 3) if agg[t][key] else 0 for t in types],
            marker_color=color,
        ))
    fig.update_layout(
        title="Метрики по типу задач", barmode="group", height=310,
        margin=dict(t=40, b=60, l=50, r=10),
        legend=dict(orientation="h", y=-0.28, x=0.5, xanchor="center", font=dict(size=11)),
        yaxis=dict(range=[0, 1], dtick=0.25, gridcolor="#ebebeb"),
        plot_bgcolor="white", paper_bgcolor="white", font=dict(size=11),
    )
    return fig


def _build_quadrant_dist_figure(records: list[dict]):
    """Стекированный bar: распределение квадрантов по типу задачи.
    Сегменты отсортированы по суммарному кол-ву: самый большой внизу."""
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {q: 0 for q in _QUAD_COLORS})
    for r in records:
        tt = r.get("task_type", "unknown")
        q  = r.get("quadrant", "incompetent")
        counts[tt][q] = counts[tt].get(q, 0) + 1
    types = [t for t in ["issue_spotting", "rule_selection", "conflict_resolution", "temporal_validity"] if t in counts]
    labels = [TYPE_RU.get(t, t) for t in types]
    # Сортируем квадранты по суммарному кол-ву по убыванию: первый (самый большой) → дно стека
    totals = {q: sum(counts[t].get(q, 0) for t in types) for q in _QUAD_COLORS}
    sorted_quads = sorted(_QUAD_COLORS, key=lambda q: totals[q], reverse=True)
    fig = go.Figure()
    for qname in sorted_quads:
        fig.add_trace(go.Bar(
            name=_QUAD_RU.get(qname, qname),
            x=labels,
            y=[counts[t].get(qname, 0) for t in types],
            marker_color=_QUAD_COLORS[qname],
        ))
    fig.update_layout(
        title="Квадранты по типу задач", barmode="stack", height=310,
        margin=dict(t=40, b=70, l=50, r=10),
        legend=dict(orientation="h", y=-0.35, x=0.5, xanchor="center", font=dict(size=11)),
        yaxis=dict(gridcolor="#ebebeb"),
        plot_bgcolor="white", paper_bgcolor="white", font=dict(size=11),
    )
    return fig


def _build_gap_by_hop_figure(records: list[dict]):
    """Сгруппированный bar: mean Closed / Open / GAP по глубине цепочки."""
    agg: dict[str, dict[str, list]] = defaultdict(lambda: {"closed": [], "open": [], "gap": []})
    for r in records:
        hg = r.get("hop_group", "unknown")
        agg[hg]["closed"].append(r.get("closed_final", 0.0))
        agg[hg]["open"].append(r.get("open_final", 0.0))
        agg[hg]["gap"].append(r.get("gap_final", 0.0))
    hops = [h for h in ["shallow", "medium", "deep"] if h in agg]
    fig = go.Figure()
    for key, color, name in [("closed", "#3498db", "Closed"), ("open", "#2ecc71", "Open"), ("gap", "#E32636", "GAP")]:
        fig.add_trace(go.Bar(
            name=name, x=hops,
            y=[round(sum(agg[h][key]) / len(agg[h][key]), 3) if agg[h][key] else 0 for h in hops],
            marker_color=color,
        ))
    fig.update_layout(
        title="Метрики по глубине цепочки", barmode="group", height=310,
        margin=dict(t=40, b=60, l=50, r=10),
        legend=dict(orientation="h", y=-0.28, x=0.5, xanchor="center", font=dict(size=11)),
        yaxis=dict(range=[0, 1], dtick=0.25, gridcolor="#ebebeb"),
        plot_bgcolor="white", paper_bgcolor="white", font=dict(size=11),
    )
    return fig


def _build_rta_by_topic_figure(records: list[dict]) -> "go.Figure":
    """Столбчатый график: RtA-ставка по теме отказа и режиму."""
    from collections import defaultdict
    agg: dict[str, dict] = defaultdict(lambda: {"closed": [0, 0], "open": [0, 0]})
    for r in records:
        for sfx, mode_key in (("_c", "closed"), ("_o", "open")):
            is_rta = r.get(f"is_rta{sfx}", False)
            topic = r.get(f"rta_topic{sfx}") or ("other" if is_rta else None)
            if topic is None:
                continue
            agg[topic][mode_key][0] += 1
            if is_rta:
                agg[topic][mode_key][1] += 1

    topics = sorted(agg.keys())
    if not topics:
        fig = go.Figure()
        fig.update_layout(title="RtA-ставка по теме отказа", height=310,
                          plot_bgcolor="white", paper_bgcolor="white")
        return fig

    fig = go.Figure()
    for mode_key, color, name in (
        ("closed", "#6c8ebf", "Closed-book"),
        ("open",   "#82b366", "Open-book"),
    ):
        rates = [
            agg[t][mode_key][1] / agg[t][mode_key][0]
            if agg[t][mode_key][0] else 0
            for t in topics
        ]
        fig.add_trace(go.Bar(name=name, x=topics, y=rates, marker_color=color))

    fig.update_layout(
        title="RtA-ставка по теме отказа", barmode="group", height=310,
        margin=dict(t=40, b=80, l=50, r=10),
        yaxis=dict(range=[0, 1], tickformat=".0%", gridcolor="#ebebeb"),
        legend=dict(orientation="h", y=-0.38, x=0.5, xanchor="center", font=dict(size=11)),
        plot_bgcolor="white", paper_bgcolor="white", font=dict(size=11),
    )
    return fig


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="MHGB Graph Explorer",
    page_icon="⚖️",
    layout="wide",
)

if not ALLOW_EXPORT:
    # Hide the element toolbar (download / fullscreen / search) that sits above
    # st.dataframe: its CSV button exports the whole filtered table at once.
    st.markdown(
        """
        <style>
          [data-testid="stElementToolbar"] { display: none !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

st.title("⚖️ MHGB — Legal Multi-hop Graph Bench")

# Загрузка
_versions = load_graph_versions()
_selected_path = str(GRAPH_PATH)
if _versions:
    _labels = [f"{v['tag']}  ({v['node_count']} нод, {v['edge_count']} рёбер)" for v in _versions]
    _chosen = st.selectbox("Версия графа", _labels, index=0, key="graph_version_select")
    _chosen_idx = _labels.index(_chosen)
    _fp = _versions[_chosen_idx].get("file_path")
    if _fp and Path(_fp).exists():
        _selected_path = _fp

G      = load_graph(_selected_path)
corpus = load_corpus()
tasks  = load_tasks()

edge_type_counts = Counter(
    d.get("edge_type", "ссылается_на") for _, _, d in G.edges(data=True)
)

# ---------------------------------------------------------------------------
# Навигация: разделы в сайдбаре (раньше здесь жили фильтры графа — они
# переехали в тело раздела «Граф», между метриками и самим графом)
# ---------------------------------------------------------------------------

PAGE_GRAPH       = "🗺 Граф"
PAGE_TASKS       = "📋 Задачи"
PAGE_ANALYTICS   = "📊 Аналитика"
PAGE_LEADERBOARD = "🏆 Лидерборд"
PAGES = [PAGE_GRAPH, PAGE_TASKS, PAGE_ANALYTICS, PAGE_LEADERBOARD]

# Меню из кнопок во всю ширину: активный раздел выделен цветом. Кнопки заметнее
# радио-списка, а высота строки задаётся стилем ниже.
st.markdown(
    """
    <style>
      section[data-testid="stSidebar"] div.stButton > button {
          font-size: 1.05rem;
          font-weight: 600;
          padding: 0.6rem 0.9rem;
          text-align: left;
          justify-content: flex-start;
          margin-bottom: 0.25rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.session_state.setdefault("nav_page", PAGE_GRAPH)

with st.sidebar:
    st.markdown("### Разделы")
    for _p in PAGES:
        _is_active = st.session_state["nav_page"] == _p
        if st.button(
            _p,
            use_container_width=True,
            type="primary" if _is_active else "secondary",
            key=f"nav_btn_{_p}",
        ):
            # Перерисовываем сразу, иначе подсветка активного раздела отстала бы
            # на одно нажатие: кнопки выше уже отрисованы с прежним значением.
            st.session_state["nav_page"] = _p
            st.rerun()

page = st.session_state["nav_page"]


# ---------------------------------------------------------------------------
# Вкладки
# ---------------------------------------------------------------------------

# Разделы выбираются кнопками в сайдбаре (см. блок «Навигация» выше).
# Выполняется тело только выбранного раздела, а не всех сразу, как было с st.tabs.

# ===========================================================================
# Раздел: Граф
# ===========================================================================

if page == PAGE_GRAPH:
    # Метрики считаются по подграфу, а подграф — по фильтрам, которые лежат НИЖЕ.
    # Резервируем место контейнером и заполняем его, когда подграф посчитан:
    # так на экране метрики остаются сверху, а фильтры — под ними.
    _metrics_slot = st.container()

    # ── Фильтры графа ────────────────────────────────────────────────────────
    # Значения хранятся в st.session_state под своими ключами и подставляются как
    # default/value. Без этого фильтры сбрасывались бы при уходе в другой раздел:
    # виджеты неотрисованного раздела Streamlit не сохраняет.
    _ss = st.session_state
    _ss.setdefault("flt_laws", ["ТК"])
    _ss.setdefault("flt_edge_types", ["исключает", "дополняет", "приоритет", "применяется_к"])
    _ss.setdefault("flt_include_refs", False)
    _ss.setdefault("flt_hide_repealed", True)
    _ss.setdefault("flt_max_nodes", 150)
    _ss.setdefault("flt_physics", True)

    _f_laws, _f_edges, _f_view = st.columns([3, 3, 3], gap="large")

    with _f_laws:
        all_laws = ["ТК", "ГК", "КоАП", "УК", "НК", "СК", "ЖК", "ЗК", "ГрК", "КРФ"]
        selected_laws = st.multiselect(
            "Кодексы",
            all_laws,
            default=_ss["flt_laws"],
            help="Показывать статьи из выбранных кодексов",
        )
        _ss["flt_laws"] = selected_laws

    with _f_edges:
        st.markdown("**Типы рёбер**")
        semantic_types = ["исключает", "дополняет", "приоритет", "применяется_к"]
        selected_edge_types = []
        for et in semantic_types:
            cnt = edge_type_counts.get(et, 0)
            if st.checkbox(f"{et} ({cnt})", value=et in _ss["flt_edge_types"]):
                selected_edge_types.append(et)
        _ss["flt_edge_types"] = selected_edge_types
        include_refs = st.checkbox(
            f"ссылается_на ({edge_type_counts.get('ссылается_на', 0)})",
            value=_ss["flt_include_refs"],
            help="Включить нейтральные ссылки (много рёбер, может замедлить)",
        )
        _ss["flt_include_refs"] = include_refs

    with _f_view:
        st.markdown("**Отображение**")
        hide_repealed = st.checkbox("Скрыть утратившие силу", value=_ss["flt_hide_repealed"])
        max_nodes     = st.slider("Макс. нод", 20, 500, _ss["flt_max_nodes"], step=10)
        physics_on    = st.checkbox("Физика (авто-расстановка)", value=_ss["flt_physics"])
        _ss["flt_hide_repealed"] = hide_repealed
        _ss["flt_max_nodes"]     = max_nodes
        _ss["flt_physics"]       = physics_on

    # ── Легенда: три группы в одну строку, без вертикальных отступов ─────────
    _chip = ("display:inline-block;padding:2px 8px;border-radius:4px;"
             "color:#000;font-size:12px;margin:0 6px 4px 0;white-space:nowrap")
    _edge_chips = "".join(
        f'<span style="background:{color};{_chip}">{et}</span>'
        for et, color in EDGE_COLORS.items()
        if not (et == "ссылается_на" and not include_refs)
    )
    _node_chips = (
        f'<span style="background:#95a5a6;{_chip}">⚠ Утратила силу</span>'
        f'<span style="background:#f0f0f0;border:2px solid #e74c3c;{_chip}">'
        f'⚠ Частичная утрата</span>'
    )
    _l1, _l2 = st.columns([5, 4], gap="small")
    with _l1:
        st.markdown(
            f'<div style="line-height:1.9"><b style="font-size:13px">Рёбра:</b> '
            f'{_edge_chips}</div>',
            unsafe_allow_html=True,
        )
    with _l2:
        st.markdown(
            f'<div style="line-height:1.9"><b style="font-size:13px">Ноды:</b> '
            f'{_node_chips}'
            f'<span style="font-size:12px;color:#666">размер = входящие ссылки</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    SG = get_subgraph(
        G, corpus,
        selected_laws=selected_laws,
        selected_edge_types=selected_edge_types,
        include_ref_edges=include_refs,
        max_nodes=max_nodes,
        hide_repealed=hide_repealed,
    )

    with _metrics_slot:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Нод (подграф)", SG.number_of_nodes())
        col2.metric("Рёбер (подграф)", SG.number_of_edges())
        col3.metric("Всего нод", G.number_of_nodes())
        col4.metric("Всего рёбер", G.number_of_edges())

    if SG.number_of_nodes() == 0:
        st.warning("Нет данных для выбранных фильтров.")
    else:
        net = build_pyvis(SG, physics=physics_on)
        html_content = net.generate_html()

        panel_html = """
        <div id="mhgb-panel" style="
            display:none; position:fixed; top:70px; right:20px;
            width:380px; max-height:560px; overflow-y:scroll;
            background:#1e2a3a; color:#e8eaf0;
            border:1px solid #3498db; border-radius:10px;
            padding:14px 16px; z-index:9999;
            font-size:13px; line-height:1.6;
            box-shadow: 0 6px 24px rgba(0,0,0,0.6);
            font-family: Arial, sans-serif;
            word-break:break-word;
        ">
          <button onclick="document.getElementById('mhgb-panel').style.display='none'"
            style="float:right; background:none; border:none; color:#aaa;
                   font-size:18px; cursor:pointer; line-height:1; margin:-4px -4px 0 0">✕</button>
          <div id="mhgb-panel-content"></div>
        </div>

        <script>
        var _mhgbTimer = setInterval(function() {
          if (typeof network !== 'undefined') {
            clearInterval(_mhgbTimer);
            network.on("click", function(params) {
              if (params.nodes.length > 0) {
                var nodeId = params.nodes[0];
                var nodeData = network.body.data.nodes.get(nodeId);
                document.getElementById('mhgb-panel-content').innerHTML = nodeData.title || "<b>" + nodeId + "</b>";
                document.getElementById('mhgb-panel').style.display = 'block';
              }
            });
          }
        }, 100);
        </script>
        """
        html_content = html_content.replace("</body>", panel_html + "</body>")
        st.components.v1.html(html_content, height=720, scrolling=False)
        st.caption("Hover по ноде — метаданные статьи. Hover по ребру — тип связи и пояснение LLM. Ctrl+scroll — зум.")

    # Семантические рёбра
    st.markdown("---")
    st.subheader("Семантические рёбра (LLM)")

    sem_rows = []
    for u, v, d in G.edges(data=True):
        et = d.get("edge_type", "ссылается_на")
        if et == "ссылается_на":
            continue
        u_law = u.split("_")[0] if "_" in u else ""
        v_law = v.split("_")[0] if "_" in v else ""
        if selected_laws and not (u_law in selected_laws or v_law in selected_laws):
            continue
        sem_rows.append({
            "Источник": u,
            "Тип": et,
            "Цель": v,
            "Пояснение": d.get("explanation", ""),
        })

    if sem_rows:
        st.dataframe(sem_rows, use_container_width=True, height=300)
    else:
        st.info("Нет семантических рёбер для выбранных фильтров.")


# ===========================================================================
# Раздел: Задачи
# ===========================================================================

if page == PAGE_TASKS:
    if not tasks:
        st.warning("Задачи не найдены. Запустите: `uv run python src/mhgb/generate_tasks.py --type all --n 25`")
        st.stop()

    # --- Статистика + Матрица ---
    type_counts = Counter(t["type"] for t in tasks)
    mode_counts = Counter(t["mode"] for t in tasks)

    # Фабул на ячейку (считаем по closed-режиму, чтобы не дублировать)
    cell_counts: dict[tuple[str, str], int] = {}
    for t in tasks:
        if t.get("mode") == "closed":
            key = (t.get("type", ""), t.get("hop_group", ""))
            cell_counts[key] = cell_counts.get(key, 0) + 1

    _BRANCH_EN = {
        "налоговое": "Tax law", "гражданское": "Civil law",
        "административное": "Administrative law", "уголовное": "Criminal law",
        "градостроительное": "Urban planning law", "земельное": "Land law",
        "трудовое": "Labor law", "жилищное": "Housing law",
        "семейное": "Family law", "конституционное": "Constitutional law",
    }
    _closed_tasks = [t for t in tasks if t.get("mode") == "closed"]
    _branch_counts = Counter(t.get("branch_of_law", "unknown") for t in _closed_tasks)
    _total_fab = len(_closed_tasks)

    col_stats, col_matrix, col_branch = st.columns([2, 3, 2])

    with col_stats:
        r1a, r1b, r1c = st.columns(3)
        r1a.metric("Всего задач",  len(tasks))
        r1b.metric("Closed-book",  mode_counts.get("closed", 0))
        r1c.metric("Open-book",    mode_counts.get("open", 0))
        r2a, r2b, r2c = st.columns(3)
        r2a.metric("Conflict Res.", type_counts.get("conflict_resolution", 0))
        r2b.metric("Issue Spotting", type_counts.get("issue_spotting", 0))
        r2c.metric("Rule / Temporal", type_counts.get("rule_selection", 0) + type_counts.get("temporal_validity", 0))

    with col_matrix:
        st.markdown("**Матрица задач** — отметь ячейки для фильтрации по типу и глубине")
        _matrix_df = pd.DataFrame([
            {
                "Тип задачи": f"{TYPE_RU.get(tt, tt)}  ({cell_counts.get((tt,'shallow'),0)}/{cell_counts.get((tt,'medium'),0)}/{cell_counts.get((tt,'deep'),0)})",
                "shallow": False,
                "medium":  False,
                "deep":    False,
            }
            for tt in TASK_TYPES
        ])
        _edited = st.data_editor(
            _matrix_df,
            column_config={
                "Тип задачи": st.column_config.TextColumn(disabled=True, width="medium"),
                "shallow":    st.column_config.CheckboxColumn("🟢 shallow", width="medium"),
                "medium":     st.column_config.CheckboxColumn("🟠 medium",  width="medium"),
                "deep":       st.column_config.CheckboxColumn("🔴 deep",    width="medium"),
            },
            hide_index=True,
            use_container_width=False,
            key="matrix_editor",
        )
        matrix_sel: set[tuple[str, str]] = set()
        for _i, tt in enumerate(TASK_TYPES):
            for _hg in ["shallow", "medium", "deep"]:
                if _edited.iloc[_i][_hg]:
                    matrix_sel.add((tt, _hg))

    with col_branch:
        st.markdown("**Branch of Law** — 300 unique fabulas")
        _branch_rows = sorted(
            [(_BRANCH_EN.get(br, br), n) for br, n in _branch_counts.items()
             if br not in ("", "unknown")],
            key=lambda x: -x[1],
        )
        _pie_fig = go.Figure(go.Pie(
            labels=[r[0] for r in _branch_rows],
            values=[r[1] for r in _branch_rows],
            hole=0.35,
            textinfo="percent",
            # Легенда крупнее в 1.8 раза (10 → 18), подписи на секторах — 13
            # (крупнее не помещались в мелкие доли); круг на треть меньше (370 → 247).
            textfont=dict(size=13),
            hovertemplate="%{label}<br>%{value} fabulas (%{percent})<extra></extra>",
        ))
        _pie_fig.update_layout(
            margin=dict(t=0, b=0, l=0, r=0),
            height=247,
            showlegend=True,
            legend=dict(orientation="v", font=dict(size=18), x=1.0, y=0.5),
        )
        st.plotly_chart(_pie_fig, use_container_width=True)

    st.markdown("---")

    # --- Фильтры (дополнительное сужение) ---
    fc1, fc2, fc3, fc4 = st.columns([2, 2, 2, 2])
    with fc1:
        type_filter = st.multiselect(
            "Тип задачи",
            ["conflict_resolution", "issue_spotting", "rule_selection", "temporal_validity"],
            default=[],
            format_func=lambda x: TYPE_RU.get(x, x),
        )
    with fc2:
        mode_filter = st.multiselect("Режим", ["closed", "open"], default=[])
    with fc3:
        hop_filter = st.multiselect("Глубина (сложность)", ["shallow", "medium", "deep"], default=[])
    with fc4:
        scope_filter = st.multiselect("Охват", ["одна норма", "внутри НПА", "межкодексная"], default=[])

    _fbr, _fsr = st.columns([3, 9])
    with _fbr:
        _all_branches = sorted({
            t.get("branch_of_law", "") for t in tasks
            if t.get("branch_of_law") and t.get("branch_of_law") not in ("", "unknown")
        })
        branch_filter = st.multiselect("Отрасль права", _all_branches, default=[])
    with _fsr:
        search_q = st.text_input("Поиск в фабуле / вопросе", placeholder="введи ключевое слово...")

    # Применяем фильтры
    filtered = tasks
    if matrix_sel:
        filtered = [t for t in filtered if (t.get("type", ""), t.get("hop_group", "")) in matrix_sel]
    if type_filter:
        filtered = [t for t in filtered if t.get("type") in type_filter]
    if mode_filter:
        filtered = [t for t in filtered if t.get("mode") in mode_filter]
    if hop_filter:
        filtered = [t for t in filtered if t.get("hop_group") in hop_filter]
    if scope_filter:
        filtered = [t for t in filtered if _task_scope(t) in scope_filter]
    if branch_filter:
        filtered = [t for t in filtered if t.get("branch_of_law") in branch_filter]
    if search_q:
        sq = search_q.lower()
        filtered = [
            t for t in filtered
            if sq in t.get("fabula", "").lower() or sq in t.get("question", "").lower()
        ]

    if not filtered:
        st.info("Нет задач, удовлетворяющих фильтрам.")
        st.stop()

    # --- Список + Детали ---
    col_list, col_detail = st.columns([5, 7], gap="large")

    with col_list:
        st.caption(f"Найдено: {len(filtered)} задач")

        # Строим таблицу для отображения
        rows = []
        for i, t in enumerate(filtered):
            rows.append({
                "i": i,
                "Тип": TYPE_RU.get(t.get("type",""), t.get("type","")),
                "Режим": t.get("mode",""),
                "Глубина (сложность)": t.get("hop_group",""),
                "Охват": _task_scope(t),
                "Нормы": ", ".join(t.get("norm_ids", [])),
                "Фабула": t.get("fabula","")[:90] + "…",
            })

        event = st.dataframe(
            [{k: v for k, v in r.items() if k != "i"} for r in rows],
            use_container_width=True,
            height=640,
            selection_mode="single-row",
            on_select="rerun",
            key="task_table",
        )

    with col_detail:
        selected_rows = event.selection.rows if hasattr(event, "selection") else []

        if not selected_rows or selected_rows[0] >= len(filtered):
            st.markdown(
                '<div style="text-align:center;color:#aaa;margin-top:60px;font-size:16px">'
                '← Выбери задачу в таблице слева</div>',
                unsafe_allow_html=True,
            )
        else:
            idx  = selected_rows[0]
            task = filtered[idx]
            render_task_detail(task, corpus)


# ===========================================================================
# Раздел: Аналитика
# ===========================================================================

if page == PAGE_ANALYTICS:
    if not _PLOTLY_OK:
        st.error("Установи plotly: `uv add plotly kaleido`")
    else:
        _all_exp_v = list_experiments_versioned()
        if not _all_exp_v:
            st.info(
                "Нет результатов экспериментов. "
                "Запусти `run_main_experiment.py` → `compute_gap_analysis.py`, "
                "чтобы появились файлы `reports/<exp>/gap_records.jsonl`."
            )
        else:
            # ── Пред-фильтры: ВЕРСИЯ экспериментов + итерация ────────────────
            _ver_col, _iter_col = st.columns([5, 4])
            with _ver_col:
                # доступные версии (только те, по которым есть эксперименты)
                _avail_versions = [
                    v for v in ("phase2", "mvp")
                    if any(e["version"] == v for e in _all_exp_v)
                ]
                _exp_version = st.radio(
                    "Версия экспериментов",
                    _avail_versions,
                    index=0,
                    format_func=lambda v: EXPERIMENT_VERSION_LABELS[v],
                    horizontal=True,
                    key="ana_exp_version",
                    help="MVP (май 2026) и Phase-2 (июнь-июль 2026) методологически "
                         "несравнимы — другой судья, контекст с метаданными рёбер, "
                         "рандомизация порядка чанков. На графиках не смешиваются.",
                )
            with _iter_col:
                _iter_filter = st.radio(
                    "Итерация эксперимента",
                    ["Полная", "Частичная", "Выбрать всё"],
                    index=0,
                    horizontal=True,
                    key="ana_iter",
                )

            # сначала версия, потом итерация
            _version_exps = [e["name"] for e in _all_exp_v if e["version"] == _exp_version]
            if _iter_filter == "Полная":
                _filtered_exps = sorted(e for e in _version_exps if "_full" in e)
            elif _iter_filter == "Частичная":
                _filtered_exps = sorted(e for e in _version_exps if "_full" not in e)
            else:
                _filtered_exps = sorted(_version_exps)

            # ── Фильтры ──────────────────────────────────────────────────────
            _exp_name = None
            _fc1, _fc2, _fc3, _fc4, _fc5 = st.columns([2, 2, 2, 2, 1])
            if not _filtered_exps:
                with _fc1:
                    st.info("Нет экспериментов для выбранной версии и итерации.")
            else:
                with _fc1:
                    # key включает версию+итерацию → сброс при смене любого
                    _exp_name = st.selectbox(
                        "Эксперимент", _filtered_exps,
                        key=f"ana_exp_{_exp_version}_{_iter_filter}",
                    )
            with _fc5:
                st.markdown('<div style="margin-top:22px"></div>', unsafe_allow_html=True)
                if st.button("🔄 Обновить", key="ana_refresh",
                             help="Пересканировать reports/ и reports/phase2/ — после запуска нового эксперимента"):
                    list_experiments.clear()
                    list_experiments_versioned.clear()
                    load_analytics_data.clear()
                    load_rta_index.clear()
                    st.rerun()

            if _exp_name is None:
                st.stop()

            _adata = load_analytics_data(_exp_name, _exp_version)

            # Обогащение RtA-данными (если results_rta.jsonl есть).
            # is_rta_c/o — КАНОН genuine (не сырое is_rta детектора): overlay ✕
            # рисуется только для настоящих отказов, без артефактов (cap-петли/
            # пустые/overflow), консистентно с метрикой compute_rta_rate.
            _rta_index = load_rta_index(_exp_name, _exp_version)  # type/topic
            _has_rta = bool(_rta_index)
            _genuine_idx: dict = {}
            if _has_rta:
                from mhgb.analysis.compute_rta_analysis import \
                    is_genuine_rta as _is_genuine
                from mhgb.analysis.compute_rta_analysis import \
                    load_max_tokens_map as _load_mt
                from mhgb.analysis.compute_rta_analysis import \
                    load_rta_data as _load_rta_data
                _rta_root_g = resolve_exp_dir(_exp_name, _exp_version).parent
                _recs_g = _load_rta_data(_exp_name, _rta_root_g)
                _mt_g = _load_mt().get(_recs_g[0].get("model_name")) if _recs_g else None
                for _rec in _recs_g:
                    _bg = _rec["task_id"].removesuffix("_closed").removesuffix("_open")
                    _sg = "c" if _rec["task_id"].endswith("_closed") else "o"
                    _genuine_idx.setdefault(_bg, {})[_sg] = _is_genuine(_rec, _mt_g)
            for _r in _adata:
                _ri = _rta_index.get(_r["task_id"], {})
                _g = _genuine_idx.get(_r["task_id"], {})
                _r["is_rta_c"]   = bool(_g.get("c", False))   # genuine
                _r["is_rta_o"]   = bool(_g.get("o", False))   # genuine
                _r["is_rta_any"] = _r["is_rta_c"] or _r["is_rta_o"]
                _r["rta_type_c"] = _ri.get("rta_type_c")
                _r["rta_type_o"] = _ri.get("rta_type_o")
                _r["rta_topic_c"] = _ri.get("rta_topic_c")
                _r["rta_topic_o"] = _ri.get("rta_topic_o")

            if not _adata:
                st.warning(f"Нет данных для эксперимента **{_exp_name}**.")
            else:
                _all_models = sorted({r["model_name"] for r in _adata})
                _all_types  = sorted({r["task_type"]  for r in _adata})
                _all_hops   = [h for h in ["shallow", "medium", "deep"]
                               if h in {r["hop_group"] for r in _adata}]

                with _fc2:
                    _sel_models = st.multiselect("Модель", _all_models,
                                                 default=_all_models, key="ana_models")
                with _fc3:
                    _sel_types  = st.multiselect("Тип задачи", _all_types,
                                                 default=_all_types,
                                                 format_func=lambda x: TYPE_RU.get(x, x),
                                                 key="ana_types")
                with _fc4:
                    _sel_hops   = st.multiselect("Глубина", _all_hops,
                                                 default=_all_hops, key="ana_hops")

                _filtered_a = [
                    r for r in _adata
                    if (not _sel_models or r["model_name"] in _sel_models)
                    and (not _sel_types  or r["task_type"]  in _sel_types)
                    and (not _sel_hops   or r["hop_group"]  in _sel_hops)
                ]

                # Чекбокс скрытия RtA (только если данные есть)
                _hide_rta = False
                if _has_rta:
                    _rta_chk_col, _ = st.columns([3, 7])
                    with _rta_chk_col:
                        _hide_rta = st.checkbox(
                            "🚫 Скрыть пары с RtA-отказами",
                            value=False, key="hide_rta",
                        )
                    if _hide_rta:
                        _filtered_a = [r for r in _filtered_a if not r.get("is_rta_any")]

                if not _filtered_a:
                    st.info("Нет данных для выбранных фильтров.")
                else:
                    # ── Сводка метрик ─────────────────────────────────────────
                    _n_a = len(_filtered_a)
                    _mean_c = sum(r["closed_final"] for r in _filtered_a) / _n_a
                    _mean_o = sum(r["open_final"]   for r in _filtered_a) / _n_a
                    _mean_g = sum(r["gap_final"]    for r in _filtered_a) / _n_a
                    _q_cnt  = Counter(r["quadrant"] for r in _filtered_a)
                    _n_rta  = sum(1 for r in _filtered_a if r.get("is_rta_any")) if _has_rta else 0
                    if _has_rta:
                        # RtA% через ЕДИНУЮ канон-функцию (та же, что в postprocess
                        # compute_rta_analysis) — настоящие отказы / содержательные
                        # ответы. Артефакты (пустой/cap-петля/overflow) исключены из
                        # числителя; петли остаются в знаменателе (failure mode).
                        # Не дублировать формулу здесь — звать модульную функцию.
                        from mhgb.analysis.compute_rta_analysis import (
                            compute_rta_rate, is_genuine_rta,
                            load_max_tokens_map, load_rta_data)
                        _rta_root = resolve_exp_dir(_exp_name, _exp_version).parent
                        _rta_recs = load_rta_data(_exp_name, _rta_root)
                        _maxtok = (load_max_tokens_map().get(_rta_recs[0].get("model_name"))
                                   if _rta_recs else None)
                        _rate = compute_rta_rate(_rta_recs, _maxtok)
                        _n_rta_total = _rate["n_rta"]
                        _n_questions = _rate["n_answered"]
                        _pct_rta     = _rate["rta_rate"] * 100
                        # пары, где ОБА режима — настоящий отказ (для «Пар (оба)»)
                        _by_base: dict = {}
                        for _r in _rta_recs:
                            _b = _r["task_id"].removesuffix("_closed").removesuffix("_open")
                            _sfx = "c" if _r["task_id"].endswith("_closed") else "o"
                            _by_base.setdefault(_b, {})[_sfx] = is_genuine_rta(_r, _maxtok)
                        _n_rta_both = sum(1 for _v in _by_base.values()
                                          if _v.get("c") and _v.get("o"))
                    else:
                        _n_rta_total = _n_rta_both = 0
                        _pct_rta = 0.0

                    # ── Двухколоночный макет:
                    #    ЛЕВО = метрики + scatter | ПРАВО = pie + детали ──────
                    _col_chart, _col_detail = st.columns([6, 5], gap="large")

                    with _col_chart:
                        # Пять метрик над графиком
                        _m1, _m2, _m3, _m4, _m5 = st.columns(5, gap="small")
                        _m1.metric("Mean Open",        f"{_mean_o:.3f}")
                        _m2.metric("Mean Closed",      f"{_mean_c:.3f}")
                        _m3.metric("Mean GAP",         f"{_mean_g:+.3f}")
                        _n_scored = count_successful_records(_exp_name, _exp_version)
                        _m4.metric("Задач / Пар (closed/open)", f"{_n_scored} / {_n_a}")
                        if _has_rta:
                            _pct_color = "#CB6366" if _pct_rta > 0 else "inherit"
                            with _m5:
                                st.markdown(
                                    "<div style='font-size:0.75rem;color:#5a5a5a;"
                                    "line-height:1.35;margin-bottom:4px'>"
                                    "RtA-отказов<br>(всего&nbsp;/&nbsp;пар&nbsp;/&nbsp;%)"
                                    "</div>"
                                    f"<div style='font-size:1.5rem;font-weight:600;"
                                    f"white-space:nowrap'>"
                                    f"{_n_rta_total}&thinsp;/&thinsp;{_n_rta_both}&thinsp;/"
                                    f"&thinsp;<span style='color:{_pct_color}'>"
                                    f"{_pct_rta:.1f}%</span>"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )

                        # Scatter квадрантов под метриками
                        _rta_ids = (
                            {r["task_id"] for r in _filtered_a if r.get("is_rta_any")}
                            if _has_rta and not _hide_rta else None
                        )
                        _fig = _build_quadrant_figure(_filtered_a, rta_task_ids=_rta_ids, n_rta_total=_n_rta_total if _has_rta else None)
                        _chart_event = st.plotly_chart(
                            _fig,
                            use_container_width=True,
                            key="quadrant_chart",
                            on_select="rerun",
                            selection_mode="points",
                        )

                    with _col_detail:
                        # Пай-плот квадрантов — сверху правой колонки
                        _q_pie_order = sorted(_QUAD_COLORS, key=lambda q: _q_cnt.get(q, 0), reverse=True)
                        _pie_fig = go.Figure(go.Pie(
                            labels=[
                                f"{_QUAD_RU.get(q, q)}: {_q_cnt.get(q, 0)} ({100*_q_cnt.get(q, 0)//_n_a}%)"
                                for q in _q_pie_order
                            ],
                            values=[_q_cnt.get(q, 0) for q in _q_pie_order],
                            marker_colors=[_QUAD_COLORS[q] for q in _q_pie_order],
                            textinfo="none",
                            hovertemplate="%{label}<extra></extra>",
                            sort=False,
                            domain=dict(x=[0, 0.40]),
                        ))
                        _pie_fig.update_layout(
                            showlegend=True,
                            legend=dict(
                                orientation="v",
                                x=0.41,
                                y=0.5,
                                xanchor="left",
                                yanchor="middle",
                                font=dict(size=13),
                            ),
                            margin=dict(l=0, r=0, t=0, b=0),
                            height=125,
                            width=355,
                        )
                        st.plotly_chart(_pie_fig, use_container_width=False, key="summary_pie")

                        # Извлекаем выбранные точки
                        _pts: list = []
                        if _chart_event:
                            _sel = (
                                _chart_event.get("selection", {})
                                if isinstance(_chart_event, dict)
                                else getattr(_chart_event, "selection", {})
                            )
                            if isinstance(_sel, dict):
                                _pts = _sel.get("points", []) or []
                            else:
                                _pts = list(getattr(_sel, "points", []) or [])

                        if not _pts:
                            _ea = load_error_analysis(_exp_name, _exp_version)
                            if _ea:
                                if len(_sel_models) == 1:
                                    _model_lbl = _sel_models[0]
                                elif _sel_models and len(_sel_models) < len(_all_models):
                                    _model_lbl = ", ".join(_sel_models)
                                else:
                                    _model_lbl = ""
                                _render_error_analysis_panel(_ea, model_label=_model_lbl)
                            else:
                                st.markdown(
                                    '<div style="text-align:center;color:#aaa;'
                                    'margin-top:80px;font-size:15px">'
                                    '← Нажми на точку для деталей</div>',
                                    unsafe_allow_html=True,
                                )
                        else:
                            _pt = _pts[0]
                            _cd = (
                                _pt.get("customdata", [])
                                if isinstance(_pt, dict)
                                else list(getattr(_pt, "customdata", []))
                            )
                            if len(_cd) >= 22:
                                (
                                    _tid, _mname, _ttype, _hgroup,
                                    _fabula, _question, _answer,
                                    _resp_c, _resp_o,
                                    _nc_c, _sc_c, _ac_c, _fc_c,
                                    _nc_o, _sc_o, _ac_o, _fc_o,
                                    _gf, _gnc, _gsc, _gac,
                                    _quad,
                                ) = _cd[:22]
                                _norm_ids_str = _cd[22] if len(_cd) > 22 else ""

                                _HOP_COL = {
                                    "shallow": ("rgba(117,93,154,0.30)", "#4a2f8f"),
                                    "medium":  ("rgba(117,93,154,0.60)", "#fff"),
                                    "deep":    ("rgba(117,93,154,0.90)", "#fff"),
                                }
                                _badges  = ""
                                _badges += _badge(TYPE_RU.get(_ttype, _ttype), "#FFA343")
                                _hbg, _hfg = _HOP_COL.get(_hgroup, ("rgba(117,93,154,0.60)", "#fff"))
                                _badges += _badge(str(_hgroup).capitalize(), _hbg, _hfg)
                                _badges += _badge(_QUAD_RU.get(_quad, str(_quad).capitalize()),
                                                  _QUAD_COLORS.get(_quad, "#888"))
                                if _norm_ids_str:
                                    _nids_text = ", ".join(
                                        n.replace("_", " ") if "_" in n else n
                                        for n in str(_norm_ids_str).split("|")
                                        if n
                                    )
                                    _badges += (
                                        f'<span style="background:#fff;color:#555;'
                                        f'border:1px solid #bbb;padding:3px 10px;'
                                        f'border-radius:12px;font-size:12px;font-weight:500;'
                                        f'margin-left:4px">{_nids_text}</span>'
                                    )
                                st.markdown(_badges, unsafe_allow_html=True)
                                st.caption(f"Модель: **{_mname}** · `{str(_tid)[:8]}…`")

                                st.markdown("**Фабула**")
                                st.markdown(
                                    f'<div style="background:#f8f9fa;border-left:4px solid #3498db;'
                                    f'padding:10px 14px;border-radius:4px;font-size:13px;line-height:1.6">'
                                    f'{_fabula}</div>',
                                    unsafe_allow_html=True,
                                )
                                st.markdown("")
                                st.markdown("**Вопрос**")
                                st.markdown(
                                    f'<div style="background:#f8f9fa;border-left:4px solid #2ecc71;'
                                    f'padding:10px 14px;border-radius:4px;font-size:13px;line-height:1.6">'
                                    f'{_question}</div>',
                                    unsafe_allow_html=True,
                                )

                                # Таблица метрик
                                st.markdown("")
                                st.markdown("**Метрики**")
                                st.markdown(
                                    '<table style="width:100%;font-size:12px;border-collapse:collapse">'
                                    '<tr style="background:#f0f0f0"><th style="padding:4px 8px;text-align:left">Метрика</th>'
                                    '<th style="padding:4px 8px">Closed</th>'
                                    '<th style="padding:4px 8px">Open</th>'
                                    '<th style="padding:4px 8px">GAP</th></tr>'
                                    f'<tr><td style="padding:3px 8px">Norm Coverage</td>'
                                    f'<td style="text-align:center">{float(_nc_c):.3f}</td>'
                                    f'<td style="text-align:center">{float(_nc_o):.3f}</td>'
                                    f'<td style="text-align:center">{float(_gnc):+.3f}</td></tr>'
                                    f'<tr style="background:#f9f9f9"><td style="padding:3px 8px">Step Correctness</td>'
                                    f'<td style="text-align:center">{float(_sc_c):.3f}</td>'
                                    f'<td style="text-align:center">{float(_sc_o):.3f}</td>'
                                    f'<td style="text-align:center">{float(_gsc):+.3f}</td></tr>'
                                    f'<tr><td style="padding:3px 8px">Answer Correctness</td>'
                                    f'<td style="text-align:center">{float(_ac_c):.3f}</td>'
                                    f'<td style="text-align:center">{float(_ac_o):.3f}</td>'
                                    f'<td style="text-align:center">{float(_gac):+.3f}</td></tr>'
                                    f'<tr style="background:#eaf4fb;font-weight:600">'
                                    f'<td style="padding:3px 8px">Final Score</td>'
                                    f'<td style="text-align:center">{float(_fc_c):.3f}</td>'
                                    f'<td style="text-align:center">{float(_fc_o):.3f}</td>'
                                    f'<td style="text-align:center">{float(_gf):+.3f}</td></tr>'
                                    '</table>',
                                    unsafe_allow_html=True,
                                )

                                # Ответы модели
                                st.markdown("")
                                if _resp_c:
                                    with st.expander("📝 Ответ модели (Closed-book)"):
                                        st.markdown(
                                            f'<div style="background:#f9f9f9;border:1px solid #ddd;'
                                            f'border-radius:4px;padding:10px;font-size:12px;'
                                            f'line-height:1.6;white-space:pre-wrap;'
                                            f'max-height:260px;overflow-y:auto">{_resp_c}</div>',
                                            unsafe_allow_html=True,
                                        )
                                if _resp_o:
                                    with st.expander("📝 Ответ модели (Open-book)"):
                                        st.markdown(
                                            f'<div style="background:#f9f9f9;border:1px solid #ddd;'
                                            f'border-radius:4px;padding:10px;font-size:12px;'
                                            f'line-height:1.6;white-space:pre-wrap;'
                                            f'max-height:260px;overflow-y:auto">{_resp_o}</div>',
                                            unsafe_allow_html=True,
                                        )

                                # Эталонный ответ
                                if _answer:
                                    st.markdown("")
                                    st.markdown("**Эталонный ответ**")
                                    st.markdown(
                                        f'<div style="background:#f8f9fa;border-left:4px solid #e67e22;'
                                        f'padding:10px 14px;border-radius:4px;font-size:13px;line-height:1.6">'
                                        f'{_answer}</div>',
                                        unsafe_allow_html=True,
                                    )

                                # Gold Chain
                                _rec = next(
                                    (r for r in _adata
                                     if r["task_id"] == _tid and r["model_name"] == _mname),
                                    {},
                                )
                                _gold_chain = _rec.get("gold_chain", [])
                                if _gold_chain:
                                    st.markdown("")
                                    with st.expander("🔗 Gold Chain (цепочка рассуждений)", expanded=True):
                                        for _step in _gold_chain:
                                            _sn  = _step.get("step", "?")
                                            _nid = _step.get("norm_id")
                                            _rsn = _step.get("reasoning") or _step.get("conclusion", "")
                                            _hdr = f"Шаг {_sn} — `{_nid}`" if _nid else f"Шаг {_sn} — **Вывод**"
                                            st.markdown(
                                                f'<div style="background:#eaf4fb;border:1px solid #aed6f1;'
                                                f'border-radius:6px;padding:10px 14px;margin-bottom:8px;font-size:13px">'
                                                f'<b>{_hdr}</b><br>{_rsn}</div>',
                                                unsafe_allow_html=True,
                                            )

                    # ── Дополнительные графики ────────────────────────────────
                    st.markdown("---")
                    _gc1, _gc2, _gc3 = st.columns(3)
                    with _gc1:
                        st.plotly_chart(
                            _build_gap_by_type_figure(_filtered_a),
                            use_container_width=True,
                            key="chart_gap_type",
                        )
                    with _gc2:
                        st.plotly_chart(
                            _build_quadrant_dist_figure(_filtered_a),
                            use_container_width=True,
                            key="chart_quad_dist",
                        )
                    with _gc3:
                        st.plotly_chart(
                            _build_gap_by_hop_figure(_filtered_a),
                            use_container_width=True,
                            key="chart_gap_hop",
                        )

                    # ── RtA-анализ (если данные загружены) ───────────────────
                    if _has_rta and _n_rta > 0:
                        st.markdown("---")
                        st.markdown("#### 🚫 RtA-анализ")
                        _ra1, _ra2 = st.columns([1, 2])
                        with _ra1:
                            # Сводка по типам и темам
                            _rta_recs_only = [r for r in _filtered_a if r.get("is_rta_any")]
                            _type_cnt = Counter(
                                r.get("rta_type_c") or r.get("rta_type_o")
                                for r in _rta_recs_only
                            )
                            _topic_cnt = Counter(
                                r.get("rta_topic_c") or r.get("rta_topic_o")
                                for r in _rta_recs_only
                            )
                            st.markdown(f"**Всего пар с отказом:** {_n_rta} из {_n_a} ({100*_n_rta//_n_a}%)")
                            st.markdown("**По типу отказа:**")
                            for k, v in _type_cnt.most_common():
                                st.markdown(f"- `{k or 'null'}`: {v}")
                            st.markdown("**По теме отказа:**")
                            for k, v in _topic_cnt.most_common():
                                st.markdown(f"- `{k or 'null'}`: {v}")
                        with _ra2:
                            st.plotly_chart(
                                _build_rta_by_topic_figure(_filtered_a),
                                use_container_width=True,
                                key="chart_rta_topic",
                            )


# ===========================================================================
# Раздел: Лидерборд
# ===========================================================================
if page == PAGE_LEADERBOARD:
    _lb_head_col, _lb_refresh_col = st.columns([6, 1])
    with _lb_head_col:
        st.subheader("🏆 Лидерборд моделей")
        st.caption("Только полные прогоны (_full). Сортировка по Mean Open.")
    with _lb_refresh_col:
        st.markdown('<div style="margin-top:22px"></div>', unsafe_allow_html=True)
        if st.button("🔄 Обновить", key="lb_refresh",
                     help="Пересканировать reports/ — нужно после добавления нового эксперимента"):
            load_leaderboard_raw.clear()
            load_model_metadata.clear()
            st.rerun()

    _lb_version = st.radio(
        "Версия экспериментов", ["phase2", "mvp"], index=0,
        format_func=lambda v: EXPERIMENT_VERSION_LABELS[v], horizontal=True,
        key="lb_exp_version",
        help="MVP (май) и Phase-2 (июль) несравнимы — другой судья/контекст/"
             "рандомизация. Одноимённые модели (o3/yandex/gigachat-max) разведены по версии.",
    )

    _lb_raw = load_leaderboard_raw(_lb_version)
    _lb_meta = load_model_metadata()

    if not _lb_raw:
        st.info(f"Нет данных лидерборда для версии «{EXPERIMENT_VERSION_LABELS[_lb_version]}».")
    else:
        # ── Фильтры ──────────────────────────────────────────────────────────
        _lbf1, _lbf2, _lbf3 = st.columns(3)

        _access_options = ["all", "Открытый", "Проприетарный"]
        with _lbf1:
            _lb_access = st.selectbox("Доступ", _access_options,
                                       format_func=lambda x: "Все" if x == "all" else x,
                                       key="lb_access")

        _all_types = sorted({r.get("task_type","unknown") for r in _lb_raw} - {"unknown",""})
        with _lbf2:
            _lb_type = st.selectbox("Тип задачи", ["all"] + _all_types,
                                     format_func=lambda x: "Все" if x == "all" else TYPE_RU.get(x, x),
                                     key="lb_task_type")

        _all_branches = sorted({r.get("branch_of_law","") for r in _lb_raw} - {"","unknown"})
        with _lbf3:
            _lb_branch = st.selectbox("Отрасль права", ["all"] + _all_branches,
                                       format_func=lambda x: "Все" if x == "all" else x,
                                       key="lb_branch")

        # ── Применяем фильтры ─────────────────────────────────────────────
        _lb_filtered = _lb_raw
        if _lb_access != "all":
            _lb_filtered = [r for r in _lb_filtered
                            if ACCESS_MAP.get(_lb_meta.get(r.get("model_name",""), {}).get("slice",""), "—") == _lb_access]
        if _lb_type != "all":
            _lb_filtered = [r for r in _lb_filtered if r.get("task_type") == _lb_type]
        if _lb_branch != "all":
            _lb_filtered = [r for r in _lb_filtered if r.get("branch_of_law") == _lb_branch]

        _lb_df = _aggregate_leaderboard(_lb_filtered, _lb_meta)

        if _lb_df.empty:
            st.warning("Нет данных для выбранных фильтров.")
        else:
            # ── HTML-таблица с групповыми заголовками ─────────────────────
            # st.components.v1.html выполняет JS и поддерживает :hover в iframe
            _tbl_height = 80 + len(_lb_df) * 36 + 20
            st.components.v1.html(
                _render_leaderboard_html(_lb_df),
                height=_tbl_height,
                scrolling=False,
            )

            # ── Radar chart ───────────────────────────────────────────────
            st.markdown("---")
            st.markdown("**Radar-сравнение моделей**")
            _model_options = _lb_df["_model_name"].tolist()
            _model_labels  = _lb_df["Модель"].tolist()
            _radar_key = f"lb_radar_select_{_lb_access}_{_lb_type}_{_lb_branch}"
            _selected_idx = st.multiselect(
                "Выберите модели для сравнения",
                options=range(len(_model_options)),
                default=list(range(min(len(_model_options), 4))),
                format_func=lambda i: _model_labels[i],
                key=_radar_key,
            )
            _selected_names = [_model_options[i] for i in _selected_idx]
            if _selected_names and _PLOTLY_OK:
                st.plotly_chart(
                    _build_radar_figure(_lb_df, _selected_names),
                    use_container_width=True,
                    key=f"lb_radar_{_lb_access}_{_lb_type}_{_lb_branch}",
                )
