"""
Генерация графиков для анализа результатов (Plotly → PNG).

Читает CSV-файлы из reports/{experiment_name}/ (созданные
aggregate_results.py и compute_gap_analysis.py).
Сохраняет PNG через kaleido (scale=2 для Retina).

CLI:
  uv run python src/mhgb/analysis/generate_plots.py \\
    --experiment-name main_v1 \\
    --output-dir reports/main_v1/plots
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from mhgb.logging_setup import logger

_QUADRANT_COLORS = {
    "knows":        "#2ecc71",
    "reasons":      "#3498db",
    "hallucinates": "#e74c3c",
    "incompetent":  "#95a5a6",
}


def _load_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(path), scale=2)
    logger.info(f"График сохранён: {path}")


# ---------------------------------------------------------------------------
# Heatmap: модели × типы задач → final_score_mean
# ---------------------------------------------------------------------------

def plot_final_score_heatmap(agg_path: Path, output_path: Path, mode: str = "closed") -> None:
    import plotly.graph_objects as go

    rows = [r for r in _load_csv(agg_path) if r.get("mode") == mode]
    if not rows:
        logger.warning(f"Нет данных для mode={mode} в {agg_path}")
        return

    models     = sorted({r["model_name"] for r in rows})
    task_types = sorted({r["task_type"]  for r in rows})

    z      = []
    text   = []
    for model in models:
        row_z, row_t = [], []
        for tt in task_types:
            match = next((r for r in rows if r["model_name"] == model and r["task_type"] == tt), None)
            val = float(match["final_score_mean"]) if match and match.get("final_score_mean") else None
            row_z.append(val)
            row_t.append(f"{val:.3f}" if val is not None else "")
        z.append(row_z)
        text.append(row_t)

    fig = go.Figure(go.Heatmap(
        z=z,
        x=task_types,
        y=models,
        text=text,
        texttemplate="%{text}",
        colorscale="RdYlGn",
        zmin=0, zmax=1,
        colorbar=dict(title="Final Score"),
    ))
    fig.update_layout(
        title=f"Final Score — модели × тип задачи ({mode})",
        xaxis_title="Тип задачи",
        yaxis_title="Модель",
        height=max(300, len(models) * 45 + 120),
        margin=dict(l=160, r=40, t=60, b=80),
    )
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Bar chart: GAP (final_score) по моделям
# ---------------------------------------------------------------------------

def plot_gap_by_model(gap_path: Path, output_path: Path) -> None:
    import plotly.graph_objects as go

    rows   = _load_csv(gap_path)
    models = [r["model_name"] for r in rows]
    gaps   = [float(r.get("final_score_gap_mean", 0) or 0) for r in rows]
    stds   = [float(r.get("final_score_gap_std",  0) or 0) for r in rows]
    colors = [_QUADRANT_COLORS["reasons"] if g >= 0 else _QUADRANT_COLORS["hallucinates"]
              for g in gaps]

    fig = go.Figure(go.Bar(
        x=models,
        y=gaps,
        error_y=dict(type="data", array=stds, visible=True),
        marker_color=colors,
        text=[f"{g:+.3f}" for g in gaps],
        textposition="outside",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=1)
    fig.update_layout(
        title="GAP-метрика по моделям (open − closed, Final Score)",
        xaxis_title="Модель",
        yaxis_title="GAP",
        xaxis_tickangle=-35,
        height=500,
        margin=dict(b=120),
    )
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Stacked bar: квадранты GAP по моделям
# ---------------------------------------------------------------------------

def plot_quadrant_distribution(gap_path: Path, output_path: Path) -> None:
    import plotly.graph_objects as go

    rows      = _load_csv(gap_path)
    models    = [r["model_name"] for r in rows]
    quadrants = ("knows", "reasons", "hallucinates", "incompetent")

    traces = []
    for q in quadrants:
        totals = [sum(int(r.get(f"q_{qq}", 0) or 0) for qq in quadrants) or 1 for r in rows]
        values = [int(r.get(f"q_{q}", 0) or 0) / t for r, t in zip(rows, totals)]
        traces.append(go.Bar(
            name=q,
            x=models,
            y=values,
            marker_color=_QUADRANT_COLORS[q],
            text=[f"{v:.0%}" for v in values],
            textposition="inside",
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        barmode="stack",
        title="Распределение квадрантов GAP по моделям",
        xaxis_title="Модель",
        yaxis_title="Доля задач",
        yaxis_tickformat=".0%",
        xaxis_tickangle=-35,
        legend_title="Квадрант",
        height=500,
        margin=dict(b=120),
    )
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Line plot: GAP по hop_group
# ---------------------------------------------------------------------------

def plot_gap_by_hop_group(gap_hop_path: Path, output_path: Path) -> None:
    import plotly.graph_objects as go

    hop_order = {"shallow": 0, "medium": 1, "deep": 2}
    rows = sorted(_load_csv(gap_hop_path),
                  key=lambda r: hop_order.get(r.get("hop_group", ""), 99))

    x     = [r.get("hop_group", "") for r in rows]
    y     = [float(r.get("final_score_gap_mean", 0) or 0) for r in rows]
    y_std = [float(r.get("final_score_gap_std",  0) or 0) for r in rows]

    y_upper = [yi + si for yi, si in zip(y, y_std)]
    y_lower = [yi - si for yi, si in zip(y, y_std)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x + x[::-1],
        y=y_upper + y_lower[::-1],
        fill="toself",
        fillcolor="rgba(52, 152, 219, 0.15)",
        line_color="rgba(255,255,255,0)",
        showlegend=False,
        name="±std",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="lines+markers+text",
        marker=dict(size=10),
        line=dict(color="#3498db", width=2),
        name="GAP mean",
        text=[f"{yi:+.3f}" for yi in y],
        textposition="top center",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="grey", line_width=1)
    fig.update_layout(
        title="GAP по уровню сложности (hop_group)",
        xaxis_title="Hop group",
        yaxis_title="GAP (final_score)",
        height=420,
    )
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Quadrant matrix: scatter closed vs open с аннотациями метрик по квадрантам
# ---------------------------------------------------------------------------

def plot_quadrant_matrix(
    gap_records_path: Path,
    results_path: Path,
    output_path: Path,
) -> None:
    import statistics
    from collections import defaultdict

    import plotly.graph_objects as go

    gap_records: list[dict] = _load_jsonl(gap_records_path)
    if not gap_records:
        logger.warning(f"Нет gap_records в {gap_records_path}")
        return

    results: dict[str, dict] = {}
    for r in _load_jsonl(results_path):
        results[r.get("task_id", "")] = r

    def fv(v: object) -> float:
        return round(float(v), 3) if v is not None else 0.0

    by_q: dict[str, list[dict]] = defaultdict(list)
    points: list[tuple] = []
    for g in gap_records:
        q    = g.get("quadrant", "incompetent")
        base = g["task_id"]
        c    = results.get(base + "_closed", {})
        o    = results.get(base + "_open", {})
        fc   = fv(c.get("final_score"))
        fo   = fv(o.get("final_score"))
        points.append((fc, fo, q, base[:8]))
        by_q[q].append({
            "norm_cov_c": fv(c.get("norm_coverage_list")),
            "step_c":     fv(c.get("step_correctness")),
            "ans_c":      fv(c.get("answer_correctness")),
            "final_c":    fc,
            "norm_cov_o": fv(o.get("norm_coverage_list")),
            "step_o":     fv(o.get("step_correctness")),
            "ans_o":      fv(o.get("answer_correctness")),
            "final_o":    fo,
            "gap":        fv(g.get("final_score_gap")),
        })

    total = sum(len(v) for v in by_q.values())

    def m(vals: list[float]) -> float:
        return round(statistics.mean(vals), 3) if vals else 0.0

    QUADS = {
        "knows":        {"label": "Знает",        "color": "rgba(52,152,219,0.12)",  "tx": 0.75, "ty": 0.75},
        "reasons":      {"label": "Рассуждает",   "color": "rgba(46,204,113,0.12)",  "tx": 0.25, "ty": 0.75},
        "hallucinates": {"label": "Галлюцинирует","color": "rgba(231,76,60,0.12)",   "tx": 0.75, "ty": 0.25},
        "incompetent":  {"label": "Некомпетентна","color": "rgba(149,165,166,0.12)", "tx": 0.25, "ty": 0.25},
    }
    COLORS = {
        "knows": "#3498db", "reasons": "#2ecc71",
        "hallucinates": "#e74c3c", "incompetent": "#95a5a6",
    }

    fig = go.Figure()

    for qkey, qcfg in QUADS.items():
        x0, x1 = (0.5, 1.0) if qcfg["tx"] > 0.5 else (0.0, 0.5)
        y0, y1 = (0.5, 1.0) if qcfg["ty"] > 0.5 else (0.0, 0.5)
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                      fillcolor=qcfg["color"], line_width=0, layer="below")

    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                  line=dict(color="rgba(0,0,0,0.25)", width=1.5, dash="dot"))
    fig.add_shape(type="line", x0=0.5, y0=0, x1=0.5, y1=1,
                  line=dict(color="rgba(0,0,0,0.3)", width=1.5))
    fig.add_shape(type="line", x0=0, y0=0.5, x1=1, y1=0.5,
                  line=dict(color="rgba(0,0,0,0.3)", width=1.5))

    annotations = []
    for qkey, qcfg in QUADS.items():
        items = by_q.get(qkey, [])
        n   = len(items)
        pct = round(100 * n / total) if total else 0
        tx, ty = qcfg["tx"], qcfg["ty"]

        if n == 0:
            txt = f"<b>{qcfg['label']}</b><br>0 задач (0%)<br>—"
        else:
            fc_m = m([i["final_c"] for i in items])
            fo_m = m([i["final_o"] for i in items])
            txt = (
                f"<b>{qcfg['label']}</b><br>"
                f"<b>{n} задач ({pct}%)</b><br><br>"
                f"<b>CLOSED</b><br>"
                f"norm_cov: {m([i['norm_cov_c'] for i in items])}<br>"
                f"step_corr: {m([i['step_c'] for i in items])}<br>"
                f"ans_corr: {m([i['ans_c'] for i in items])}<br>"
                f"<b>final: {fc_m}</b><br><br>"
                f"<b>OPEN</b><br>"
                f"norm_cov: {m([i['norm_cov_o'] for i in items])}<br>"
                f"step_corr: {m([i['step_o'] for i in items])}<br>"
                f"ans_corr: {m([i['ans_o'] for i in items])}<br>"
                f"<b>final: {fo_m}</b><br><br>"
                f"<b>GAP: {m([i['gap'] for i in items]):+}</b>"
            )

        annotations.append(go.layout.Annotation(
            x=tx, y=ty, text=txt, showarrow=False, align="center",
            font=dict(size=12, family="Arial"),
            bgcolor="rgba(255,255,255,0.75)",
            bordercolor="rgba(0,0,0,0.15)", borderwidth=1, borderpad=6,
        ))

    annotations.append(go.layout.Annotation(
        x=1.03, y=1.03, text="GAP=0", showarrow=False,
        font=dict(size=10, color="rgba(0,0,0,0.4)"),
    ))

    for qkey, qcfg in QUADS.items():
        pts = [(fc, fo, tid) for fc, fo, q, tid in points if q == qkey]
        if not pts:
            continue
        fig.add_trace(go.Scatter(
            x=[p[0] for p in pts], y=[p[1] for p in pts],
            mode="markers+text",
            marker=dict(size=12, color=COLORS[qkey], line=dict(width=1.5, color="white")),
            text=[p[2] for p in pts],
            textposition="top center",
            textfont=dict(size=9),
            name=qcfg["label"],
        ))

    fig.update_layout(
        title=dict(
            text="Матрица квадрантов GAP (closed vs open Final Score)",
            font=dict(size=15),
        ),
        xaxis=dict(title="Final Score (closed-book)", range=[-0.05, 1.05], dtick=0.25,
                   gridcolor="rgba(0,0,0,0.08)"),
        yaxis=dict(title="Final Score (open-book)", range=[-0.05, 1.05], dtick=0.25,
                   gridcolor="rgba(0,0,0,0.08)"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=860, height=820,
        annotations=annotations,
        legend=dict(x=0.01, y=0.01, bgcolor="rgba(255,255,255,0.8)", borderwidth=1),
    )
    _save(fig, output_path)


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Генерация графиков (Plotly PNG)")
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--output-dir",      default=None)
    args = parser.parse_args()

    report_dir = Path(f"reports/{args.experiment_name}")
    output_dir = Path(args.output_dir or report_dir / "plots")

    p = report_dir / "agg_model_name_task_type_mode.csv"
    if p.exists():
        plot_final_score_heatmap(p, output_dir / "heatmap_closed.png",       mode="closed")
        plot_final_score_heatmap(p, output_dir / "heatmap_open.png",         mode="open_full_graph")

    p = report_dir / "gap_model_name.csv"
    if p.exists():
        plot_gap_by_model(p,           output_dir / "gap_by_model.png")
        plot_quadrant_distribution(p,  output_dir / "quadrant_dist.png")

    p = report_dir / "gap_hop_group.csv"
    if p.exists():
        plot_gap_by_hop_group(p,       output_dir / "gap_by_hop_group.png")

    gap_records_p = report_dir / "gap_records.jsonl"
    results_p     = report_dir / "results.jsonl"
    if gap_records_p.exists() and results_p.exists():
        plot_quadrant_matrix(
            gap_records_p, results_p,
            output_dir / "quadrant_matrix_detailed.png",
        )

    logger.info("Готово")


if __name__ == "__main__":
    main()
