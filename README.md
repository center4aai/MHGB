# MHGB — Multi-Hop Graph Bench for Legal Reasoning

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21611555.svg)](https://doi.org/10.5281/zenodo.21611555)

MHGB is a benchmark for evaluating multi-hop legal reasoning over Russian statutory law.
It represents statutory knowledge as a typed graph of 5,519 article nodes connected by
8,760 relations of five types across ten federal codes, and uses that structure to
generate 600 multi-hop reasoning tasks, each carrying a gold reasoning chain. Every task
is posed in two modes — **closed-book**, answered from parametric knowledge alone, and
**open-book**, with graph-derived statutory context — so that what a model *knows* can be
separated from how well it *reasons over norms placed before it*. Responses are scored at
three levels by two LLM judges held outside the leaderboard, and the **GAP** metric
measures the change between the two modes.

No prior benchmark targets legal reasoning for Russian, and none represents statutory
knowledge as a typed graph that both generates the tasks and supplies the evaluation
context.

---

## At a glance

| Component | Scale |
|---|---|
| Statutory codes | 10 |
| Articles in corpus | 5,570 |
| Graph nodes | 5,519 |
| Graph edges | 8,760 (7,332 structural, 1,428 doctrinal) |
| Cross-code edges | 115 |
| Task instances (full set) | 600 (300 fabulas × 2 modes) |
| Task instances (public subset) | 120 (60 fabulas × 2 modes) |
| Task types × reasoning depths | 4 × 3 |
| Models evaluated in the paper | 12 |

**Edge types.** `refers_to` (4,380) and `applies_to` (2,952) are *structural*;
`excludes` (715), `supplements` (381) and `prevails_over` (332) are *doctrinal*.

---

## Data access

MHGB is released in three tiers. The task set is split deliberately: publishing all 600
tasks in a crawlable repository would let them enter future pretraining corpora, which
would silently invalidate the closed-book mode the benchmark is built on. This is standard
practice for contamination-sensitive benchmarks, not a restriction on use.

| Tier | What | How |
|---|---|---|
| **1. Public** | Full code, full knowledge graph, full statutory corpus, a stratified **120-task subset** with gold chains, and **per-model scores over all 300 fabulas** | This repository |
| **2. Browsable** | All 600 tasks with their gold reasoning chains and context, inspectable one by one, but not downloadable in bulk | Interactive web interface — *link to be added* |
| **3. On request** | Full 600-task set as data files | Email **polukoshko.marina@gmail.com**, stating your intended use |

The public subset is a stratified sample of 10 task instances per cell of the 4×3 matrix,
drawn at the fabula level with a fixed seed so closed/open pairs stay intact and the GAP
metric remains computable. Anyone holding the full set (tier 2 or 3) can reproduce the
published subset exactly:

```bash
python scripts/select_public_subset.py \
    --tasks data/tasks_raw.jsonl \
    --out   data/tasks_public_120.jsonl
```

The entire pipeline — context construction, model inference, three-level scoring,
aggregation — runs end to end on the public subset.

---

## Repository layout

```
.
├── configs/
│   └── models.yaml              Model and judge definitions (endpoints from env vars)
├── data/
│   ├── corpus.jsonl             5,570 articles of 10 federal codes
│   ├── graph.json               Typed knowledge graph (NetworkX node-link format)
│   ├── graph_diagnostics_v2.json  Graph statistics
│   ├── tasks_public_120.jsonl   Public task subset with gold chains
│   └── tasks_public_120_meta.json
├── docs/
│   ├── methodology.md           Metrics, scoring, context construction, judging
│   └── datasheet.md             Datasheet for the graph and the task set
├── results/                     Per-model scores for all 12 models, 300 fabulas each
│   ├── <experiment>/            gap_records.jsonl + aggregated CSVs (no text, numbers only)
│   └── statistics/              Bootstrap CIs and paired tests
├── prompts/                     Judge, generation and edge-classification prompts
├── scripts/
│   ├── select_public_subset.py  Reproduce the public subset
│   ├── reproduce_paper_stats.py Recompute the paper's CIs and significance tests
│   ├── graph_diagnostics.py     Graph statistics
│   ├── run_cross_judge.py       Re-score with the secondary judge
│   ├── compute_crossjudge_kappa.py
│   ├── run_rta_detection.py     Refusal-to-answer detection
│   ├── compute_rta_kappa.py
│   ├── merge_crossjudge_into_results.py
│   ├── export_tasks_for_experts.py    Build expert annotation packages
│   ├── import_expert_validation.py    Fleiss κ, edge precision, Wilson CI
│   └── select_edge_validation_sample.py
├── src/mhgb/
│   ├── parse_docs.py            Statutes (.docx) → corpus.jsonl
│   ├── build_graph.py           Corpus → typed graph (pattern matching + LLM typing)
│   ├── generate_tasks.py        Graph → multi-hop tasks with gold chains
│   ├── eval/                    Scoring: norm coverage, step and answer correctness,
│   │                            final score, GAP, context builders, refusal detector
│   ├── experiments/             Main evaluation pipeline
│   ├── analysis/                Aggregation, GAP analysis, error taxonomy, tables, plots
│   ├── models/                  LLM clients (OpenAI-compatible, GigaChat, native Ollama)
│   ├── validation/              Task validation and expert-annotation utilities
│   ├── schemas/                 Pydantic task schema
│   └── storage/                 Optional MongoDB persistence
├── tests/unit/                  Unit test suite (664 tests)
└── graph_explorer.py            Streamlit web interface
```

---

## Installation

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/). Developed and tested on
CPython 3.14; `.python-version` pins that for `uv`.

```bash
git clone https://github.com/center4aai/MHGB.git
cd MHGB
uv sync

cp .env.example .env     # then fill in endpoints and API keys
```

Run the tests:

```bash
uv run pytest tests/unit -q
```

Only the models you actually evaluate need credentials. Self-hosted models run through
[Ollama](https://ollama.com/) or [llama.cpp](https://github.com/ggml-org/llama.cpp); no
external API is required to reproduce the benchmark with open-weight models.

---

## Reproducing an evaluation on the public subset

The evaluation needs two models: the **participant** being scored and the **judge**. In
the paper the primary judge is Llama 3.3 70B; both judges sit outside the leaderboard, so
no model ever scores its own answers.

```bash
# 1. Dry run — exercises the whole pipeline without calling any LLM
uv run python -m mhgb.experiments.run_main_experiment \
  --models-config configs/models.yaml \
  --tasks data/tasks_public_120.jsonl \
  --experiment-name demo \
  --models t-pro-it-2.1 \
  --judge  llama3.3-70b-judge \
  --match-task-mode \
  --dry-run --max-tasks 4

# 2. Real run over all 120 instances, each in its native mode
uv run python -m mhgb.experiments.run_main_experiment \
  --models-config configs/models.yaml \
  --tasks data/tasks_public_120.jsonl \
  --experiment-name demo \
  --models t-pro-it-2.1 \
  --judge  llama3.3-70b-judge \
  --match-task-mode
# → reports/demo/results.jsonl  (append-only and resumable: rerun to continue)
```

Then aggregate:

```bash
EXP=demo

# Metrics by slice → CSV
uv run python -m mhgb.analysis.aggregate_results \
  --experiment-name $EXP --results reports/$EXP/results.jsonl --output-dir reports/$EXP

# GAP: pair closed with open, classify quadrants
uv run python -m mhgb.analysis.compute_gap_analysis \
  --experiment-name $EXP --results reports/$EXP/results.jsonl --output-dir reports/$EXP

# Error taxonomy
uv run python -m mhgb.analysis.error_analysis --results reports/$EXP/results.jsonl

# Tables (Markdown + LaTeX) and figures
uv run python -m mhgb.analysis.generate_tables --experiment-name $EXP
uv run python -m mhgb.analysis.generate_plots  --experiment-name $EXP
```

Reliability checks reported in the paper:

```bash
# Re-score every response with the secondary judge, then measure agreement
uv run python scripts/run_cross_judge.py --experiment $EXP --judge qwen3-27b-server
uv run python scripts/compute_crossjudge_kappa.py --experiment $EXP

# Refusal-to-answer detection and its cross-judge agreement
uv run python scripts/run_rta_detection.py --experiment $EXP --judge llama3.3-70b-judge
uv run python -m mhgb.analysis.compute_rta_analysis --experiment $EXP --by mode
```

### Rebuilding the graph and the tasks from scratch

```bash
uv run python -m mhgb.parse_docs                 # .docx statutes → data/corpus.jsonl
uv run python -m mhgb.build_graph --no-llm       # pattern-matched candidate edges only
uv run python -m mhgb.build_graph                # + LLM relation typing
uv run python scripts/graph_diagnostics.py       # graph statistics
uv run python -m mhgb.generate_tasks --type all --n 25
```

`parse_docs` expects the source `.docx` files in `legal_docs/`. They are not redistributed
here; `data/corpus.jsonl` is the parsed result and is sufficient for everything else.

### Web interface

```bash
uv run streamlit run graph_explorer.py
```

Browse the graph, inspect tasks with their gold chains and context, and view the
leaderboard and quadrant matrices for any local evaluation runs.

### Reproducing the published numbers without running any model

The per-model scores behind every table in the paper are released in `results/` — one
record per fabula holding the closed-book score, the open-book score and the GAP, over
the full 300-fabula benchmark. They contain identifiers and numbers only, no task text
and no model responses.

```bash
uv run python scripts/reproduce_paper_stats.py
```

Seconds, no API keys, no GPU: recomputes the main results table, bootstrap 95% confidence
intervals, paired *t*-tests on the GAP, and quadrant distributions. Bootstrap runs with
2,000 resamples and seed 42, so the intervals match the published ones exactly. See
[`results/README.md`](results/README.md).

---

## Evaluation protocol

Responses follow a three-part format — applicable articles, reasoning chain, conclusion —
and are scored at three levels:

| Level | Metric | How |
|---|---|---|
| 1 | **Norm coverage** | Deterministic F1 between the norms the model cites and the gold norm set |
| 2 | **Step correctness** | A judge scores each gold-chain step on {0, 0.5, 1} |
| 3 | **Answer correctness** | A judge scores the conclusion on {0, 0.33, 0.67, 1} |

**Final Score** = ⅓·NC + ⅓·SC + ⅓·AC, with all three components always reported
separately. **GAP** = FS_open − FS_closed, computed per fabula. Each (model, fabula) pair
falls into one of four quadrants at threshold τ = 0.5: *Knows*, *Reasons*, *Hallucinates*,
*Incompetent*.

Full details — context construction, judge selection, bootstrap procedures, refusal
detection — are in [`docs/methodology.md`](docs/methodology.md).

---

## Licenses

MHGB has three components under three different terms.

| Component | License |
|---|---|
| **Code** — `src/`, `scripts/`, `tests/`, `prompts/`, `graph_explorer.py` | [Apache-2.0](LICENSE) |
| **Derived data** — knowledge graph, tasks, gold chains, corpus metadata | [CC BY-4.0](LICENSE-DATA) |
| **Statutory text** — `data/corpus.jsonl` article text | Public domain |

The text of Russian federal codes is not subject to copyright under Article 1259(6) of the
Civil Code of the Russian Federation, which excludes official documents of state bodies.
We claim no rights over those texts; the parsing, segmentation and metadata we added are
released under CC BY-4.0 with the rest of the derived data. See [`NOTICE`](NOTICE).

---

## How to cite

```bibtex
@inproceedings{polukoshko2027mhgb,
  title     = {{MHGB}: A Multi-Hop Knowledge Graph Benchmark for Legal Reasoning Evaluation},
  author    = {Polukoshko, Marina and Anichkov, Yegor and Akhmetov, Vadim and
               Oruzheynikova, Nataliya and Bolovtsov, Sergey and
               Ivanova, Marina Alexandrovna and Golosov, Pavel},
  booktitle = {Proceedings of the ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year      = {2027}
}
```

The archived release carries the DOI [10.5281/zenodo.21611555](https://doi.org/10.5281/zenodo.21611555),
which always resolves to the latest version. See also [`CITATION.cff`](CITATION.cff).

---

## Ethics and intended use

The statutory corpus consists of published federal codes — official acts in the public
domain containing no personal data. The fabulas are synthetic scenarios written to
instantiate statutory reasoning; they are not records of real cases and contain no
identifiable individuals.

MHGB measures the legal reasoning of language models under controlled conditions. It is
not a legal-advice system and has not been validated for practical use. A benchmark score
says nothing about a model's fitness to answer real legal questions, and publishing one
does not endorse deploying that model in a legal setting. All findings concern Russian
statutory law and should not be generalized to other jurisdictions.

---

## Contact

Questions, full-dataset requests, and issues: **polukoshko.marina@gmail.com**
