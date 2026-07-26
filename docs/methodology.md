# MHGB — Evaluation Methodology

This document describes how MHGB scores model responses, how open-book context is built,
and how the reliability of the automatic evaluation is established. It is the reference
for the metrics summarized in the paper.

---

## 1. Response format

Every model is prompted, in Russian and at temperature 0, to answer in a three-part
structured format:

```
ПРИМЕНИМЫЕ СТАТЬИ:   [applicable articles, as «ст. N CodeX», comma-separated]
ЦЕПОЧКА РАССУЖДЕНИЙ: [step-by-step legal reasoning]
ВЫВОД:               [conclusion: lawful / unlawful / other]
```

`src/mhgb/eval/response_parser.py` splits the response by header position
(`parse_structured_answer`) and extracts norm identifiers of the form `CODE_NUMBER`
(for example `ТК_261`) from any section (`extract_norms_from_text`). Extraction is
filtered against the corpus, so references to articles that do not exist are dropped
rather than counted.

**Hyphenated article numbers** are disambiguated structurally, because a hyphen plays
three different roles: an insertion suffix (`25.13-1` is one article, folded to its base
`25.13`), a range (`254-269`, which yields both endpoints), and a date (`2020-11-09`,
which is not a norm at all).

---

## 2. Three-level scoring

### Level 1 — Norm coverage (deterministic)

`src/mhgb/eval/norm_coverage.py`. F1 between the set of norms the model names and the
gold norm set:

```
precision = |predicted ∩ gold| / |predicted|
recall    = |predicted ∩ gold| / |gold|
F1        = 2 · precision · recall / (precision + recall)
```

Norm coverage is computed over the union of the *applicable articles* list and the
*reasoning chain*, against the task's seed norms — the articles the task was generated
from. Precision therefore penalizes any norm named beyond that set. This is deliberate:
a model that lists half a code is not thereby more accurate, and measurements on the full
panel show that the large majority of extra norms are noise rather than defensible
alternatives. The gold set is never widened to accommodate model answers, which would
contaminate the reference.

A diagnostic Jaccard measure (`list_vs_reasoning_consistency`) reports how far the
declared article list diverges from the norms actually used in the reasoning; it feeds the
error taxonomy but not the score.

### Level 2 — Step correctness (LLM judge)

`src/mhgb/eval/step_correctness.py`. The judge scores every step of the gold chain:

| Score | Meaning |
|---|---|
| 1 | The norm is applied correctly and the justification matches the reference |
| 0.5 | The norm is mentioned but the justification is incomplete or partly wrong |
| 0 | The norm is not applied, or applied in a fundamentally wrong way |

Steps with no norm anchor (`norm_id = null`) are skipped. Step correctness is the mean
over the remaining steps — one judge call per step. Gold chains average 3.8 scored steps
per task (1–7; shallow ≈ 1.8, medium ≈ 3.4, deep ≈ 6.5).

### Level 3 — Answer correctness (LLM judge)

`src/mhgb/eval/answer_correctness.py`. One judge call per task, scoring the conclusion
only, on {0, 0.33, 0.67, 1}: correct, mostly correct, partially correct, incorrect.
Completeness of the norm list is Level 1's concern and is not re-judged here.

### Final Score

```
FS = α · NC + β · SC + γ · AC,    α = β = γ = 1/3
```

`src/mhgb/eval/final_score.py`. Weights are configurable and must sum to 1.0. All three
components are always reported alongside FS. Re-ranking the panel under norm-heavy,
step-heavy and answer-heavy weightings (0.5/0.25/0.25) leaves the closed-book ordering
essentially unchanged (Kendall τ = 1.000, 0.909, 0.970).

### Degenerate inputs

An empty or whitespace-only response is scored **0 on all three levels without calling the
judge**. This matters: asked to score an empty answer against a reference, judges
hallucinate — they attribute the reference reasoning to the model and award a high score.
The short-circuit fires before the judge call, so the judge prompt itself is untouched.
Truncated-but-non-empty responses are *not* zeroed; they are real, if incomplete, answers.
Zeroed records are marked `EMPTY_RESPONSE_SHORTCIRCUIT` so they can be separated in
analysis from genuine judge zeros.

---

## 3. GAP metric and quadrants

`src/mhgb/eval/gap_metric.py`.

```
GAP = FS_open − FS_closed
```

computed per fabula and also reported per component (norm coverage, step correctness,
answer correctness). Each (model, fabula) pair is classified at threshold τ = 0.5:

| Quadrant | closed | open | Reading |
|---|---|---|---|
| **Knows** | ≥ τ | ≥ τ | Strong parametric knowledge; context confirms it |
| **Reasons** | < τ | ≥ τ | Does not know the norm but reasons well over it |
| **Hallucinates** | ≥ τ | < τ | Knows the law but is disrupted by supplied context |
| **Incompetent** | < τ | < τ | Fails in both modes |

`aggregate_gaps_by_slice` groups results by model, task type, reasoning depth or branch of
law, returning mean, standard deviation and the quadrant distribution.

---

## 4. Open-book context construction

`src/mhgb/eval/context_builders.py`. Context is assembled in two layers.

**Seed norms** — the task's `norm_ids` — are always included and bypass the token budget.
A consequence is that on the deepest tasks the seed norms alone can exceed a model's input
window; those tasks are then excluded for that model, which is a property of the corpus
rather than of the pipeline.

**One-hop neighbors** are added until the budget is exhausted. Traversal runs in both
directions — norms the seed cites and norms that cite the seed. Limits: `max_chunks = 25`
and `token_budget = 40,000` estimated tokens (≈3 characters per token for Russian legal
prose); YandexGPT uses 27,000 because of its 32K input limit. Neighbors are added in
traversal order, without relevance ranking.

Each chunk is rendered as plain text:

```
[norm_id] Article heading (valid from DD.MM.YYYY)
<full article text>
```

**What the model is not told.** Seed norms are not marked: they appear in the same format
as their neighbors, so the model must decide for itself which provisions the facts
trigger. Relation types *are* supplied in the main open-book configuration
(`full_graph_edges`) but **without direction and without explanation** — a bare statement
that a relation of a given type holds between two articles. Validity dates are supplied as
**raw metadata**; the computed judgment of which version was in force at the event date is
never given, as that would be the answer.

Supplying dates was checked for a shortcut: on conflict-resolution tasks, the norm that
wins the conflict is the most recently amended one in 34 of 69 cases with known dates
(49%, chance level for a two-norm choice), with a further 19% tied. Conflicts in this
dataset turn on *lex specialis*, which is date-independent, so dates are supplied
uniformly across all task types.

**Positional order is shuffled per task**, deterministically, by a SHA-256 hash of the
task id. Without this, seed norms always occupied the head of the context and gained an
unearned advantage from primacy effects. Python's built-in `hash()` cannot be used here:
`PYTHONHASHSEED` randomizes it between processes, which would break reproducibility.

Context is a pure function of (task, configuration): every model sees exactly the same
context for a given task, so differences in open-book score reflect reasoning, not
retrieval luck.

---

## 5. Judges

### Selection

The primary judge was chosen by a pilot against manual human annotation of 72
task–answer pairs (24 tasks × 3 models of differing strength), scored with the same judge
prompts. Agreement was measured separately for step correctness and answer correctness
(Cohen's κ plus exact match), on the full set and excluding items whose reference was
flagged as suspect.

| Candidate | SC κ | SC exact | AC κ | AC exact |
|---|---|---|---|---|
| **Llama 3.3 70B** | **0.695** | 0.853 | **0.463** | 0.625 |
| gpt-oss-20B | 0.482 | 0.766 | 0.397 | 0.583 |

Llama 3.3 70B dominates on both components and was fixed as the primary judge; the judge
prompt has been frozen since. Qwen3.6-27B serves as the secondary judge for cross-judge
agreement. Both are outside the leaderboard, so no evaluated model scores its own answers,
and the judge is never told which model produced a response.

**A documented limitation:** both candidates are less graduated than a human annotator on
partial credit, gravitating toward 0/1 where a human assigns 0.33/0.67. AC κ (≈0.46) is
correspondingly lower than SC κ (≈0.70) for both. Answer correctness on partially correct
responses is the weakest point of automatic judging here.

### Cross-judge agreement

The secondary judge re-scores every response of every model. Agreement is Cohen's κ on a
5-bin quantization, `bin(v) = min(4, int(v · 5))`, matched by task id, with bootstrap
confidence intervals. Across the panel: **SC κ ∈ [0.55, 0.67]**, **AC κ ∈ [0.45, 0.58]**.
The secondary judge is uniformly slightly stricter on graded scores, a small constant
shift that does not reorder the leaderboard.

κ depends on how the ordinal scales are coarsened: binarizing to correct/incorrect gives
κ ≈ 0.78. The 5-bin value is reported as the primary one because it matches the resolution
of the scales; finer bins would measure the number of steps rather than judge agreement.
Norm coverage is deterministic and is not re-judged.

---

## 6. Refusal to answer (RtA)

`src/mhgb/eval/rta_detector.py`, run as a separate pass over saved responses so that a
model declining a task is not silently scored as a reasoning failure. The detector
receives the triple (fabula, question, response) and returns `is_rta` plus two independent
fields:

- **`rta_type`** — *how* the model refuses: `generic_refusal`, `scope_refusal`
  ("outside my competence"), and rarer forms.
- **`rta_topic`** — *what* it refuses about: `financial_crime`, `military`, `religious`,
  `minorities`, `political`, `medical`, `other`, or null. `other` is a filled topic — a
  sensitive area outside the named taxonomy — not an unassigned label.

**Counting rule.** The raw detector flags technical artifacts as refusals: context
overflow, empty responses, and truncated degenerate output such as a runaway enumeration
of non-existent articles. The canonical rate counts, in the numerator, only responses
whose *text* is a substantive refusal; the denominator excludes errors, overflows and
empty responses, but retains truncated and degenerate answers, since those are genuine
failure modes and are scored as such at Levels 1–3. A single function
(`compute_rta_rate`) implements this for both the analysis scripts and the web interface.

**The rate is judge-dependent.** On identical YandexGPT responses, one detector judge
returned 10.5% and another 5.5%, the difference being whether hedged, uncertain answers
count as refusals. Topical profiles were stable across both. Absolute rates are therefore
comparable only within one detector; cross-judge κ on the refusal label is reported
alongside the rate (pooled κ = 0.917 over the models with a non-empty positive class).
Where a model never refuses, the positive class is empty and cross-judge validation is not
applicable — this is reported as "not checked", never as "confirmed zero".

---

## 7. Error taxonomy

`src/mhgb/analysis/error_analysis.py`. Only responses scoring below 0.5 are classified;
0.5 and above is `ok` and outside the taxonomy. Categories are checked in priority order
and the first match wins.

| Category | Condition | Reading |
|---|---|---|
| `hallucination` | NC_reasoning − NC_list > 0.3 | Norms used in reasoning but absent from the declared list, or invented |
| `multi_fail` | NC_list < 0.3 **and** SC < 0.4 **and** AC < 0.33 | All three components collapse |
| `norm_miss` | NC_list < 0.3 | The relevant provisions were not found |
| `reasoning_fail` | SC < 0.4 | Norms found, chain of steps wrong |
| `answer_fail` | otherwise | Norms and steps largely right, conclusion wrong |

Thresholds are design decisions matched to each scale, not derived values: 0.3 is a
conventional "low F1" for retrieval; 0.4 sits just below the minimum partial credit of 0.5
per step; 0.33 is below the first non-zero level of the answer scale. They are held fixed
across versions so that distributions stay comparable.

Two consequences worth stating: `norm_miss` does not mean "found nothing" — 0.25 counts,
0.35 does not; and `multi_fail` is checked *before* `norm_miss`, which separates a systemic
collapse from a targeted failure on norms.

---

## 8. Statistics

`src/mhgb/analysis/statistics.py`.

- **Bootstrap percentile CI** — 2,000 resamples, seed 42, 95% interval from the
  [2.5, 97.5] percentiles. Applied to Final Score, its components, and GAP, per model.
- **Paired t-test on GAP** — over each model's closed/open fabula pairs, two-tailed, with
  a bootstrap CI on the mean difference.
- **Two-sample bootstrap** for depth slices (5,000 resamples). Shallow and deep tasks are
  *different fabulas*, not two conditions on one fabula, so no pairing exists and a paired
  test would be invalid here.
- **Sensitivity analysis** — Kendall's τ between the equal-weight ranking and alternative
  weightings, plus threshold variation τ ∈ {0.4, 0.5, 0.6}.
- **Wilson intervals** for proportions, including expert-validated edge precision.
- **Fleiss' κ** for multi-annotator agreement. On skewed annotation, κ is deflated by the
  prevalence paradox, so raw agreement, Gwet's AC1 and PABAK are reported alongside it and
  the interpretation is read from all four.

---

## 9. Reproducibility notes

- The graph is versioned; all results regenerate from the released corpus.
- Context assembly is deterministic given (task, configuration) — including the per-task
  shuffle.
- The evaluation pipeline is append-only and resumable: rerunning skips completed
  (task, model, mode) triples and retries failed ones.
- Some models need a specific setting to produce a scorable answer at all. This is a
  condition of comparability, documented rather than tuned: Gemma-4 runs with its native
  thinking channel disabled, because on some deep tasks it runs away and leaves the answer
  field empty, so its reasoning goes inline like every other general model.
