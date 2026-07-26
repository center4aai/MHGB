# Datasheet — MHGB knowledge graph and task set

Following the datasheet-for-datasets convention. Numbers are computed from the released
artifacts (`data/graph.json`, `data/corpus.jsonl`, `data/graph_diagnostics_v2.json`).

---

## Motivation

**Why was this dataset created?** No published benchmark targets legal reasoning over
Russian statutory law, and none represents statutory knowledge as a typed graph used both
to generate multi-hop tasks and to supply evaluation context. MHGB was built to measure
two capabilities separately: what a model knows about the law from its parameters, and how
well it applies norms placed in front of it.

**Who created it and who funded it?** The authors listed in `CITATION.cff`.

---

## Composition

### Statutory corpus (`data/corpus.jsonl`)

Full text of ten codified federal statutes of the Russian Federation, parsed from official
`.docx` publications into 5,570 article records.

| Code | Abbrev. | Articles | Branch of law |
|---|---|---|---|
| Civil Code | ГК | 1,727 | civil |
| Code of Administrative Offenses | КоАП | 1,106 | administrative |
| Tax Code | НК | 790 | tax |
| Labor Code | ТК | 535 | labor |
| Criminal Code | УК | 521 | criminal |
| Housing Code | ЖК | 243 | housing |
| Land Code | ЗК | 193 | land |
| Family Code | СК | 176 | family |
| Constitution | КРФ | 142 | constitutional |
| Urban Planning Code | ГрК | 137 | urban planning |
| **Total** | | **5,570** | |

Per-article fields: `id`, `law`, `article_num` (integer or fractional, e.g. 256.1), `text`,
`valid_from` (known for 68% of articles), `valid_to`, `is_repealed` (193 articles),
`has_repealed_parts`, `branch_of_law`.

The corpus is a **snapshot**. Last revision reflected per code ranges from July 2020
(Constitution) to February 2025 (Criminal, Tax, Administrative Offenses). It is not
updated as statutes are amended; results are tied to this snapshot.

### Knowledge graph (`data/graph.json`)

A typed directed graph, NetworkX node-link format.

| Property | Value |
|---|---|
| Nodes (articles) | 5,519 |
| Edges | 8,760 |
| Intra-code edges | 8,645 |
| Cross-code edges | 115 |
| Structural edges (`refers_to` + `applies_to`) | 7,332 |
| Doctrinal edges (`excludes` + `supplements` + `prevails_over`) | 1,428 |

Nodes per code: ГК 1,715 · КоАП 1,094 · НК 773 · ТК 529 · УК 521 · ЖК 243 · ЗК 192 ·
СК 176 · КРФ 142 · ГрК 134. The gap from 5,570 corpus articles to 5,519 nodes is 27
articles carrying insertion suffixes (e.g. `15.1-1`), which are folded into their base
article; no article is lost.

**Edge types.**

| Type (EN) | Type (RU) | Count | Meaning |
|---|---|---|---|
| `refers_to` | ссылается_на | 4,380 | Explicit textual cross-reference; one article invokes another as an existing rule |
| `applies_to` | применяется_к | 2,952 | Fixes where or to whom another norm applies, without adding content |
| `excludes` | исключает | 715 | Displaces the application of another norm where they conflict |
| `supplements` | дополняет | 381 | Extends or specifies the rule another establishes |
| `prevails_over` | приоритет | 332 | Takes precedence over a competing norm without repealing it |

Largest cross-code pairs: КоАП↔УК 26, ГК↔СК 25, КоАП↔НК 14, НК↔УК 13, ГрК↔ЗК 8.

### Task set

600 task instances from 300 fabulas, each posed closed-book and open-book, over a 4×3
matrix of task type × reasoning depth with 25 fabulas per cell.

| Slice | Count |
|---|---|
| Issue spotting / Rule selection / Conflict resolution / Temporal validity | 150 each |
| Shallow (1–2 norms) / Medium (3–4) / Deep (5–10) | 200 each |

Distribution by branch of law (primary branch = first norm in `norm_ids`): tax 41.0%,
civil 21.3%, administrative 12.7%, criminal 9.0%, urban planning 7.3%, land 5.7%, labor
4.3%, housing 4.3%, family 1.3%, constitutional 0.3%. 20 fabulas (6.7%) span more than one
branch. Deep tasks skew toward tax law (63 of 100), since multi-hop chains can only be
generated where the graph supports them — an acknowledged limitation.

**Public subset** (`data/tasks_public_120.jsonl`): 120 instances from 60 fabulas, 10 per
matrix cell, sampled at fabula level with seed 42. Regenerate with
`scripts/select_public_subset.py`.

Per-task fields: `id` (`<uuid>_closed` / `<uuid>_open`), `mode`, `type`, `difficulty`,
`hop_count`, `hop_group`, `norm_ids` (seed norms), `fabula`, `question`, `answer`
(reference answer in the three-part format), `gold_chain` (ordered steps, each with
`norm_id` and `reasoning`), `context_chunks`, `expected_answer_format`.

---

## Collection process

**Corpus.** Official `.docx` publications of the ten codes, parsed by
`src/mhgb/parse_docs.py` using article-header pattern matching plus document styles, with
special handling for the Constitution's non-standard styling.

**Graph, stage 1 — candidate extraction.** Statutory text states its own dependencies:
provisions cite one another by article number and code. `src/mhgb/build_graph.py` extracts
these with pattern matching over an alias table covering all ten codes and their naming
variants, yielding 7,655 directed candidates, each provisionally labelled `refers_to`.

**Graph, stage 2 — relation typing.** A citation's legal meaning is not visible in its
surface form. An LLM classifier (Qwen3.6-27B, run locally) receives each candidate pair
with the full text of both articles and assigns a type and a direction. Where the
candidate's direction is confirmed the edge is retyped in place; where the relation runs
the other way, the reverse edge is added — 1,105 edges that no citation states in that
direction. Prompt: `prompts/edge_classification.md`.

**Tasks.** Generated from the graph by `src/mhgb/generate_tasks.py`: a subgraph is sampled
so that answering requires traversing the relation types characteristic of the target task
type and depth, then an LLM instantiates it as a concrete factual scenario with a gold
reasoning chain. Prompts: `prompts/task_generation.md`.

**Validation.** Three independent legal experts reviewed, under a frozen blind protocol,
a stratified sample of 70 edges (50 doctrinal, 20 structural) and 24 tasks across the 4×3
matrix. Edge-typing precision against the experts' majority vote: 0.743 overall (95% Wilson
CI [0.630, 0.831]); structural 0.850 [0.640, 0.948], doctrinal 0.700 [0.562, 0.809]. Of 70
edges, only one was rejected by all three experts as connecting unrelated provisions —
89–95% of each expert's rejections were reclassifications of the type, not denials of the
relation. On tasks: raw agreement 0.728, Fleiss κ 0.149 (deflated by prevalence), Gwet's
AC1 0.600. No expert proposed a materially different reasoning chain on any task.

Individual expert annotations are not released: the annotators did not consent to
publication of their identities or their per-item labels.

---

## Preprocessing and labelling

Article text is preserved verbatim from the official publications. Added metadata:
article boundaries, validity dates, repeal flags, branch-of-law labels. Articles with
insertion suffixes are folded into their base article for graph node identity, while the
corpus retains the original records.

Fabulas are written in their own language rather than paraphrasing the provisions they
instantiate: lexical overlap between a fabula and its seed norms averages ≈0.05 (Jaccard)
and ≈0.06 (trigram), so the tasks cannot be solved by surface matching.

---

## Uses

**Intended.** Evaluating multi-hop statutory reasoning of language models; measuring the
separation between parametric legal knowledge and reasoning over supplied context;
research on legal knowledge graphs and on legal-domain evaluation methodology.

**Out of scope.** MHGB is not a retrieval benchmark: open-book context is built around
each task's seed norms, so the governing provisions are always present, and nothing here
evaluates finding them in an unindexed corpus. It is not a legal-advice system, has not
been validated for practical use, and a score on it says nothing about a model's fitness
to answer real legal questions. All findings concern Russian statutory law and do not
transfer to other jurisdictions.

**Contamination.** The closed-book mode assumes the tasks are not in a model's training
data. To preserve that, only a 120-task subset is published openly; the full 600-task set
is browsable through the web interface and available on request. Whether contamination has
occurred for any given model cannot be verified from outside.

---

## Distribution and maintenance

Code under Apache-2.0, derived data under CC BY-4.0, statutory text in the public domain
(Article 1259(6) of the Civil Code of the Russian Federation excludes official documents
of state bodies from copyright). See `LICENSE`, `LICENSE-DATA` and `NOTICE`.

Distributed via this repository and archived with a DOI. The corpus snapshot is fixed;
the construction pipeline is released so the graph can be rebuilt over updated statutory
text or extended to further codes.

Contact: **polukoshko.marina@gmail.com**
