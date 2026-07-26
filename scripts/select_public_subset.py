"""
Select the public stratified subset of MHGB tasks.

The public release ships a stratified subset of the full 600-task benchmark:
10 task instances per cell of the 4 (task type) x 3 (reasoning depth) matrix,
120 instances in total.

Sampling is done at the *fabula* level, not the instance level: each fabula
yields exactly two instances (closed-book and open-book), so closed/open pairs
stay intact and the GAP metric remains computable on the subset.

    12 cells x 5 fabulas x 2 modes = 120 task instances

Selection is deterministic: fabulas within a cell are sorted by their base id
and sampled with a fixed seed, so re-running this script reproduces the exact
same subset.

Usage:
    python scripts/select_public_subset.py \\
        --tasks data/tasks_raw.jsonl \\
        --out data/tasks_public_120.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

SEED = 42
FABULAS_PER_CELL = 5

TASK_TYPES = ["issue_spotting", "rule_selection", "conflict_resolution", "temporal_validity"]
HOP_GROUPS = ["shallow", "medium", "deep"]


def base_id(task_id: str) -> str:
    """Strip the mode suffix: '<uuid>_closed' / '<uuid>_open' -> '<uuid>'."""
    for suffix in ("_closed", "_open"):
        if task_id.endswith(suffix):
            return task_id[: -len(suffix)]
    return task_id


def load_tasks(path: Path) -> list[dict]:
    tasks = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def select(tasks: list[dict], per_cell: int = FABULAS_PER_CELL, seed: int = SEED):
    """Return (selected_tasks, meta) for a stratified fabula-level sample."""
    # group instances by (type, hop_group) -> base_id -> [instances]
    cells: dict[tuple[str, str], dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for task in tasks:
        cell = (task["type"], task["hop_group"])
        cells[cell][base_id(task["id"])].append(task)

    rng = random.Random(seed)
    selected: list[dict] = []
    per_cell_counts: dict[str, int] = {}

    for task_type in TASK_TYPES:
        for hop_group in HOP_GROUPS:
            cell = (task_type, hop_group)
            fabulas = cells.get(cell, {})
            # sort for determinism before sampling
            candidates = sorted(fabulas.keys())
            if len(candidates) < per_cell:
                raise ValueError(
                    f"cell {cell} has only {len(candidates)} fabulas, need {per_cell}"
                )
            chosen = rng.sample(candidates, per_cell)
            for fid in sorted(chosen):
                instances = sorted(fabulas[fid], key=lambda t: t["mode"])
                if len(instances) != 2:
                    raise ValueError(
                        f"fabula {fid} has {len(instances)} instances, expected 2 (closed+open)"
                    )
                selected.extend(instances)
            per_cell_counts[f"{task_type}/{hop_group}"] = per_cell * 2

    meta = {
        "description": (
            "Public stratified subset of the MHGB benchmark. "
            "Sampled at fabula level so closed/open pairs are preserved."
        ),
        "seed": seed,
        "fabulas_per_cell": per_cell,
        "n_fabulas": len(selected) // 2,
        "n_instances": len(selected),
        "n_closed": sum(1 for t in selected if t["mode"] == "closed"),
        "n_open": sum(1 for t in selected if t["mode"] != "closed"),
        "matrix": per_cell_counts,
        "source": "tasks_raw.jsonl (600 instances, 300 fabulas)",
    }
    return selected, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=Path("data/tasks_raw.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/tasks_public_120.jsonl"))
    parser.add_argument("--per-cell", type=int, default=FABULAS_PER_CELL)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    tasks = load_tasks(args.tasks)
    selected, meta = select(tasks, per_cell=args.per_cell, seed=args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for task in selected:
            fh.write(json.dumps(task, ensure_ascii=False) + "\n")

    meta_path = args.out.with_name(args.out.stem + "_meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"wrote {len(selected)} task instances ({meta['n_fabulas']} fabulas) -> {args.out}")
    print(f"wrote metadata -> {meta_path}")
    for cell, n in meta["matrix"].items():
        print(f"  {cell:38} {n}")


if __name__ == "__main__":
    main()
