"""Local experiment bookkeeping required by the evaluation protocol.

This module deliberately does not execute a model.  It validates data-group
isolation and turns immutable run records into calibration and paper tables.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from arc.agent.config import load_config, run_dir, write_manifest
from arc.agent.io import read_jsonl, write_json


GROUPS = ("D_seed", "D_value", "D_dev", "D_cal", "D_eval")


def validate_groups(config: dict[str, Any]) -> dict[str, Any]:
    groups = dict(config.get("experiment_groups") or {})
    missing = [name for name in GROUPS if not groups.get(name)]
    if missing:
        raise ValueError(f"experiment_groups is missing: {', '.join(missing)}")
    owner: dict[str, str] = {}
    overlaps: dict[str, list[str]] = {}
    for name in GROUPS:
        for group_id in map(str, groups[name]):
            if group_id in owner:
                overlaps.setdefault(group_id, [owner[group_id]]).append(name)
            owner[group_id] = name
    if overlaps:
        raise ValueError(f"history groups overlap across experimental partitions: {overlaps}")
    return {"groups": {name: list(map(str, groups[name])) for name in GROUPS}, "counts": {name: len(groups[name]) for name in GROUPS}}


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _bootstrap_ci(values: list[float], draws: int = 2000, seed: int = 7) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    rng = random.Random(seed); n = len(values)
    samples = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(draws))
    return samples[int(.025 * (draws - 1))], samples[int(.975 * (draws - 1))]


def paired_summary(rows: Iterable[dict[str, Any]], full_arm: str = "full_memory") -> dict[str, Any]:
    # Compare the same QA IDs and keep failed attempts as scored failures.
    by_arm: dict[str, dict[tuple[str, str], float]] = defaultdict(dict)
    for row in rows:
        score = _finite(row.get("score"))
        if score is not None and int(row.get("category", 0)) in {1, 2, 3, 4}:
            by_arm[str(row["arm"])][(str(row["sample_id"]), str(row["qa_id"]))] = score
    full = by_arm.get(full_arm, {})
    result = {}
    for arm, values in sorted(by_arm.items()):
        grouped: dict[str, list[float]] = defaultdict(list)
        for key in full.keys() & values.keys():
            grouped[key[0]].append(full[key] - values[key])
        paired = [mean(grouped[group]) for group in sorted(grouped)]
        result[arm] = {
            "paired_questions": sum(len(v) for v in grouped.values()),
            "independent_history_groups": len(paired),
            "mean_full_minus_arm": mean(paired) if paired else None,
            "paired_bootstrap_95_ci": list(_bootstrap_ci(paired)) if len(paired) >= 2 else [None, None],
            "groups_with_worse_than_full": sum(value > 0 for value in paired),
        }
    return result


def summarize(scores_path: str | Path, costs_path: str | Path, output: str | Path | None = None) -> dict[str, Any]:
    from arc.agent.data.score import aggregate_scores
    scores = list(read_jsonl(scores_path))
    costs = list(read_jsonl(costs_path))
    # Modern score rows contain the same measured totals as their audit records.
    # Old shared cost files cannot reliably identify the arm; do not guess.
    summary = {"paired_vs_full": paired_summary(scores), "arms": aggregate_scores(scores),
               "unassigned_cost_rows": sum(not row.get("arm") for row in costs)}
    if output:
        write_json(output, summary)
    return summary


def break_even(extra_offline_cost: float, online_saving: float) -> float | None:
    return max(0.0, extra_offline_cost) / online_saving if online_saving > 0 else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment accounting and calibration")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate-groups"); validate.add_argument("--config", default="configs/locomo.yaml")
    summary = sub.add_parser("summarize"); summary.add_argument("--scores", required=True); summary.add_argument("--costs", required=True); summary.add_argument("--output")
    args = parser.parse_args()
    if args.command == "validate-groups": result = validate_groups(load_config(args.config))
    else: result = summarize(args.scores, args.costs, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
