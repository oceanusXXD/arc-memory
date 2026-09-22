"""Label and train a small architecture smoke set from real LoCoMo rows.

The labels are explicit, source-grounded annotations for a fit smoke test.  The
script never calls a model to invent labels and writes into a task-local run
directory so formal artifacts are untouched.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from arc.agent.io import read_jsonl, write_jsonl
from arc.algorithm.architecture import ARCHITECTURES, solution_record
from arc.algorithm.features import FeatureSchema
from arc.algorithm.selector import decode_joint, load_selector, train_selector
from arc.algorithm.memory import tokenizer_from_config
from arc.agent.config import load_config


# Each entry is a human-readable, source-grounded architecture annotation.  The
# selected IDs are the internal retrieval IDs used by the current selector.
ANNOTATIONS: dict[str, dict[str, Any]] = {
    "conv-26:qa:0": {
        "architecture": "Flat", "source_ids": [3],
        "reason": "One source states the date of Caroline's LGBTQ support-group visit.",
    },
    "conv-26:qa:1": {
        "architecture": "Flat", "source_ids": [1],
        "reason": "One source directly states when Melanie painted a sunrise.",
    },
    "conv-26:qa:14": {
        "architecture": "Chain", "source_ids": [3, 5, 17, 30],
        "reason": "The answer follows a support -> transition -> motivation -> counseling chain.",
    },
    "conv-42:qa:197": {
        "architecture": "Chain", "source_ids": [18, 20],
        "reason": "The two statements connect turtle behavior to Nate's calm/peace preference.",
    },
    "conv-26:qa:4": {
        "architecture": "Tree", "source_ids": [13, 27, 41, 49, 64],
        "reason": "A transition/identity root branches into community, art expression, and self-acceptance evidence.",
    },
    "conv-26:qa:65": {
        "architecture": "Tree", "source_ids": [56, 45],
        "reason": "The transition outcome is the parent concept, with authentic self-expression and art as supporting branches.",
    },
    "conv-30:qa:18": {
        "architecture": "Graph", "source_ids": [1, 2, 12, 19, 40],
        "reason": "Two independent business/interest chains (Jon and Gina) must be answered together.",
    },
    "conv-41:qa:11": {
        "architecture": "Graph", "source_ids": [28, 35, 42, 43],
        "reason": "People met, children at the shelter, and a helping event form cross-linked evidence.",
    },
    "conv-26:qa:15": {
        "architecture": "Cluster", "source_ids": [1, 2, 17, 23, 26, 27, 32, 36, 40, 45, 46, 48, 52, 57, 59],
        "reason": "The answer aggregates distinct activity clusters: art, pottery, advocacy, running, self-care, and family.",
    },
    "conv-42:qa:5": {
        "architecture": "Cluster", "source_ids": [3, 9, 12, 19, 27, 35, 43, 50],
        "reason": "Joanna's hobbies are separate topic/session clusters rather than one causal chain.",
    },
}


def entropy(probabilities: tuple[tuple[str, float], ...]) -> float:
    return -sum(value * math.log(max(value, 1e-12)) for _, value in probabilities)


def make_records(input_path: str | Path, output_path: str | Path, *, budget: int) -> list[dict[str, Any]]:
    rows = {str(row["qa_id"]): dict(row) for row in read_jsonl(input_path)}
    records: list[dict[str, Any]] = []
    for qa_id, label in ANNOTATIONS.items():
        if qa_id not in rows:
            raise KeyError(f"annotated QA is missing from {input_path}: {qa_id}")
        row = rows[qa_id]
        sources = list(row.get("sources") or [])
        source_ids = {int(source["id"]) for source in sources}
        selected = sorted({int(value) for value in label["source_ids"]})
        if not selected or not set(selected) <= source_ids:
            raise ValueError(f"invalid source IDs for {qa_id}: {selected}")
        architecture = str(label["architecture"])
        if architecture not in ARCHITECTURES:
            raise ValueError(f"invalid architecture annotation: {architecture}")
        solution = solution_record(architecture, selected, status="PASS", utility=1.0)
        # The domain entry supplies the architecture-specific terminal record;
        # training still computes exact costs for every sequential prefix.
        record = {
            "schema": "arc",
            "sample_id": row.get("sample_id"),
            "qa_id": qa_id,
            "question": row["question"],
            "sources": sources,
            "budget": budget,
            "budgets": {str(budget): {"budget": budget, "successful_solutions": [solution]}},
            "successful_solutions": [solution],
            "domain": [solution],
            "architecture_annotation": label,
            "annotation_source": "human_source_grounded_smoke",
            "utility_source": "manual_positive_label_for_selector_smoke",
            "requirements": row.get("requirements") or [],
        }
        records.append(record)
    write_jsonl(output_path, records)
    return records


def predict(model, records: list[dict[str, Any]], schema: FeatureSchema, tokenizer: Any, *, before: bool) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in records:
        budget = int(row["budget"])
        # Deployment follows Algorithm 1: decode content first, then score
        # every feasible architecture conditional on each completed source
        # set, and compare complete joint probabilities.
        result = decode_joint(model, None, list(row["sources"]), budget,
                              width=2, top_l=2, schema=schema, input_limit=15872,
                              tokenizer=tokenizer)
        target = row["architecture_annotation"]["architecture"]
        probabilities = dict(result.probabilities)
        output.append({
            "qa_id": row["qa_id"],
            "stage": "before" if before else "after",
            "target_architecture": target,
            "predicted_architecture": result.architecture,
            "architecture_probabilities": probabilities,
            "architecture_entropy": entropy(result.probabilities),
            "architecture_score": result.score,
            "predicted_source_ids": list(result.source_ids),
            "source_score": result.score - result.architecture_score,
            "source_exact": sorted(result.source_ids) == sorted(row["architecture_annotation"]["source_ids"]),
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/processed/requirements.train-dev.jsonl")
    parser.add_argument("--output-dir", default="runs/architecture_smoke")
    parser.add_argument("--config", default="configs/locomo.yaml")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--budget", type=int, default=8192)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"
    records = make_records(args.input, records_path, budget=args.budget)
    config = load_config(args.config)
    first_source = next(source for row in records for source in row.get("sources") or [])
    dimension = int((first_source.get("embedding_metadata") or {}).get("dimension") or len(first_source.get("embedding") or []))
    schema = FeatureSchema.fit(records, dimension=dimension)
    tokenizer = tokenizer_from_config(config)

    # This is a fresh smoke checkpoint. Existing formal checkpoints are never
    # reused and train_selector refuses to overwrite an existing output.
    checkpoint = output_dir / "selector.pt"
    model = __import__("arc.algorithm.selector", fromlist=["Selector"]).Selector(width=args.width, input_size=schema.input_size)
    before = predict(model, records, schema, tokenizer, before=True)
    result = train_selector(records, checkpoint, epochs=args.epochs, width=args.width,
                            schema=schema, tokenizer=tokenizer, seed=7)
    trained = load_selector(checkpoint)
    after = predict(trained, records, schema, tokenizer, before=False)
    report = {
        "records": len(records),
        "architectures": list(ARCHITECTURES),
        "training": result,
        "before_top1_accuracy": sum(row["predicted_architecture"] == row["target_architecture"] for row in before) / len(before),
        "after_top1_accuracy": sum(row["predicted_architecture"] == row["target_architecture"] for row in after) / len(after),
        "after_source_exact_accuracy": sum(row["source_exact"] for row in after) / len(after),
        "before": before,
        "after": after,
        "label_notes": {qa_id: value for qa_id, value in ANNOTATIONS.items()},
    }
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(output_dir / "predictions.jsonl", [*before, *after])
    print(json.dumps({k: report[k] for k in ("records", "training", "before_top1_accuracy", "after_top1_accuracy", "after_source_exact_accuracy")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
