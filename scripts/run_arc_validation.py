#!/usr/bin/env python3
"""Small, semantic validation runbook.

This is deliberately separate from ``run_arc_runbook.py``.  The formal
runbook owns the complete Grok/compile/train/final pipeline; this command is a
cheap validation loop for deciding whether the selector learns useful source
sets.  It never treats matching gold source IDs as the success criterion.

Typical pilot:

    export PYTHONPATH=src
    python scripts/run_arc_validation.py --stage coverage
    python scripts/run_arc_validation.py --stage oracle --oracle-limit 6
    python scripts/run_arc_validation.py --stage make-train --train-limit 40
    python scripts/run_arc_validation.py --stage train --epochs 10 --seeds 7
    python scripts/run_arc_validation.py --stage eval --dev-limit 20 --budget 4096
    python scripts/run_arc_validation.py --stage report

The model stages use the configured real provider and therefore may make API
calls.  Coverage, archive construction, and report stages are local.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    rows: list[dict[str, Any]] = []
    with target.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{target}:{number}: expected a JSON object")
            rows.append(value)
    return rows


def write_json(path: str | Path, value: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{__import__('os').getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(target)
    return target


def write_jsonl(path: str | Path, values: Iterable[dict[str, Any]]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{__import__('os').getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(target)
    return target


def load_inputs(args: argparse.Namespace) -> tuple[dict[str, dict], dict[str, dict]]:
    qas = {str(row["qa_id"]): row for row in read_jsonl(args.qa)}
    requirements = {str(row["qa_id"]): row for row in read_jsonl(args.requirements)}
    return qas, requirements


def gold_positions(qa: dict[str, Any], row: dict[str, Any]) -> list[int]:
    gold = {str(value) for value in qa.get("evidence") or []}
    positions: list[int] = []
    for position, source in enumerate(row.get("sources") or [], 1):
        dia_id = str(source.get("dia_id"))
        source_id = str(source.get("source_id", ""))
        # The second form supports source IDs such as sample:D1:3:0.
        source_dia = source_id.split(":")[-2] if ":" in source_id else source_id
        if dia_id in gold or source_dia in gold:
            positions.append(position)
    return sorted(set(positions))


def selected_rows(qas: dict[str, dict], requirements: dict[str, dict], split: str | None = None,
                 limit: int | None = None, complete_gold_only: bool = False) -> list[tuple[dict, dict, list[int]]]:
    result: list[tuple[dict, dict, list[int]]] = []
    for qid, row in requirements.items():
        qa = qas.get(qid)
        if qa is None or (split and str(row.get("split")) != split):
            continue
        positions = gold_positions(qa, row)
        if complete_gold_only and (not qa.get("evidence") or len(positions) != len(qa["evidence"])):
            continue
        result.append((qa, row, positions))
    result.sort(key=lambda item: (str(item[1].get("sample_id")), int(item[0].get("qa_index", 0)), str(item[0]["qa_id"])))
    return result[:limit] if limit is not None else result


def coverage(args: argparse.Namespace) -> dict[str, Any]:
    qas, requirements = load_inputs(args)
    by_split: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "qas": 0, "complete": 0, "partial": 0, "none": 0,
        "gold_evidence_blocks": 0, "mapped_evidence_blocks": 0,
        "missing_examples": [],
    })
    for qid, row in requirements.items():
        qa = qas.get(qid)
        if qa is None:
            continue
        split = str(row.get("split") or "unknown")
        record = by_split[split]
        record["qas"] += 1
        gold = list(qa.get("evidence") or [])
        mapped = gold_positions(qa, row)
        record["gold_evidence_blocks"] += len(gold)
        record["mapped_evidence_blocks"] += len(mapped)
        if gold and len(mapped) == len(gold):
            record["complete"] += 1
        elif mapped:
            record["partial"] += 1
        else:
            record["none"] += 1
            if len(record["missing_examples"]) < 10:
                record["missing_examples"].append({"qa_id": qid, "evidence": gold,
                                                    "candidate_count": len(row.get("sources") or [])})
    for record in by_split.values():
        record["qa_coverage"] = record["complete"] / record["qas"] if record["qas"] else None
        record["evidence_coverage"] = (record["mapped_evidence_blocks"] / record["gold_evidence_blocks"]
                                        if record["gold_evidence_blocks"] else None)
    output = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "qa_path": str(args.qa), "requirements_path": str(args.requirements),
              "splits": dict(by_split)}
    path = write_json(Path(args.output_dir) / "coverage.json", output)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"wrote {path}")
    return output


def make_train(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_jsonl(args.compilation)
    candidates = [row for row in rows if str(row.get("split")) == "train"
                  and row.get("domain_complete") is True
                  and (row.get("annotation") or {}).get("valid") is True]
    usable: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    pass_counts: list[int] = []
    unique_pass_counts: list[int] = []
    for row in candidates:
        budgets = row.get("budgets") or {}
        pass_count = sum(len(value.get("successful_solutions") or value.get("successful_sets") or []) for value in budgets.values())
        labels = [solution for value in budgets.values() for solution in
                  value.get("successful_solutions") or value.get("successful_sets") or []]
        if not budgets:
            labels = row.get("successful_solutions") or row.get("successful_sets") or []
        # A row with no semantic PASS is not a selector training example.  Do
        # not replace its labels with gold evidence just to increase n.
        if not labels:
            rejected.append({"qa_id": row.get("qa_id"), "reason": "no_successful_set"})
            continue
        verified = {(str(item.get("architecture", "Flat")), tuple(sorted(item["source_ids"]))) for item in row.get("evaluations") or []
                    if item.get("status") == "PASS"
                    and (item.get("source_audit") or {}).get("status") == "PASS"
                    and (item.get("requirement_audit") or {}).get("status") == "PASS"
                    and not any(usage.get("status") == "error" for usage in item.get("usage") or [])}
        unique_labels = set()
        for source_set in labels:
            if isinstance(source_set, dict):
                unique_labels.add((str(source_set.get("architecture", "Flat")), tuple(sorted(source_set.get("source_ids") or []))))
            else:
                unique_labels.add(("Flat", tuple(sorted(source_set))))
        if not unique_labels <= verified:
            raise ValueError(f"{row.get('qa_id')}: training labels lack successful compiler audits")
        usable.append(row)
        pass_counts.append(pass_count or len(row.get("successful_sets") or []))
        unique_pass_counts.append(len(unique_labels))
        if args.train_limit is not None and len(usable) >= args.train_limit:
            break
    target = Path(args.output_dir) / "compilation.train.validation.jsonl"
    write_jsonl(target, usable)
    report = {"input": str(args.compilation), "output": str(target), "requested": args.train_limit,
              "candidate_rows": len(candidates), "usable_rows": len(usable), "rejected": rejected,
              "rows_with_multiple_budget_pass_sets": sum(value > 1 for value in pass_counts),
              "total_pass_sets": sum(pass_counts),
              "total_unique_pass_sets": sum(unique_pass_counts),
              "rows_with_multiple_distinct_pass_sets": sum(value > 1 for value in unique_pass_counts),
              "avg_pass_sets_per_row": sum(pass_counts) / len(pass_counts) if pass_counts else 0.0}
    write_json(Path(args.output_dir) / "training_archive.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def train(args: argparse.Namespace) -> dict[str, Any]:
    from arc.agent.config import load_config
    from arc.algorithm.selector import load_selector, train_selector_file

    config = load_config(args.config)
    input_path = Path(args.train_input)
    if not input_path.exists():
        raise FileNotFoundError(f"missing {input_path}; run --stage make-train first")
    seeds = [int(value) for value in str(args.seeds).split(",") if value.strip()]
    outputs: list[dict[str, Any]] = []
    for seed in seeds:
        settings = copy.deepcopy(config)
        settings["seed"] = seed
        target = Path(args.output_dir) / f"selector.seed-{seed}.pt"
        result = train_selector_file(input_path, target, epochs=args.epochs,
                                     learning_rate=args.learning_rate, width=args.width,
                                     config=settings, resume=args.resume)
        load_selector(target)
        outputs.append({"seed": seed, **result, "checkpoint": str(target)})
    report = {"input": str(input_path), "epochs": args.epochs, "seeds": seeds, "checkpoints": outputs}
    write_json(Path(args.output_dir) / "training.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def _gold_oracle_one(config: dict[str, Any], qa: dict, row: dict, positions: list[int], arm: str) -> dict[str, Any]:
    from arc.agent.data.score import exact_match_prediction, score_prediction
    from arc.agent.runtime import question_answer_task, run_agent
    from arc.algorithm.memory import build_once, configured_input_cost, memory_packet_from_build

    sources = list(row.get("sources") or [])
    if arm == "gold_oracle":
        selected = [source for index, source in enumerate(sources, 1) if index in set(positions)]
    elif arm == "full_memory":
        selected = sources
    else:
        raise ValueError(f"unknown oracle arm: {arm}")
    local = copy.deepcopy(config)
    local["execution"] = {"resume_calls": True, "work_key": {
        "stage": "validation_oracle", "qa_id": qa["qa_id"], "arm": arm}}
    result = build_once(local, None, selected,
                        sample_id=str(qa["sample_id"]), qa_id=str(qa["qa_id"]))
    packet = memory_packet_from_build(result, selected)
    prediction = "No information available"
    agent_usage: dict[str, Any] = {}
    status = "empty_memory"
    builder_errors = [item for item in result.usage if item.get("status") == "error"]
    if builder_errors:
        status = "builder_error"
    elif packet.text:
        agent_config = copy.deepcopy(config)
        agent_config["execution"] = {"resume_calls": True, "work_key": {
            "stage": "validation_oracle", "qa_id": qa["qa_id"], "arm": arm + "_agent"}}
        try:
            agent = run_agent(question_answer_task(qa["question"]), packet.text, config=agent_config)
            prediction, agent_usage, status = agent.outcome, agent.usage, "ok"
        except Exception as exc:  # preserve failed rows for diagnosis
            status = "agent_error"
            agent_usage = {**getattr(exc, "usage", {}), "error": str(exc)[:500]}
    cost = configured_input_cost(config, None, selected) if selected else 0
    measured = status not in {"builder_error", "agent_error"}
    return {"arm": arm, "source_ids": [int(source["id"]) for source in selected],
            "builder_status": result.status, "builder_memories": list(result.memories),
            "source_audit": result.source_audit.__dict__,
            "requirement_audit": result.requirement_audit.__dict__,
            "structure_audit": result.structure_audit.__dict__,
            "prediction": prediction, "status": status,
            "score": score_prediction(prediction, qa.get("answer", ""), int(qa["category"])) if measured else None,
            "exact_match": exact_match_prediction(prediction, qa.get("answer", ""), int(qa["category"])) if measured else None,
            "input_tokens": cost, "builder_usage": list(result.usage), "agent_usage": agent_usage}


def oracle(args: argparse.Namespace) -> dict[str, Any]:
    from arc.agent.config import load_config

    qas, requirements = load_inputs(args)
    selected = selected_rows(qas, requirements, split=args.split, limit=args.oracle_limit,
                             complete_gold_only=True)
    if not selected:
        raise ValueError("no complete gold-evidence rows in the requested split")
    config = load_config(args.config)
    # Warm the tokenizer before any future extension adds parallel workers;
    # concurrent first imports are unreliable in some Transformers builds.
    from arc.algorithm.memory import tokenizer_from_config
    tokenizer_from_config(config)
    rows: list[dict[str, Any]] = []
    for qa, row, positions in selected:
        full = _gold_oracle_one(config, qa, row, positions, "full_memory")
        gold = _gold_oracle_one(config, qa, row, positions, "gold_oracle")
        rows.append({"qa_id": qa["qa_id"], "sample_id": qa["sample_id"], "question": qa["question"],
                     "answer": qa.get("answer"), "gold_positions": positions,
                     "full": full, "gold_oracle": gold,
                     "token_saving": (1 - gold["input_tokens"] / full["input_tokens"]
                                       if full["input_tokens"] else None)})
        write_json(Path(args.output_dir) / "oracle.progress.json", {"expected": len(selected), "rows": rows})
        print(qa["qa_id"], gold["score"], full["score"], flush=True)
        if gold["score"] is None or full["score"] is None:
            raise RuntimeError(f"oracle API failure for {qa['qa_id']}; retained in oracle.progress.json; no quality conclusion")
    report = {"n": len(rows),
              "gold_mean_score": sum(row["gold_oracle"]["score"] for row in rows) / len(rows),
              "full_mean_score": sum(row["full"]["score"] for row in rows) / len(rows),
              "gold_exact_match": sum(row["gold_oracle"]["exact_match"] for row in rows) / len(rows),
              "full_exact_match": sum(row["full"]["exact_match"] for row in rows) / len(rows),
              "avg_token_saving": sum(row["token_saving"] for row in rows if row["token_saving"] is not None) / len(rows),
              "rows": rows}
    # The CLI has additional system/context overhead. Keep its measured usage
    # distinct from the frozen tokenizer's serialized builder input size.
    usage_totals = {}
    for arm in ("full", "gold_oracle"):
        calls = [usage for row in rows for usage in row[arm]["builder_usage"]]
        complete = all(usage.get("usage_complete") is True and usage.get("prompt_tokens") is not None
                       for usage in calls)
        usage_totals[arm] = {
            "serialized_input_tokens": sum(row[arm]["input_tokens"] for row in rows),
            "provider_prompt_tokens": sum(usage["prompt_tokens"] for usage in calls) if complete else None,
            "builder_calls": sum(usage.get("request_count", 0) for usage in calls),
            "builder_latency_ms": sum(usage.get("latency_ms", 0) for usage in calls),
            "usage_complete": complete,
        }
    report["usage_totals"] = usage_totals
    for field in ("serialized_input_tokens", "provider_prompt_tokens"):
        full_total, gold_total = (usage_totals[arm][field] for arm in ("full", "gold_oracle"))
        report[f"saving_{field}"] = 1 - gold_total / full_total if full_total and gold_total is not None else None
    write_json(Path(args.output_dir) / "oracle.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, ensure_ascii=False, indent=2))
    return report


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    from arc.agent.config import load_config
    from arc.baseline.evaluate import run_eval

    config = load_config(args.config)
    qas, requirements = load_inputs(args)
    dev = selected_rows(qas, requirements, split="dev", limit=args.dev_limit)
    qa_ids = {qa["qa_id"] for qa, _, _ in dev}
    if not qa_ids:
        raise ValueError("no dev QA rows found in requirements input")
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"missing selector checkpoint: {checkpoint}")
    config["selector"]["checkpoint"] = str(checkpoint)
    config["budgets"]["builder_input_tokens"] = int(args.budget)
    output_root = Path(args.output_dir) / "evaluation" / f"budget-{args.budget}"
    results: dict[str, Any] = {"split": "dev", "budget": args.budget, "qa_ids": sorted(qa_ids), "arms": {}}
    for arm in ("full_memory", "rank_pack", "our"):
        path = output_root / f"{arm}.jsonl"
        manifest = path.with_name(path.stem + ".manifest.json")
        result = run_eval(config, [arm], "dev", output=str(path), resume=manifest.exists(), qa_ids=qa_ids)
        results["arms"][arm] = result
    write_json(output_root / "evaluation.json", results)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return results


def report(args: argparse.Namespace) -> dict[str, Any]:
    from arc.agent.io import read_jsonl
    from arc.agent.data.score import aggregate_scores
    from run_arc_runbook import paired_summary

    root = Path(args.output_dir) / "evaluation" / f"budget-{args.budget}"
    paths = {arm: root / f"{arm}.jsonl" for arm in ("full_memory", "rank_pack", "our")}
    if any(not path.exists() for path in paths.values()):
        raise FileNotFoundError(f"missing evaluation rows under {root}; run --stage eval first")
    rows = {arm: list(read_jsonl(path)) for arm, path in paths.items()}
    evaluation = json.loads((root / "evaluation.json").read_text(encoding="utf-8"))
    expected = set(evaluation["qa_ids"])
    for arm, values in rows.items():
        if len(values) != len(expected) or {row["qa_id"] for row in values} != expected:
            raise ValueError(f"{arm}: incomplete or duplicate evaluation QA rows")
        if any(row.get("status") != "ok" or row.get("construction_usage_complete") is not True
               or row.get("agent_usage_complete") is not True for row in values):
            raise ValueError(f"{arm}: incomplete API calls or usage; resume evaluation before reporting")
    comparisons = {
        "our_vs_full": paired_summary(rows["our"], rows["full_memory"], samples=args.bootstrap_samples),
        "our_vs_rank_pack": paired_summary(rows["our"], rows["rank_pack"], samples=args.bootstrap_samples),
    }
    output = {"budget": args.budget, "arms": {arm: aggregate_scores(value) for arm, value in rows.items()},
              "comparisons": comparisons,
              "interpretation": {
                  "quality_is_measured_on_final_agent_scores": True,
                  "gold_source_id_match_is_not_a_success_criterion": True,
                  "pilot_conclusion": "indicative_only; use the prespecified CI gate on a formal dev set",
              }}
    write_json(root / "report.json", output)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Semantic validation runbook")
    parser.add_argument("--stage", choices=("coverage", "oracle", "make-train", "train", "eval", "report"), required=True)
    parser.add_argument("--config", default="configs/locomo.yaml")
    parser.add_argument("--qa", default="data/processed/locomo_qa.jsonl")
    parser.add_argument("--requirements", default="data/processed/requirements.train-dev.jsonl")
    parser.add_argument("--compilation", default="runs/locomo_arc/compilation.train-dev.jsonl")
    parser.add_argument("--output-dir", default="runs/locomo_arc/validation")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--oracle-limit", type=int, default=6)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--dev-limit", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--seeds", default="7")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--train-input", default="runs/locomo_arc/validation/compilation.train.validation.jsonl")
    parser.add_argument("--checkpoint", default="runs/locomo_arc/validation/selector.seed-7.pt")
    parser.add_argument("--budget", type=int, default=4096)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    stages = {"coverage": coverage, "oracle": oracle, "make-train": make_train,
              "train": train, "eval": evaluate, "report": report}
    stages[args.stage](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
