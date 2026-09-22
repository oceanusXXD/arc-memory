#!/usr/bin/env python3
"""End-to-end verification against the real API through Claude Code.

This script performs no mocking.  Every question is a genuine LoCoMo question
and every model call is a real Claude Code CLI invocation (cloud SiliconFlow
relay or the repository's local Qwen bridge).  It verifies the pipeline
stage by stage so a failure points at a specific component rather than at
"the algorithm".

Stages:
  1. credentials  - the configured endpoint has usable credentials
  2. agent        - target agent answers a real question from injected memory
  3. builder      - G extracts source-supported memory from real evidence
  4. retrieval    - the fixed hybrid retrieval cache loads (needs `index`)
  5. arms         - each baseline arm runs on one real question

Usage:
    export PYTHONPATH=src
    python scripts/verify_real.py --config configs/verify_small.yaml --stage all
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arc.agent.config import load_config
from arc.agent.data.locomo import load_blocks, split_qas
from arc.agent.reader import ClaudeCodeClient, CompletionError
from arc.agent.runner_helpers import answer_one, build_evidence_for


def _report(name: str, ok: bool, detail: dict) -> dict:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    for key, value in detail.items():
        print(f"        {key}: {value}")
    return {"stage": name, "ok": ok, **detail}


def stage_credentials(config: dict) -> dict:
    client = ClaudeCodeClient(config)
    status = client.credential_status()
    ok = bool(status.get("credentials_present") or not status.get("base_url"))
    return _report("credentials", ok, {
        "base_url": status.get("base_url"),
        "credentials_present": status.get("credentials_present"),
        "model": client.model_for("agent"),
    })


def stage_agent(config: dict, qa: dict, evidence: list[dict]) -> dict:
    result = answer_one(config, qa, evidence)
    usage = result.get("agent_usage") or {}
    ok = result.get("status") == "ok" and bool(result.get("prediction"))
    return _report("agent", ok, {
        "question": qa["question"],
        "reference": qa.get("answer"),
        "prediction": result.get("prediction"),
        "score": result.get("score"),
        "model": usage.get("model"),
        "total_tokens": usage.get("total_tokens"),
        "usage_complete": usage.get("usage_complete"),
    })


def stage_builder(config: dict, qa: dict, evidence: list[dict]) -> dict:
    from arc.algorithm.memory import build_once, memory_packet_from_build, normalize_sources
    normalized, _ = normalize_sources(evidence)
    result = build_once(config, None, normalized,
                        sample_id=str(qa["sample_id"]), qa_id=str(qa["qa_id"]))
    packet = memory_packet_from_build(result, normalized)
    usage = [dict(row) for row in result.usage]
    ok = result.status == "PASS" and bool(result.memories)
    return _report("builder", ok, {
        "status": result.status,
        "memories": list(result.memories),
        "packet": packet.text,
        "calls": [{"role": row.get("role"), "model": row.get("model"),
                   "total_tokens": row.get("total_tokens")} for row in usage],
    })


def stage_retrieval(config: dict, qa: dict, evidence: list[dict]) -> dict:
    from arc.agent.data.retrieval import load_retrieval
    try:
        rows = load_retrieval(config)
    except (FileNotFoundError, ValueError) as exc:
        return _report("retrieval", False, {"error": str(exc),
                                            "hint": "run: python -m arc.agent.data.retrieval index --config <config>"})
    key = (str(qa["sample_id"]), str(qa["qa_id"]))
    row = rows.get(key)
    ok = row is not None and row.get("backend") == "hybrid_rrf"
    return _report("retrieval", ok, {
        "cache_rows": len(rows),
        "question_found": row is not None,
        "backend": (row or {}).get("backend"),
        "retrieved": len((row or {}).get("source_ids") or []),
        "evidence_recall": (row or {}).get("evidence_recall"),
    })


def stage_arms(config: dict, qa: dict, evidence: list[dict], full: list[dict]) -> dict:
    from arc.baseline.evaluate import _packet, _answer
    results = {}
    for arm in ("no_memory", "naive_rag", "full_memory"):
        costs: list[dict] = []
        packet, context = _packet(config, qa, evidence, full, arm)
        row = _answer(config, qa, arm, context, packet.text, costs)
        results[arm] = {"status": row.get("status"), "prediction": row.get("prediction"),
                        "score": row.get("score"), "memory_mode": packet.mode,
                        "constructor_calls": row.get("construction_calls"),
                        "constructor_tokens": row.get("construction_tokens")}
    ok = all(v["status"] == "ok" for v in results.values())
    return _report("baseline arms", ok, results)


def main() -> int:
    parser = argparse.ArgumentParser(description="End-to-end verification against the real Claude Code API")
    parser.add_argument("--config", default="configs/verify_small.yaml")
    parser.add_argument("--split", default="final")
    parser.add_argument("--index", type=int, default=0, help="which real question to verify")
    parser.add_argument("--stage", default="all",
                        choices=["all", "credentials", "agent", "builder", "retrieval", "arms"])
    args = parser.parse_args()

    config = load_config(args.config)
    qas = split_qas(config, args.split)
    if not qas:
        print(f"no questions in split {args.split!r}", file=sys.stderr)
        return 2
    qa = qas[args.index]
    blocks = load_blocks(config)
    full = [b for b in blocks if str(b.get("sample_id")) == str(qa["sample_id"])]
    evidence = build_evidence_for(config, qa, blocks)

    print(f"config: {args.config}")
    print(f"real question: {qa['sample_id']} / {qa['qa_id']}")
    print(f"  {qa['question']}")
    print(f"  reference: {qa.get('answer')!r}  evidence: {qa.get('evidence')}")
    print(f"  evidence rows selected: {len(evidence)}")
    print()

    stages = {
        "credentials": lambda: stage_credentials(config),
        "agent": lambda: stage_agent(config, qa, evidence),
        "builder": lambda: stage_builder(config, qa, evidence),
        "retrieval": lambda: stage_retrieval(config, qa, evidence),
        "arms": lambda: stage_arms(config, qa, evidence, full),
    }
    order = list(stages) if args.stage == "all" else [args.stage]

    reports = []
    for name in order:
        try:
            reports.append(stages[name]())
        except CompletionError as exc:
            reports.append(_report(name, False, {"error": str(exc)[:500]}))
        except Exception as exc:  # noqa: BLE001 - verification must report everything
            reports.append(_report(name, False, {"error": f"{type(exc).__name__}: {exc}"}))
            traceback.print_exc()
        print()

    failed = [r["stage"] for r in reports if not r["ok"]]
    print("=" * 60)
    print(f"stages run: {len(reports)} | passed: {len(reports) - len(failed)} | failed: {len(failed)}")
    if failed:
        print("failed stages: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
