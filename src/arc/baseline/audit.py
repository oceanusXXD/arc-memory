"""Audit records for source selection, tri-state construction and usage."""
from __future__ import annotations
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable
from arc.agent.io import read_jsonl, write_json, write_jsonl

def record(row: dict[str, Any], evidence: list[dict], arm: str) -> dict[str, Any]:
    selection = dict(row.get("selection") or {})
    memories = list(row.get("memories") or [])
    errors = list(row.get("errors") or [])
    usage = dict(row.get("agent_usage") or {})
    selected_ids = selection.get("source_ids") or selection.get("selected_ids") or []
    return {"sample_id": str(row.get("sample_id") or ""), "qa_id": str(row.get("qa_id") or ""), "arm": arm,
            "architecture": row.get("architecture") or selection.get("architecture") or "Flat",
            "status": str(row.get("status") or "unknown"), "construction_status": row.get("construction_status"),
            "memory_mode": str(row.get("memory_mode") or selection.get("mode") or "none"),
            "evidence_source_ids": [str(block.get("source_id")) for block in evidence],
            "evidence_tokens": _evidence_tokens(evidence),
            "context_source_ids": [str(value) for value in row.get("context_sources") or []],
            "selected_ids": list(selected_ids), "selection": selection,
            "memory_count": len(memories), "memories": memories, "errors": errors,
            "source_audit": row.get("source_audit") or selection.get("source_audit"),
            "requirement_audit": row.get("requirement_audit") or selection.get("requirement_audit"),
            "fallback": selection.get("fallback"), "error": row.get("error"),
            "builder_usage": {k: row.get(k) for k in ("construction_calls", "construction_tokens", "construction_usage_complete")},
            "agent_usage": usage, "agent_usage_complete": row.get("agent_usage_complete"),
            "agent_total_tokens": row.get("agent_total_tokens"), "memory_read_tokens": row.get("memory_read_tokens"),
            "retrieval_query_tokens": row.get("retrieval_query_tokens"), "retrieval_latency_ms": row.get("retrieval_latency_ms"), "score": row.get("score")}

def _evidence_tokens(evidence: list[dict]) -> int:
    total = 0
    for block in evidence:
        if block.get("token_count") is None:
            raise ValueError(f"missing frozen token_count for source {block.get('source_id')}")
        total += int(block["token_count"])
    return total

def write_records(path: str | Path, rows: Iterable[dict[str, Any]]) -> int: return write_jsonl(path, rows)

def summarize(path: str | Path) -> dict[str, Any]:
    rows = list(read_jsonl(path)); by_arm: dict[str, list[dict]] = defaultdict(list)
    for row in rows: by_arm[str(row.get("arm") or "unknown")].append(row)
    arms = {}
    for arm, items in sorted(by_arm.items()):
        statuses = Counter(str(item.get("construction_status") or item.get("status") or "unknown") for item in items)
        fallbacks = Counter(str(item.get("fallback")) for item in items if item.get("fallback"))
        known = [item.get("builder_usage", {}).get("construction_tokens") for item in items if item.get("builder_usage", {}).get("construction_tokens") is not None]
        arms[arm] = {"runs": len(items), "statuses": dict(statuses), "fallbacks": dict(fallbacks), "avg_evidence_tokens": sum(int(item.get("evidence_tokens") or 0) for item in items) / len(items) if items else 0.0, "avg_builder_tokens": sum(known) / len(known) if known else None, "builder_tokens_known_items": len(known), "avg_memory_count": sum(int(item.get("memory_count") or 0) for item in items) / len(items) if items else 0.0}
    return {"records": len(rows), "arms": arms}

def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect audit records"); parser.add_argument("--input", required=True); parser.add_argument("--output"); args = parser.parse_args(); result = summarize(args.input)
    if args.output: write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
