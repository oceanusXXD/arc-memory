"""LoCoMo baselines and evaluation with auditable cost records."""
from __future__ import annotations
import argparse, copy, hashlib, json, random, time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from tqdm import tqdm
from arc.agent import Agent
from arc.agent.config import load_config, public_config, run_dir, file_sha256
from arc.agent.data.locomo import load_blocks, split_qas
from arc.agent.data.retrieval import blocks_by_source, load_retrieval, select_sources
from arc.agent.data.score import aggregate_scores, exact_match_prediction, score_prediction
from arc.agent.io import append_jsonl, read_json, write_json, write_jsonl
from arc.agent.memory import MemoryPacket, MemoryRequest, render_memory_text
from arc.agent.reader import completion_error_row
from arc.agent.runtime import agent_usage_row, question_answer_task, run_agent
from arc.algorithm.adapter import build_memory, rank_pack
from arc.algorithm.architecture import normalize_architecture
from arc.algorithm.memory import build_once, configured_input_cost, complete_input_cost, memory_packet_from_build, normalize_sources, tokenizer_basis, tokenizer_from_config
from arc.baseline.audit import record as audit_record

ARMS = ("no_memory", "full_memory", "naive_rag", "rank_pack", "our")
PROMPT_NAME = "locomo"

def _qa_key(qa: dict) -> tuple[str, str]: return str(qa["sample_id"]), str(qa["qa_id"])

def construction_usage(costs: list[dict]) -> dict[str, Any]:
    calls = sum(int(c.get("request_count") or 0) for c in costs)
    complete = all(bool(c.get("usage_complete")) for c in costs)
    out: dict[str, Any] = {"construction_calls": calls, "construction_usage_complete": complete}
    for src, dst in (("prompt_tokens", "construction_prompt_tokens"), ("completion_tokens", "construction_completion_tokens"), ("total_tokens", "construction_tokens")):
        vals = [c.get(src) for c in costs if c.get(src) is not None]
        out[dst] = sum(vals) if complete and len(vals) == len(costs) else None
        out[dst + "_reported"] = sum(vals)
    return out

def _answer(config: dict, qa: dict, arm: str, context: list[dict], memory_text: str, costs: list[dict]) -> dict:
    row = {k: qa.get(k) for k in ("sample_id", "qa_id", "qa_index", "category", "question")}
    count_tokens = tokenizer_from_config(config)
    row.update(arm=arm, reference=qa.get("answer", ""), prediction="", score=0.0, exact_match=0.0, context_sources=[str(b.get("source_id")) for b in context], memory_read_tokens=count_tokens(memory_text) if memory_text else 0, memory_read_token_basis="frozen_builder_tokenizer")
    try:
        result = run_agent(question_answer_task(qa["question"]), memory_text, config=config)
        costs.append(agent_usage_row(str(qa["sample_id"]), str(qa["qa_id"]), f"agent:{arm}", result.usage))
        row.update(prediction=result.outcome, status="ok", score=score_prediction(result.outcome, qa.get("answer", ""), int(qa["category"])), exact_match=exact_match_prediction(result.outcome, qa.get("answer", ""), int(qa["category"])), agent_usage=result.usage)
    except Exception as exc:
        error = completion_error_row(str(qa["sample_id"]), str(qa["qa_id"]), f"agent:{arm}", "agent", exc); costs.append(error)
        row.update(status="agent_error", error=error["error"], agent_usage=getattr(exc, "usage", {}))
    usage = row.get("agent_usage") or {}; complete = bool(usage.get("usage_complete"))
    row.update(agent_model=usage.get("model"), agent_usage_complete=complete)
    for name in ("prompt_tokens", "completion_tokens", "total_tokens"): row["agent_" + name] = usage.get(name) if complete else None
    return row

def _packet(config: dict, qa: dict, evidence: list[dict], full: list[dict], arm: str) -> tuple[MemoryPacket, list[dict]]:
    if arm == "no_memory": return MemoryPacket("", {"mode": "none", "status": "ACTIVE_EMPTY"}), []
    if arm == "full_memory":
        # Full baseline uses the same retrieved E and frozen builder G as the method;
        # the only difference is that selection is bypassed.
        normalized_full, _ = normalize_sources(evidence)
        result = build_once(config, None, normalized_full,
                            sample_id=str(qa["sample_id"]), qa_id=str(qa["qa_id"]), architecture="Flat")
        return memory_packet_from_build(result, normalized_full), normalized_full
    normalized, _ = normalize_sources(evidence)
    if arm == "naive_rag": return MemoryPacket(render_memory_text(evidence), {"mode": "raw", "status": "ACTIVE"}, evidence_source_ids=tuple(str(b.get("source_id")) for b in evidence)), evidence
    budget = int((config.get("budgets") or {}).get("builder_input_tokens") or 4096)
    if arm == "rank_pack":
        selected = rank_pack(qa["question"], normalized, budget, config=config, architecture="Flat"); result = build_once(config, None, selected, sample_id=str(qa["sample_id"]), qa_id=str(qa["qa_id"]), architecture="Flat")
        return memory_packet_from_build(result, selected), selected
    # ``our`` is evaluated as one query-independent write followed by the
    # fixed query-time Retrieve path.  The question is passed only to the
    # latter and never to H or G.
    write_request = MemoryRequest("", normalized, arm, *_qa_key(qa), phase="write", write_batch=normalized)
    writer = Agent(config, build_memory)
    write_packet = writer.prepare_memory(write_request)
    state = write_packet.persistent_state
    read_request = MemoryRequest(qa["question"], [], arm, *_qa_key(qa), phase="query", memory_state=state)
    read_packet = writer.prepare_memory(read_request)
    selected_ids = set((write_packet.selected.get("selection") or {}).get("source_ids") or [])
    context = [source for source in normalized if int(source["id"]) in {int(value) for value in selected_ids}]
    packet = MemoryPacket(
        read_packet.text,
        {**read_packet.selected, "write_selection": write_packet.selected.get("selection"),
         "selection": write_packet.selected.get("selection"), "architecture": write_packet.selected.get("architecture", "Flat")},
        tuple([*write_packet.usage, *read_packet.usage]),
        read_packet.memories,
        tuple([*write_packet.errors, *read_packet.errors]),
        tuple(str(source.get("source_id")) for source in context),
        state,
    )
    return packet, context

def _load_journal(path: Path) -> list[dict]:
    if not path.exists(): return []
    rows = []
    with path.open("rb+") as handle:
        while True:
            start = handle.tell(); line = handle.readline()
            if not line: break
            if not line.endswith(b"\n"):
                try:
                    rows.append(json.loads(line))
                    handle.seek(0, 2); handle.write(b"\n")
                except ValueError:
                    interrupted = path.with_name(path.name + f".interrupted-{start}.bin")
                    if not interrupted.exists(): interrupted.write_bytes(line)
                    handle.truncate(start)
                break
            rows.append(json.loads(line))
    return rows

def _completed(row: dict) -> bool:
    errors = [e for e in row.get("errors", []) if not isinstance(e, dict) or e.get("reason") != "audit_skipped"]
    return (row.get("status") == "ok" and not errors
            and row.get("agent_usage_complete") is True
            and (row.get("construction_usage_complete") is True or row.get("construction_calls") == 0))

def _merge_journal(journal: list[dict]) -> list[dict]:
    merged = {}
    for item in journal:
        row = item["result"]
        key = (str(row["sample_id"]), str(row["qa_id"]), row["arm"])
        if key not in merged or not _completed(merged[key]):
            merged[key] = row
    return list(merged.values())

def run_eval(config: dict[str, Any], arms: list[str], split: str, limit: int | None = None, output: str | None = None, *, sample: int | None = None, resume: bool = False, qa_ids: set[str] | None = None, progress_callback=None) -> dict[str, Any]:
    if not arms or len(set(arms)) != len(arms) or any(a not in ARMS for a in arms): raise ValueError(f"arms must be unique members of {ARMS}")
    config = copy.deepcopy(config); qas = split_qas(config, split)
    if qa_ids is not None: qas = [qa for qa in qas if qa["qa_id"] in qa_ids]
    if sample is not None: qas = [qas[i] for i in sorted(random.Random(config.get("seed", 7)).sample(range(len(qas)), sample))]
    elif limit is not None: qas = qas[:limit]
    if not qas: raise ValueError("no questions in selected split")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"); target = Path(output) if output else run_dir(config) / timestamp / "scores.jsonl"
    paths = {"scores": target, "costs": target.with_name(target.stem + ".costs.jsonl"), "audit": target.with_name(target.stem + ".audit.jsonl"), "summary": target.with_name(target.stem + ".summary.json"), "manifest": target.with_name(target.stem + ".manifest.json")}
    settings = {"config": public_config(config), "prompt_name": PROMPT_NAME,
                "tokenizer_basis": tokenizer_basis(config), "split": split, "arms": arms,
                "checkpoint_sha256": file_sha256(config["selector"].get("checkpoint", "")) if config["selector"].get("checkpoint") else None,
                "selected_questions": [list(_qa_key(q)) for q in qas]}
    if resume:
        if not paths["manifest"].exists() or read_json(paths["manifest"]).get("settings") != settings: raise ValueError("resume settings differ")
        journal = _load_journal(paths["audit"])
    else:
        if any(path.exists() for path in paths.values()): raise FileExistsError("run files already exist")
        write_json(paths["manifest"], {"created_at": timestamp, "settings": settings, "files": {k: str(v) for k, v in paths.items()}}); journal = []
    rows = _merge_journal(journal); done = {(str(r["sample_id"]), str(r["qa_id"]), r["arm"]) for r in rows if _completed(r)}; blocks = load_blocks(config); lookup = blocks_by_source(blocks)
    # Audit is the durable transaction journal. Reconstruct canonical views if
    # interrupted between writing audit, scores, and costs.
    if resume:
        write_jsonl(target, rows)
        write_jsonl(paths["costs"], (cost for item in journal for cost in item.get("costs", [])))
    blocks_by_sample: dict[str, list[dict]] = {}
    for block in blocks:
        blocks_by_sample.setdefault(str(block.get("sample_id")), []).append(block)
    count_tokens = tokenizer_from_config(config)
    retrieval = load_retrieval(config)
    # Build the write-stage candidate cache without consulting the current QA.
    # The retrieval cache contributes frozen source vectors/metadata; candidate
    # order and truncation use only the conversation history and exact builder
    # budget.  H therefore receives E_i, never q.
    source_vectors_by_sample: dict[str, dict[str, list[float]]] = defaultdict(dict)
    embedding_metadata_by_sample: dict[str, dict[str, Any]] = {}
    for cached in retrieval.iter_rows():
        sample_id = str(cached.get("sample_id"))
        for source_id, vector in (cached.get("source_vectors") or {}).items():
            source_vectors_by_sample[sample_id].setdefault(str(source_id), vector)
        if cached.get("embedding_metadata") and sample_id not in embedding_metadata_by_sample:
            embedding_metadata_by_sample[sample_id] = dict(cached["embedding_metadata"])

    def write_candidates(sample_id: str) -> list[dict[str, Any]]:
        source_cap = int(config["retrieval"].get("source_cap_tokens", 8192))
        max_blocks = int(config["retrieval"].get("max_blocks", 64))
        input_limit = int(config["budgets"].get("builder_input_limit", 15872))
        vectors = source_vectors_by_sample.get(str(sample_id), {})
        metadata = embedding_metadata_by_sample.get(str(sample_id))
        selected: list[dict[str, Any]] = []
        token_total = 0
        for block in sorted(blocks_by_sample.get(str(sample_id), []),
                            key=lambda item: (int(item.get("turn_index", 0)), int(item.get("chunk_index", 0)), str(item.get("source_id")))):
            source_id = str(block.get("source_id"))
            vector = vectors.get(source_id)
            if vector is None or metadata is None:
                continue
            item = dict(block)
            item["embedding"] = vector
            item["embedding_metadata"] = dict(metadata)
            block_tokens = int(item.get("token_count") or 0)
            if len(selected) >= max_blocks or token_total + block_tokens > source_cap:
                continue
            candidate = [*selected, item]
            if configured_input_cost(config, None, candidate) > input_limit:
                continue
            selected.append(item)
            token_total += block_tokens
        return selected

    write_candidates_by_sample = {sample_id: write_candidates(sample_id) for sample_id in blocks_by_sample}
    for qa in tqdm(qas, desc=f"eval:{split}"):
        if all((*_qa_key(qa), arm) in done for arm in arms): continue
        started_retrieval = time.perf_counter(); rr = retrieval[_qa_key(qa)]
        evidence, dropped = select_sources(
            rr,
            lookup,
            int(config["retrieval"].get("source_cap_tokens", 8192)),
            int(config["retrieval"].get("max_blocks", 64)),
            cost_fn=lambda candidate: configured_input_cost(config, qa["question"], candidate),
            input_limit=int(config["budgets"].get("builder_input_limit", 15872)),
        )
        full = blocks_by_sample.get(str(qa["sample_id"]), [])
        write_evidence = write_candidates_by_sample.get(str(qa["sample_id"]), [])
        for arm in arms:
            key = (*_qa_key(qa), arm)
            if key in done: continue
            config["execution"] = {"resume_calls": True, "work_key": {
                "stage": "evaluation", "qa_id": qa["qa_id"], "split": split,
                "arm": arm, "seed": config.get("seed", 7),
                "budget": config["budgets"]["builder_input_tokens"],
                "checkpoint_sha256": settings["checkpoint_sha256"],
                "config_sha256": hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest(),
            }}
            started = time.perf_counter(); costs: list[dict] = []
            arm_evidence = write_evidence if arm == "our" else evidence
            packet, context = _packet(config, qa, arm_evidence, full, arm)
            construction_errors = [cost for cost in packet.usage if cost.get("status") == "error"]
            if construction_errors:
                from arc.agent.reader import CompletionError
                raise CompletionError("Construction failed; agent request not attempted", construction_errors[-1])
            row = _answer(config, qa, arm, context, packet.text, costs); row.update(construction_usage(list(packet.usage)), selection=packet.selected, memories=list(packet.memories), errors=list(packet.errors), memory_mode=packet.mode, construction_status=packet.selected.get("status"), total_latency_ms=int((time.perf_counter()-started)*1000), retrieval_backend=rr.get("backend"), retrieval_query_tokens=count_tokens(str(qa.get("question") or "")), retrieval_latency_ms=int((time.perf_counter()-started_retrieval)*1000), dropped_source_ids=dropped if arm in {"naive_rag", "rank_pack", "our"} else [])
            row.update(split=split, seed=config.get("seed", 7), budget=config["budgets"]["builder_input_tokens"],
                       architecture=normalize_architecture((row.get("selection") or {}).get("architecture", "Flat")))
            row["selector_latency_ms"] = (packet.selected.get("selection") or {}).get("latency_ms", 0)
            costs = [*packet.usage, *costs]
            for cost in costs: cost["arm"] = arm
            audit = audit_record(row, context, arm); audit.update(result=row, costs=costs, prompt_name=PROMPT_NAME)
            append_jsonl(paths["audit"], [audit]); append_jsonl(paths["costs"], costs)
            rows = [r for r in rows if (str(r["sample_id"]), str(r["qa_id"]), r["arm"]) != key]
            rows.append(row); write_jsonl(target, rows)
            if _completed(row): done.add(key)
            write_json(paths["summary"], aggregate_scores(rows))
            if progress_callback: progress_callback(row)
    summary = aggregate_scores(rows); write_json(paths["summary"], summary); return {**{k: str(v) for k, v in paths.items()}, "items": len(rows), "metrics": summary}

def main() -> None:
    parser = argparse.ArgumentParser(description="Run LoCoMo evaluation"); sub = parser.add_subparsers(dest="command", required=True); run = sub.add_parser("run"); run.add_argument("--config", default="configs/locomo.yaml"); run.add_argument("--arms", nargs="+", default=["no_memory", "full_memory"]); run.add_argument("--split", default="final"); run.add_argument("--limit", type=int); run.add_argument("--sample", type=int); run.add_argument("--output"); run.add_argument("--resume", action="store_true"); run.add_argument("--deterministic", action="store_true"); args = parser.parse_args(); config = load_config(args.config)
    if args.deterministic: config["decoder"]["temperature"] = config["agent"]["temperature"] = 0.0
    print(json.dumps(run_eval(config, args.arms, args.split, args.limit, args.output, sample=args.sample, resume=args.resume), ensure_ascii=False, indent=2))
if __name__ == "__main__": main()
