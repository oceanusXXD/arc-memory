#!/usr/bin/env python3
"""Evidence-based runbook ledger and resumable, non-destructive execution.

The state is a ledger, never evidence by itself. File signatures invalidate
cached checks. Model work is allowed only after its inputs have been checked.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
STATE = ROOT / "runs/arc_run_state.json"


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def signature(path):
    stat = Path(path).stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "inode": stat.st_ino}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name("." + path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def rows(path):
    with Path(path).open("rb") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                raise ValueError(f"{path}:{number}: invalid JSON; original file preserved") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: expected object")
            yield number, row


class Ledger:
    def __init__(self):
        self.data = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {
            "schema": "arc_run_state", "created_at": now(), "runbook": "docs/runbook.md",
            "key_fields": ["stage", "qa_id", "split", "arm", "seed", "budget", "config_sha256"],
            "stages": {}, "blockers": {}, "failures": [], "events": [], "resources": {},
            "existing_call_paths": [str(p) for p in Path("runs").rglob("calls/*.json")],
        }
        self.start = time.monotonic()
        self.cpu_start = time.process_time()
        self.data.setdefault("artifacts", {})
        self.data.setdefault("tasks", {})
        self.data.setdefault("invocations", []).append({"started_at": now(), "command": sys.argv})
        self.invocation = self.data["invocations"][-1]
        self.apply_decisions()

    def apply_decisions(self):
        path = Path("runs/arc_run_decisions.json")
        if not path.exists():
            return
        decisions = json.loads(path.read_text(encoding="utf-8"))
        self.data["decisions"] = decisions
        for field, blocker in (("evidence", "evidence_decision"),):
            if field in decisions and blocker in self.data["blockers"]:
                previous = self.data["blockers"].pop(blocker)
                self.data.setdefault("resolved_blockers", {})[blocker] = {**previous, "status": "RESOLVED", "resolution": decisions[field]}
        evidence = decisions.get("evidence", {})
        if evidence.get("decision") == "exclude_from_train_and_calibration":
            self.data["effective_counts"] = {"train_1_4": evidence["train_1_4_expected"], "dev_1_4": evidence["dev_1_4_expected"], "train_dev_1_4": evidence["train_dev_1_4_expected"], "final": 400}
        scope = decisions.get("annotation_scope")
        if scope:
            ids = scope["qa_ids"]
            if len(ids) != len(set(ids)) or len(ids) != scope["total_target"]:
                raise ValueError("annotation target must contain exactly the requested number of unique QA IDs")
            self.data["annotation_scope"] = scope
            stage = self.data["stages"].get("annotations")
            if stage:
                stage.update(expected=scope["total_target"], full_runbook_requirement=scope["full_runbook_requirement"],
                             split_targets=scope["split_counts"], missing=max(0, scope["total_target"]-stage.get("complete", 0)))

    def save(self, stage=None):
        if stage:
            self.data["stage"] = stage
        self.data["updated_at"] = now()
        self.invocation.update(wall_seconds=round(time.monotonic() - self.start, 3),
                               cpu_seconds=round(time.process_time() - self.cpu_start, 3),
                               max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        atomic_json(STATE, self.data)

    def event(self, action, **detail):
        self.data["events"].append({"at": now(), "action": action, **detail})
        self.save()

    def block(self, key, reason, **detail):
        self.data["blockers"][key] = {"status": "BLOCKED", "reason": reason, **detail}
        self.data["status"] = "BLOCKED"
        self.save()


def inventory(ledger):
    import numpy as np
    from arc.agent.config import load_config, public_config
    from arc.agent.data.retrieval import _validate_retrieval_row, _write_retrieval_index, _read_retrieval_index
    from annotate_with_grok import _validate as validate_annotation

    ledger.data["status"] = "RUNNING"
    ledger.save("inventory")
    config = load_config("configs/locomo.yaml")
    ledger.data["config"] = public_config(config)
    ledger.data["config_sha256"] = digest(public_config(config))
    qas = {r["qa_id"]: r for _, r in rows("data/processed/locomo_qa.jsonl")}
    qa_count = sum(1 for _ in rows("data/processed/locomo_qa.jsonl"))
    assert qa_count == len(qas) == 1986
    blocks = [r for _, r in rows("data/processed/locomo_blocks.jsonl")]
    assert len(blocks) == len({b["source_id"] for b in blocks}) == 5882
    assert all(b["text"] and isinstance(b["token_count"], int) for b in blocks)
    source_map = {b["source_id"]: b for b in blocks}
    splits = {sample: split for split, samples in config["splits"].items() for sample in samples}
    counts = collections.Counter(splits[q["sample_id"]] for q in qas.values())
    core = collections.Counter(splits[q["sample_id"]] for q in qas.values() if q["category"] in (1, 2, 3, 4))
    assert dict(counts) == {"train": 1157, "dev": 429, "final": 400}
    assert dict(core) == {"train": 885, "dev": 341, "final": 314}
    ledger.data["stages"]["prepare"] = {"status": "VERIFIED_SKIP", "qas": len(qas), "blocks": len(blocks), "splits": dict(counts), "categories_1_4": dict(core)}
    audit = json.loads(Path("data/processed/locomo_audit.json").read_text())
    excluded = {x["qa_id"] for x in ledger.data.get("decisions", {}).get("evidence", {}).get("excluded", [])}
    issues = [x for x in audit["qa_audit"]["missing_evidence"]
              if f'{x["sample_id"]}:qa:{x["qa_index"]}' not in excluded]
    if issues:
        ledger.block("evidence_decision", "Runbook requires a human correction/exclusion decision for these evidence IDs; no rows silently removed.", items=issues)
    packages = {}
    for package in ("numpy", "PyYAML", "tqdm", "torch", "rank-bm25", "transformers", "sentence-transformers"):
        packages[package] = importlib.metadata.version(package)
    ledger.data["stages"]["dependencies"] = {"status": "VERIFIED_SKIP", "packages": packages}
    ledger.save()

    for path in (Path("data/cache/retrieval/locomo_topk.jsonl"), Path("data/cache/verify_small/retrieval/locomo_topk.jsonl")):
        name = str(path)
        prior = ledger.data["artifacts"].get(name, {})
        if prior.get("signature") == signature(path) and prior.get("status") == "VERIFIED":
            print(f"verified cache unchanged: {name}", flush=True)
            continue
        offsets, recalls, status_counts = {}, [], collections.Counter()
        expected = set(qas) if "verify_small" not in name else {q["qa_id"] for q in qas.values() if q["sample_id"] == "conv-49"}
        sha = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                sha.update(line)
                row = json.loads(line)
                _validate_retrieval_row(row, f"{path}:{offset}")
                key = (row["sample_id"], row["qa_id"])
                assert key not in offsets and row["qa_id"] in expected
                assert qas[key[1]]["sample_id"] == key[0] and qas[key[1]]["question"] == row["question"]
                ids = row["source_ids"]
                assert len(ids) == len(set(ids)) == len(row["scores"])
                assert all(i in source_map and source_map[i]["sample_id"] == key[0] for i in ids)
                metadata = row["embedding_metadata"]
                assert metadata["model"] == config["retrieval"]["embedding_model"]
                dim = metadata["dimension"]
                qv = np.asarray(row["query_vector"])
                assert qv.shape == (dim,) and np.isfinite(qv).all() and np.linalg.norm(qv) > 0
                vectors = np.asarray([row["source_vectors"][i] for i in ids])
                assert vectors.shape == (len(ids), dim) and np.isfinite(vectors).all()
                assert np.all(np.linalg.norm(vectors, axis=1) > 0)
                evidence = set(qas[key[1]]["evidence"])
                recall = len(evidence & set(row["dia_ids"])) / len(evidence) if evidence else 1.0
                assert abs(recall - row["evidence_recall"]) < 1e-8, (key, recall, row["evidence_recall"])
                offsets[key] = offset
                recalls.append(recall)
                status_counts[metadata.get("revision")] += 1
                if len(offsets) % 100 == 0:
                    ledger.data["stages"]["retrieval_scan"] = {"status": "RUNNING", "path": name, "checked_rows": len(offsets)}
                    ledger.save()
                    print(f"cache schema/vectors/evidence: {name} {len(offsets)}/{len(expected)}", flush=True)
        assert {key[1] for key in offsets} == expected
        sidecar_valid = _read_retrieval_index(path) == offsets
        if not sidecar_valid:
            _write_retrieval_index(path, offsets)
        manifest = json.loads(path.with_name("index_manifest.json").read_text())
        assert manifest["items"] == len(offsets)
        assert abs(manifest["avg_evidence_recall"] - sum(recalls) / len(recalls)) < 1e-8
        record = {"status": "VERIFIED", "signature": signature(path), "sha256": sha.hexdigest(), "rows": len(offsets), "qa_ids": sorted(expected), "avg_evidence_recall": sum(recalls) / len(recalls), "embedding_revisions": dict(status_counts), "sidecar": "verified_existing" if sidecar_valid else "refreshed_offsets_only"}
        ledger.data["artifacts"][name] = record
        ledger.event("verified_skip_retrieval", path=name, rows=len(offsets), sidecar=record["sidecar"])

    question_keys = collections.defaultdict(list)
    for q in qas.values():
        question_keys[(q["sample_id"], q["question"])].append(q["qa_id"])
    paths = sorted(set(Path("runs").rglob("*.jsonl")) | set(Path("data/processed").glob("requirements*.jsonl")))
    for path in paths:
        name = str(path)
        if name.startswith("runs/arc_runbook/"):
            continue
        prior = ledger.data["artifacts"].get(name, {})
        if prior.get("signature") == signature(path) and prior.get("status") == "VERIFIED":
            continue
        ids, fingerprints, statuses, errors = [], set(), collections.Counter(), []
        count = 0
        for number, row in rows(path):
            count += 1
            qid = row.get("qa_id")
            if not qid and row.get("schema") == "arc" and row.get("sources"):
                candidates = question_keys[(row["sources"][0]["sample_id"], row.get("question"))]
                if len(candidates) == 1:
                    qid = candidates[0]
            if qid:
                assert qid in qas
                ids.append(qid)
            if "grok" in name and "requirements" in name:
                validate_annotation(row)
                assert qas[qid]["question"] == row["question"]
            if row.get("schema") == "arc" and "evaluations" in row:
                assert qid is not None, f"cannot uniquely recover QA identity in {path}:{number}"
                if row.get("status") == "annotation_error":
                    statuses["annotation_error"] += 1
                    errors.append({"line": number, "qa_id": qid, "error": row.get("annotation")})
                    continue
                assert isinstance(row.get("domain_complete"), bool)
                statuses["annotation_valid" if (row.get("annotation") or {}).get("valid") else "annotation_invalid"] += 1
                statuses["domain_complete" if row["domain_complete"] else "domain_incomplete"] += 1
                for evaluation in row["evaluations"]:
                    assert evaluation["status"] in ("PASS", "FAIL", "UNKNOWN")
                    statuses[evaluation["status"]] += 1
                passed = {(str(e.get("architecture", "Flat")), tuple(e["source_ids"]))
                          for e in row["evaluations"] if e["status"] == "PASS"}
                recorded = {(str(s.get("architecture", "Flat")), tuple(s["source_ids"]))
                            for s in row.get("successful_solutions") or []}
                assert recorded == passed
                assert all(isinstance(d["cost"], int) and d["cost"] >= 0 for d in row["domain"])
            if row.get("arm"):
                statuses[row.get("status", "unknown")] += 1
                for metric in ("score", "exact_match", "construction_calls", "agent_total_tokens", "total_latency_ms"):
                    if metric not in row:
                        errors.append({"line": number, "missing": metric})
            fingerprints.add(digest(row))
        ledger.data["artifacts"][name] = {"status": "VERIFIED" if not errors else "INCOMPLETE_SCHEMA", "signature": signature(path), "rows": count, "unique_payloads": len(fingerprints), "qa_ids": sorted(set(ids)), "splits": dict(collections.Counter(splits[qas[q]["sample_id"]] for q in set(ids))), "statuses": dict(statuses), "schema_issues": errors}
        ledger.save()
        print(f"artifact: {name} {count} rows / {len(set(ids))} QA", flush=True)

    verify_annotation_artifacts(ledger)
    import torch
    from arc.algorithm.selector import load_selector
    checkpoints = {}
    for path in Path("models").glob("*.pt"):
        record = torch.load(path, map_location="cpu", weights_only=True)
        model = load_selector(path)
        checkpoints[str(path)] = {"status": "VERIFIED_LOADABLE_SMOKE" if "verify_small" in str(path) else "NEEDS_TRAINING_PROVENANCE", "signature": signature(path), "implementation": record["implementation"], "parameter_count": sum(p.numel() for p in model.parameters()), "metadata_keys": [k for k in record if k != "state_dict"], "epochs_provenance": record.get("epochs"), "seed_provenance": record.get("seed")}
    ledger.data["checkpoints"] = checkpoints
    ledger.data["stages"]["inventory"] = {"status": "PASS", "finished_at": now()}
    ledger.data["stages"]["retrieval_scan"] = {"status": "VERIFIED_SKIP", "rows": 2182}
    ledger.save()
    resource_summary(ledger)


def resource_summary(ledger):
    before = set(ledger.data["existing_call_paths"])
    summary = {}
    failed = []
    for label, paths in (("historical", [p for p in Path("runs").rglob("calls/*.json") if str(p) in before]), ("this_task", [p for p in Path("runs").rglob("calls/*.json") if str(p) not in before])):
        totals = collections.Counter()
        by_role = {}
        for path in paths:
            row = json.loads(path.read_text())
            usage = row.get("usage") or {}
            role = row.get("role", "unknown")
            role_totals = by_role.setdefault(role, collections.Counter())
            role_totals["calls"] += 1
            for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
                role_totals[field + "_reported"] += int(usage.get(field) or 0)
            totals["log_records"] += 1
            totals["request_count"] += int(row.get("request_count") or 0)
            totals["provider_requests"] += len(row.get("api_requests") or [])
            totals["provider_attempts_without_response"] += sum(x.get("status") is None for x in row.get("api_requests") or [])
            totals["success" if row.get("status") == "ok" else "failed"] += 1
            totals["usage_complete_records"] += bool(usage.get("usage_complete"))
            for key in ("prompt_tokens", "completion_tokens", "total_tokens", "latency_ms"):
                totals[key + "_reported"] += int(usage.get(key) or 0)
            if row.get("status") != "ok":
                failed.append({"path": str(path), "scope": label, "role": row.get("role"), "error": str(row.get("error", ""))[:1000]})
        summary[label] = dict(totals)
        summary[label]["by_role"] = {key: dict(value) for key, value in by_role.items()}
    summary["money"] = None
    summary["money_reason"] = "No verified billing amounts/prices; token usage is reported without fabricated currency cost. Historical Grok responses have no saved usage."
    summary["token_accounting_note"] = "Reported tokens are from saved responses. Upstream attempts without a response may have additional unknown usage; no zero-cost assumption is made."
    summary["gpu_available"] = False
    teacher_totals = collections.Counter()
    teacher_failures = []
    for path in Path("runs/locomo_arc/teacher_calls").glob("*.json"):
        row = json.loads(path.read_text())
        teacher_totals["log_records"] += 1
        teacher_totals["request_count"] += int(row.get("request_count") or 0)
        teacher_totals[row.get("status", "unknown")] += 1
        usage = row.get("usage") or {}
        for field in ("input_tokens", "output_tokens", "total_tokens", "cost_in_usd_ticks"):
            if usage.get(field) is not None:
                teacher_totals[field + "_reported"] += int(usage[field])
        teacher_totals["usage_records"] += bool(usage)
        if row.get("status") != "ok":
            teacher_failures.append({"path": str(path), "qa_id": (row.get("work_key") or {}).get("qa_id"),
                                     "status": row.get("status"), "http_status": row.get("http_status"),
                                     "error": row.get("error")})
    summary["teacher_this_task"] = dict(teacher_totals)
    summary["wrong_teacher_qwen"] = summary["this_task"].get("by_role", {}).get("teacher", {})
    summary["new_model_calls"] = summary["this_task"].get("request_count", 0) + teacher_totals.get("request_count", 0)
    ledger.data["resources"].update(summary)
    ledger.data["resources"]["runner_wall_seconds"] = round(sum(x.get("wall_seconds", 0) for x in ledger.data["invocations"]), 3)
    ledger.data["resources"]["runner_cpu_seconds"] = round(sum(x.get("cpu_seconds", 0) for x in ledger.data["invocations"]), 3)
    ledger.data["resources"]["runner_peak_rss_kib"] = max((x.get("max_rss_kib", 0) for x in ledger.data["invocations"]), default=0)
    ledger.data["historical_call_failures"] = failed
    ledger.data["teacher_call_failures"] = teacher_failures
    ledger.save()


def assess(ledger):
    from arc.agent.config import load_config, public_config
    from arc.baseline.evaluate import _completed
    config = load_config("configs/locomo.yaml")
    qas = [r for _, r in rows("data/processed/locomo_qa.jsonl")]
    splits = {sample: split for split, samples in config["splits"].items() for sample in samples}
    core = [q for q in qas if splits[q["sample_id"]] in ("train", "dev") and q["category"] in (1, 2, 3, 4)]
    excluded = {x["qa_id"] for x in ledger.data.get("decisions", {}).get("evidence", {}).get("excluded", [])}
    core = [q for q in core if q["qa_id"] not in excluded]
    full_expected = len(core)
    scope = ledger.data.get("annotation_scope") or {}
    expected = scope.get("total_target", full_expected)
    ledger.data.setdefault("protocol", {"selector_seeds": [7, 17, 27], "budgets": [1024, 2048, 4096], "epochs": 30, "primary_seed": 7, "compile_limit": 16, "compile_d": 8})
    annotation_index = index_records("data/processed/requirements.train-dev.jsonl")
    annotations = len(annotation_index)
    compiled_path = Path("runs/locomo_arc/compilation.train-dev.jsonl")
    compiled = len(index_records(compiled_path)) if compiled_path.exists() else 0
    ledger.data["stages"]["annotations"] = {"status": "TARGET_COMPLETE" if annotations == expected and expected < full_expected else "PASS" if annotations == expected else "BLOCKED", "complete": annotations, "expected": expected, "missing": expected-annotations, "full_runbook_requirement": full_expected, "historical_final_smoke": 3, "split_targets": scope.get("split_counts")}
    ledger.data["stages"]["compilation"] = {"status": "PASS" if compiled == full_expected else "BLOCKED", "complete": compiled, "expected": full_expected, "missing": full_expected-compiled, "historical_final_smoke": 3}
    if annotations < full_expected:
        ledger.block("compilation_dependency", "Full train/dev teacher annotations are incomplete; the user has capped this annotation run.", missing=full_expected-annotations)
    if compiled < full_expected:
        ledger.block("training_dependency", "Full training split compilation is not available; final smoke checkpoint cannot be used for formal training/evaluation.")
    for stage in ("annotation", "compilation"):
        for qa in core:
            key = work_key(ledger, stage, qa["qa_id"], splits[qa["sample_id"]], seed=7, budget=15872)
            if digest(key) not in ledger.data["tasks"]:
                task_record(ledger, key, "BLOCKED", reason="builder_service/teacher_service/evidence_decision" if stage == "annotation" else "annotation_dependency")
    checkpoints = [f"models/arc_selector.seed-{seed}.pt" for seed in ledger.data["protocol"]["selector_seeds"]]
    ledger.data["stages"].setdefault("selector_training", {"status": "BLOCKED", "complete": 0, "expected_seeds": ledger.data["protocol"]["selector_seeds"], "epochs": 30, "checkpoint_targets": checkpoints})
    for seed in ledger.data["protocol"]["selector_seeds"]:
        key = work_key(ledger, "selector_training", "__all_train__", "train", seed=seed)
        if digest(key) not in ledger.data["tasks"]:
            task_record(ledger, key, "BLOCKED", reason="training_dependency")
    if ledger.data["stages"]["selector_training"].get("status") != "PASS":
        ledger.block("dev_dependency", "Formal multi-seed checkpoints are missing; b* and CI cannot be calculated.")
    if ledger.data["stages"].get("dev_budget", {}).get("b_star") is None:
        ledger.block("final_dependency", "Runbook step 8 requires a formal checkpoint and a budget selected on Dev.")
    for qa in qas:
        if splits[qa["sample_id"]] != "final":
            continue
        for arm in ("no_memory", "full_memory", "naive_rag", "rank_pack", "our"):
            key = work_key(ledger, "evaluation", qa["qa_id"], "final", arm=arm, seed=7, budget=ledger.data["stages"].get("dev_budget", {}).get("b_star"))
            if digest(key) not in ledger.data["tasks"]:
                task_record(ledger, key, "BLOCKED", reason="final_dependency", budget_resolution="await_dev_b_star")
    ledger.data["stages"].setdefault("dev_budget", {"status": "BLOCKED", "b_star": None, "budgets": [1024, 2048, 4096], "qa_count_1_4": ledger.data.get("effective_counts", {}).get("dev_1_4", 341)})
    ledger.data["stages"].setdefault("final_evaluation", {"status": "BLOCKED", "complete": 0, "expected": 2000, "qa_count": 400})
    external = ["Top-k", "MMR", "PACMS", "Qwen-Rerank-pack", "Qwen-Prompt-Pick", "Provence", "EXIT", "LLMLingua-2", "LongLLMLingua", "RECOMP-*", "Prompt-SAW", "GraphMemix-text", "Context-Picker-memory", "RepoShapley-teacher", "块释放＋前缀目标"]
    ledger.block("external_baseline_specifications", "Runbook lists names but supplies no executable implementations or exact variants/checkpoints/protocols for the 15 external baselines; method choices require resolution before full-method comparisons.", methods=external)
    ledger.data["stages"]["external_baselines"] = {"status": "BLOCKED", "complete": 0, "expected": 15}
    smoke_rows = []
    for path in sorted(Path("runs/verify_small").rglob("scores.jsonl")):
        settings = json.loads(path.with_name("scores.manifest.json").read_text())["settings"]
        for number, row in rows(path):
            key = {"stage": "smoke_evaluation", "qa_id": row["qa_id"], "split": settings["split"], "arm": row["arm"], "seed": settings["config"]["seed"], "budget": settings["config"]["budgets"]["builder_input_tokens"], "config_sha256": digest(settings["config"])}
            key_id = digest(key)
            task = ledger.data["tasks"].setdefault(key_id, {"key": key, "status": "VERIFIED_SKIP_SMOKE", "attempts": []})
            reference = {"path": str(path), "line": number, "status": row["status"], "calls": row["construction_calls"], "checkpoint_sha256": None}
            if reference not in task["attempts"]:
                task["attempts"].append(reference)
            smoke_rows.append(row)
    real_smoke = any(r["arm"] == "our" and r["construction_calls"] > 0 and _completed(r) for r in smoke_rows)
    ledger.data["stages"]["smoke"] = {"status": "VERIFIED_SKIP" if real_smoke else "BLOCKED", "historical_score_rows": len(smoke_rows), "formal_score_rows": 0, "note": "Historical successful real builder+selector+agent chain; no repeat API/training/evaluation. Checkpoint epochs/seed hashes were not saved in historical training metadata."}
    cost_check = Path("runs/arc_compile_cost_check.json")
    if cost_check.exists():
        ledger.data["compiled_cost_validation"] = json.loads(cost_check.read_text())
    acceptance = {"historical_real_chain": "PASS" if real_smoke else "FAIL", "current_real_chain": "BLOCKED" if "builder_service" in ledger.data["blockers"] or "teacher_service" in ledger.data["blockers"] else "NOT_FULLY_TESTED", "full_selector_training": ledger.data["stages"]["selector_training"]["status"], "dev_b_star": ledger.data["stages"]["dev_budget"].get("b_star"), "quality_gap_our_full": None, "saving_build": None, "quality_gap_our_rank_pack": None, "build_token_gap_our_rank_pack": None, "one_sided_95_ci_upper": None, "ci_upper_le_0_02": None, "saving_build_gt_0": None, "multi_seed_stable": None, "conclusion": "inconclusive"}
    final_stage = ledger.data["stages"]["final_evaluation"]
    if final_stage.get("status") == "PASS":
        comparisons = final_stage["comparisons"][str(acceptance["dev_b_star"])]
        primary = comparisons["7"]
        full, rank = primary["full_memory"], primary["rank_pack"]
        stable = all(v["full_memory"]["one_sided_95_upper"] <= 0.02 and v["full_memory"]["saving_build"] > 0 and v["rank_pack"]["quality_loss"] <= 0 for v in comparisons.values())
        acceptance.update(current_real_chain="PASS", quality_gap_our_full=-full["quality_loss"], saving_build=full["saving_build"],
                          quality_gap_our_rank_pack=-rank["quality_loss"], build_token_gap_our_rank_pack=rank["our_input_tokens"]-rank["baseline_input_tokens"],
                          one_sided_95_ci_upper=full["one_sided_95_upper"], ci_upper_le_0_02=full["one_sided_95_upper"] <= 0.02,
                          saving_build_gt_0=full["saving_build"] > 0, multi_seed_stable=stable, conclusion="work" if stable else "not work")
    ledger.data["acceptance"] = acceptance
    for failure in ledger.data["failures"]:
        if failure["stage"] == "inventory" and ledger.data["stages"].get("inventory", {}).get("status") == "PASS":
            failure.update(resolved=True, resolution="Scanner now classifies historical annotation_error records; successful inventory preserved and resumed without model work.")
    resource_summary(ledger)
    ledger.data["status"] = "ANNOTATION_TARGET_COMPLETE" if scope.get("stop_after_annotation_target") and annotations == expected else "BLOCKED"
    ledger.save("acceptance")
    r = ledger.data["resources"]
    report = ["ARC runbook 实际执行与验收", "", f"更新时间：{now()}。状态：{ledger.data['status']}；结论：{acceptance['conclusion']}。", "", f"当前 Grok 标注目标：{expected} 条（已完成 {annotations} 条，计入既有有效结果）；完整 runbook 需求仍为 {full_expected} 条。", "", "教师：Grok-4.6，读取实际 Codex URL/key。实验 builder、agent、auditor：SiliconFlow Qwen3.5-4B。", "", "11 条误跑的 Qwen 教师标注保留在 runs/locomo_arc/quarantine/requirements.wrong-teacher-qwen.jsonl，不计入正式训练或校准；相应调用消耗计入资源记录。", "", "| 阶段 | 状态 | 核验结果 |", "|---|---|---|"]
    for stage, record in ledger.data["stages"].items():
        description = json.dumps({k: v for k, v in record.items() if k not in ("status", "packages", "comparisons")}, ensure_ascii=False)
        report.append(f"| {stage} | {record.get('status')} | {description} |")
    report += ["", "实际执行：完整缓存与产物扫描、本地凭证/依赖检查、缺失原始输入增量生成、正式 train/dev 试跑、断点恢复修复及本地回归测试。详情以状态文件和调用日志为准。", "", "跳过：已有数据准备、两套检索索引/recall、3 条 final Grok 标注、3 条有效 final 编译、已有 smoke checkpoint 与真实 smoke 评估。旧数据未替换为新测量。", "", "失败历史：3 个历史调用错误、2 份 annotation_error 编译产物；扫描器遇到旧失败 schema 的问题已修复。未因旧失败重复执行已由有效产物覆盖的 smoke。", "", "阻塞项："]
    for key, blocker in ledger.data["blockers"].items():
        report.append(f"- {key}: {blocker['reason']}")
    report += ["", "Grok 失败调用记录（包括已恢复的历史尝试）：", "", "```json", json.dumps(ledger.data.get("teacher_call_failures", []), ensure_ascii=False, indent=2), "```"]
    if ledger.data.get("scheduled_resume", {}).get("enabled"):
        report += ["", f"后台将在 {ledger.data['scheduled_resume']['at']} 自动续跑；等待期间不发送 API 请求。"]
    report += ["", "资源记录：", "", "```json", json.dumps(r, ensure_ascii=False, indent=2), "```", "", "验收：", "", "```json", json.dumps(ledger.data["acceptance"], ensure_ascii=False, indent=2), "```", "", "续跑：python scripts/run_arc_pipeline.py --workers 2 --wait-for-quota-reset。按固定名单累计达到 100 条后停止；已有成功调用按完整任务键和请求哈希复用，服务端结果未知的请求须先核对日志。编译、训练、Dev 与 Final 不在本轮范围内。", ""]
    Path("runs/arc_run_report.md").write_text("\n".join(report), encoding="utf-8")


def work_key(ledger, stage, qa_id, split, arm=None, seed=None, budget=None):
    return {"stage": stage, "qa_id": qa_id, "split": split, "arm": arm,
            "seed": seed, "budget": budget, "config_sha256": ledger.data["config_sha256"]}


def task_record(ledger, key, status, **details):
    ledger.data["tasks"][digest(key)] = {"key": key, "status": status, "updated_at": now(), **details}


def append_record(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
    with path.open("ab") as handle:
        offset = handle.tell()
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return {"path": str(path), "offset": offset, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def index_records(path):
    result = {}
    path = Path(path)
    if not path.exists():
        return result
    with path.open("rb") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            try:
                row = json.loads(line)
            except ValueError:
                # Preserve a torn append before repairing the journal. Complete
                # records are never discarded, and middle corruption blocks.
                if handle.read(1):
                    raise ValueError(f"corrupt record within {path} at {offset}")
                tail = path.with_name(path.name + f".interrupted-{offset}.bin")
                if not tail.exists():
                    tail.write_bytes(line)
                with path.open("r+b") as repair:
                    repair.truncate(offset)
                break
            qid = row["qa_id"]
            if qid in result:
                raise ValueError(f"duplicate QA ID in canonical output: {path} {qid}")
            if not line.endswith(b"\n"):
                with path.open("ab") as repair:
                    repair.write(b"\n")
                line += b"\n"
            result[qid] = {"path": str(path), "offset": offset, "bytes": len(line), "sha256": hashlib.sha256(line).hexdigest(), "provenance": row.get("provenance")}
    return result


def read_record(reference):
    with Path(reference["path"]).open("rb") as handle:
        handle.seek(reference["offset"])
        line = handle.readline()
    assert hashlib.sha256(line).hexdigest() == reference["sha256"]
    return json.loads(line)


def verify_annotation_artifacts(ledger, inputs=None, full_index=None, done=None):
    """Verify canonical rows against frozen inputs and saved successful calls."""
    from annotate_with_grok import _validate, HARNESS
    from arc.agent.config import load_config, public_config
    config = load_config("configs/locomo.yaml")
    assert digest(public_config(config)) == ledger.data["config_sha256"], "configuration changed since inventory"
    inputs = index_records("data/processed/requirements.raw.jsonl") if inputs is None else inputs
    full_index = index_records("runs/locomo_arc/requirements.full.jsonl") if full_index is None else full_index
    done = index_records("data/processed/requirements.train-dev.jsonl") if done is None else done
    scope = ledger.data.get("annotation_scope") or {}
    if scope:
        assert set(done) <= set(scope["qa_ids"]), "existing annotations outside active target"
    qas = {r["qa_id"]: r for _, r in rows("data/processed/locomo_qa.jsonl")}
    splits = {sample: split for split, samples in config["splits"].items() for sample in samples}
    excluded = {x["qa_id"] for x in ledger.data.get("decisions", {}).get("evidence", {}).get("excluded", [])}
    for qid in set(full_index) | set(done):
        assert qid in inputs and qid not in excluded
        row = read_record(inputs[qid])
        split = splits[qas[qid]["sample_id"]]
        assert split in ("train", "dev") and qas[qid]["category"] in (1, 2, 3, 4)
        assert row["split"] == split and row["question"] == qas[qid]["question"]
        assert row["provenance"]["config_sha256"] == ledger.data["config_sha256"]
        assert row["provenance"]["retrieval_sha256"] == ledger.data["artifacts"]["data/cache/retrieval/locomo_topk.jsonl"]["sha256"]
        assert 1 <= len(row["sources"]) <= 64 and 0 < row["complete_input_tokens"] <= 15872
        assert len({s["source_id"] for s in row["sources"]}) == len(row["sources"])
        full = read_record(full_index[qid])
        full_key = work_key(ledger, "full_for_teacher", qid, split, arm="full_memory", seed=7, budget=15872)
        assert full["key"] == full_key and full["input_sha256"] == inputs[qid]["sha256"]
        assert full["full_raw"].strip() and full["full_status"] in ("PASS", "FAIL", "UNKNOWN")
        assert full["full_usage"]
        for usage in full["full_usage"]:
            audit = json.loads(Path(usage["audit_path"]).read_text(encoding="utf-8"))
            assert usage["model"] == config["models"]["builder"] == audit["model"]
            assert audit["status"] == "ok" and audit["work_key"] == full_key
        task_record(ledger, full_key, "VERIFIED_SKIP", artifact=full_index[qid])
        if qid not in done:
            continue
        annotation = read_record(done[qid])
        _validate(annotation)
        assert all(annotation[k] == v for k, v in row.items()), f"annotation input mismatch: {qid}"
        assert annotation.get("teacher_model", "grok-4.6") == "grok-4.6"
        audit = json.loads(Path(annotation["teacher_audit_path"]).read_text(encoding="utf-8"))
        key = work_key(ledger, "annotation", qid, split, seed=7, budget=15872)
        assert audit["status"] == "ok" and audit["model"] == "grok-4.6" and audit["work_key"] == key
        assert audit["parsed"]["requirements"] == annotation["requirements"]
        prompt = json.loads(audit["payload"]["input"][0]["content"][0]["text"])
        expected_sources = [{"id": i, "source_id": s.get("source_id"), "time": s.get("session_datetime"),
                             "speaker": s.get("speaker"), "text": s.get("text")} for i, s in enumerate(row["sources"], 1)]
        assert prompt == {"task": HARNESS, "question": row["question"], "sources": expected_sources, "full_output": full["full_raw"]}
        task_record(ledger, key, "VERIFIED_SKIP", artifact=done[qid])
    ledger.data["annotation_verification"] = {"status": "PASS", "at": now(), "annotations": len(done),
                                               "full_outputs": len(full_index), "raw_rows": len(inputs)}
    ledger.data.setdefault("annotation_batch", {"started_at": now(), "initial_completed": len(done),
                                              "initial_full_outputs": len(full_index), "target": scope.get("total_target")})
    ledger.save()
    return inputs, full_index, done


_INPUT_WORKER = {}


def init_input_worker():
    from arc.agent.config import load_config
    from arc.agent.data.retrieval import load_retrieval, blocks_by_source
    config = load_config("configs/locomo.yaml")
    _INPUT_WORKER.update(config=config, cache=load_retrieval(config), lookup=blocks_by_source([r for _, r in rows("data/processed/locomo_blocks.jsonl")]))


def make_input(item):
    from arc.agent.data.requirements import _compiler_row
    from arc.algorithm.memory import configured_input_cost
    row = _compiler_row(_INPUT_WORKER["config"], _INPUT_WORKER["cache"][item], _INPUT_WORKER["lookup"])
    cost = configured_input_cost(_INPUT_WORKER["config"], row["question"], row["sources"])
    return row, cost


def input_results(keys, workers):
    if workers == 1:
        init_input_worker()
        for key in keys:
            yield key, make_input(key)
        return
    from concurrent.futures import ProcessPoolExecutor
    from collections import deque
    with ProcessPoolExecutor(max_workers=workers, initializer=init_input_worker) as pool:
        pending = deque()
        iterator = iter(keys)
        for _ in range(workers):
            key = next(iterator, None)
            if key is not None:
                pending.append((key, pool.submit(make_input, key)))
        while pending:
            key, future = pending.popleft()
            yield key, future.result()
            key = next(iterator, None)
            if key is not None:
                pending.append((key, pool.submit(make_input, key)))


def prepare_inputs(ledger, limit=None, workers=1):
    from arc.agent.config import load_config
    from arc.agent.data.retrieval import load_retrieval, blocks_by_source
    from arc.agent.data.requirements import _compiler_row
    from arc.algorithm.memory import configured_input_cost
    assert ledger.data["stages"].get("inventory", {}).get("status") == "PASS", "inventory must pass first"
    config = load_config("configs/locomo.yaml")
    lookup = blocks_by_source([r for _, r in rows("data/processed/locomo_blocks.jsonl")])
    qas = {r["qa_id"]: r for _, r in rows("data/processed/locomo_qa.jsonl")}
    desired_ids = set(ledger.data.get("annotation_scope", {}).get("qa_ids") or qas)
    assert desired_ids <= set(qas)
    splits = {sample: split for split, samples in config["splits"].items() for sample in samples}
    provenance = {"config_sha256": ledger.data["config_sha256"], "retrieval_sha256": ledger.data["artifacts"]["data/cache/retrieval/locomo_topk.jsonl"]["sha256"]}
    path = Path("data/processed/requirements.raw.jsonl")
    index = index_records(path)
    for qid, ref in index.items():
        if ref["provenance"] != provenance:
            ledger.block("raw_input_conflict", "Existing raw input has different or unverified provenance; preserved for reconciliation.", qa_id=qid, path=str(path))
            return
    if desired_ids <= set(index):
        ledger.data["stages"]["raw_inputs"] = {"status": "PASS", "expected": len(desired_ids),
            "complete": len(desired_ids), "total_preserved_rows": len(index), "added_this_invocation": 0,
            "path": str(path)}
        ledger.event("raw_inputs_verified_skip", skipped=len(desired_ids), total_preserved_rows=len(index))
        return
    ledger.data["status"] = "RUNNING"
    ledger.save("raw_inputs")
    cache = load_retrieval(config)
    added = 0
    pending = []
    for sample, qid in cache:
        if qid not in desired_ids:
            continue
        key = work_key(ledger, "raw_inputs", qid, splits[sample], budget=15872)
        if qid in index:
            task_record(ledger, key, "VERIFIED_SKIP", artifact=index[qid])
            continue
        pending.append((sample, qid))
    if limit is not None:
        pending = pending[:limit]
    for (sample, qid), (row, cost) in input_results(pending, workers):
        key = work_key(ledger, "raw_inputs", qid, splits[sample], budget=15872)
        assert row["question"] == qas[qid]["question"]
        assert 1 <= len(row["sources"]) <= 64
        assert len({s["source_id"] for s in row["sources"]}) == len(row["sources"])
        assert cost <= 15872
        row.update(split=splits[sample], category=qas[qid]["category"],
                   answer=qas[qid].get("answer"),
                   future_queries=[{"question": qas[qid].get("question", ""),
                                    "answer": qas[qid].get("answer"),
                                    "category": qas[qid].get("category")}],
                   provenance=provenance, complete_input_tokens=cost)
        reference = append_record(path, row)
        index[qid] = reference
        added += 1
        task_record(ledger, key, "COMPLETE", artifact=reference, source_count=len(row["sources"]), complete_input_tokens=cost)
        ledger.data["stages"]["raw_inputs"] = {"status": "RUNNING", "expected": len(desired_ids), "complete": len(desired_ids & set(index)), "total_preserved_rows": len(index), "added_this_invocation": added, "path": str(path)}
        ledger.save()
        if added % 10 == 0:
            print(f"raw inputs for active scope: {len(desired_ids & set(index))}/{len(desired_ids)} (+{added}; {len(index)} total retained)", flush=True)
        if limit is not None and added >= limit:
            ledger.data["stages"]["raw_inputs"].update(status="PARTIAL")
            ledger.save()
            return
    assert desired_ids <= set(index)
    ledger.data["stages"].setdefault("raw_inputs", {}).update(status="PASS", complete=len(desired_ids), expected=len(desired_ids), total_preserved_rows=len(index), path=str(path))
    ledger.event("raw_inputs_complete", added=added, skipped=len(desired_ids)-added, total_preserved_rows=len(index))


def annotate(ledger, limit=None, workers=1):
    import copy
    from arc.agent.config import load_config
    from arc.algorithm.memory import build_once
    from annotate_with_grok import _credential, _call_teacher, _validate
    assert ledger.data["stages"].get("raw_inputs", {}).get("status") in ("PASS", "PARTIAL"), "raw inputs must be checked first"
    config = load_config("configs/locomo.yaml")
    config["execution"] = {"resume_calls": True}
    inputs = index_records("data/processed/requirements.raw.jsonl")
    full_index = index_records("runs/locomo_arc/requirements.full.jsonl")
    annotation_path = "data/processed/requirements.train-dev.jsonl"
    done = index_records(annotation_path)
    verify_annotation_artifacts(ledger, inputs, full_index, done)
    issues = ledger.data["blockers"].get("evidence_decision", {}).get("items", [])
    blocked_ids = {f'{x["sample_id"]}:qa:{x["qa_index"]}' for x in issues}
    selected = [r for _, r in rows("data/processed/locomo_qa.jsonl") if r["sample_id"] in config["splits"]["train"] + config["splits"]["dev"] and r["category"] in (1, 2, 3, 4)]
    assert len(selected) == 1226
    excluded = {x["qa_id"] for x in ledger.data.get("decisions", {}).get("evidence", {}).get("excluded", [])}
    selected = [qa for qa in selected if qa["qa_id"] not in excluded]
    full_expected = len(selected)
    scope = ledger.data.get("annotation_scope") or {}
    if scope:
        wanted = set(scope["qa_ids"])
        assert wanted <= {qa["qa_id"] for qa in selected}
        selected = [qa for qa in selected if qa["qa_id"] in wanted]
    expected = len(selected)
    if set(done) - {qa["qa_id"] for qa in selected}:
        raise ValueError("Existing valid annotations fall outside the configured subset; preserve them and reconcile the target")
    teacher = ledger.data.get("decisions", {}).get("teacher", {}).get("model", "grok-4.6")
    if teacher != "grok-4.6":
        raise ValueError("Formal requirements must use Grok-4.6 via the active Codex URL/key; Qwen is the experiment model")
    if len(done) == expected:
        ledger.data["stages"]["annotations"] = {"status": "TARGET_COMPLETE" if expected < full_expected else "PASS",
            "complete": len(done), "expected": expected, "missing": 0, "full_runbook_requirement": full_expected, "teacher": teacher}
        ledger.save()
        return
    try:
        base_url, api_key = _credential(Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))))
    except (ValueError, OSError) as exc:
        ledger.block("teacher_credentials", str(exc))
        return
    ledger.data["status"] = "RUNNING"
    ledger.save("annotations")
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def run_annotation_job(job):
        """Run one independent Full(builder) -> Grok pair.

        Workers write only provider audit files. Canonical JSONL and the ledger
        are updated by the main thread after each future completes.
        """
        qid, row, key, full_key, full = job
        local_config = copy.deepcopy(config)
        if full is None:
            local_config["execution"] = {"resume_calls": True, "work_key": full_key}
            result = build_once(local_config, None, row["sources"], sample_id=row["sample_id"], qa_id=qid)
            full = {"qa_id": qid, "sample_id": row["sample_id"], "split": row["split"],
                    "full_raw": result.raw, "full_status": result.status, "full_memories": list(result.memories),
                    "full_usage": list(result.usage), "input_sha256": inputs[qid]["sha256"], "key": full_key}
            if result.error or not result.raw:
                return {"qid": qid, "row": row, "key": key, "full_key": full_key, "full": full,
                        "error_stage": "builder", "error": result.error or "Builder returned no Full output"}
        try:
            annotation = _call_teacher(base_url, api_key, row["question"], row["sources"], full["full_raw"],
                                       audit_dir=Path("runs/locomo_arc/teacher_calls"), work_key=key)
            output = {**row, "requirements": annotation["requirements"], "teacher_model": teacher,
                      "teacher_usage": annotation.get("_usage"), "teacher_audit_path": annotation.get("_audit_path"),
                      "syntax_repair": annotation.get("_syntax_repair")}
            _validate(output)
            return {"qid": qid, "row": row, "key": key, "full_key": full_key, "full": full,
                    "output": output}
        except Exception as exc:
            return {"qid": qid, "row": row, "key": key, "full_key": full_key, "full": full,
                    "error_stage": "teacher", "error": str(exc)}

    jobs = []
    for qa in selected:
        if len(done) + len(jobs) >= expected:
            break
        qid = qa["qa_id"]
        if qid not in inputs:
            continue
        row = read_record(inputs[qid])
        key = work_key(ledger, "annotation", qid, row["split"], seed=7, budget=15872)
        if qid in done:
            existing = read_record(done[qid])
            _validate(existing)
            if existing.get("teacher_model", "grok-4.6") != teacher:
                raise ValueError(f"Wrong teacher model in formal annotations: {qid}; preserve and quarantine before resuming")
            task_record(ledger, key, "VERIFIED_SKIP", artifact=done[qid])
            continue
        if qid in blocked_ids:
            task_record(ledger, key, "BLOCKED", reason="evidence_decision")
            continue
        full_key = work_key(ledger, "full_for_teacher", qid, row["split"], arm="full_memory", seed=7, budget=15872)
        full = read_record(full_index[qid]) if qid in full_index else None
        task_record(ledger, full_key, "VERIFIED_SKIP" if full is not None else "RUNNING",
                    artifact=full_index.get(qid))
        task_record(ledger, key, "RUNNING")
        jobs.append((qid, row, key, full_key, full))
    ledger.save()

    added = 0
    max_workers = max(1, min(int(workers), 4))
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="arc-annotate") as pool:
        futures = [pool.submit(run_annotation_job, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            qid, key, full_key, full = result["qid"], result["key"], result["full_key"], result["full"]
            if qid not in full_index:
                if result.get("error_stage") == "builder" and (not full.get("full_raw") or not full.get("full_usage")):
                    log = append_record("runs/locomo_arc/requirements.full.failures.jsonl", full)
                    task_record(ledger, full_key, "BLOCKED", error=result["error"], artifact=log)
                else:
                    full_index[qid] = append_record("runs/locomo_arc/requirements.full.jsonl", full)
                    task_record(ledger, full_key, "COMPLETE", artifact=full_index[qid])
            if result.get("error_stage"):
                task_record(ledger, key, "BLOCKED", error=result["error"])
                blocker = "builder_service" if result["error_stage"] == "builder" else "teacher_service"
                ledger.block(blocker, result["error"], qa_id=qid, teacher_model=teacher)
                continue
            done[qid] = append_record(annotation_path, result["output"])
            previous = ledger.data["blockers"].get("teacher_service")
            if previous and previous.get("qa_id") == qid:
                ledger.data.setdefault("resolved_blockers", {})["teacher_format:" + qid] = {**previous, "status": "RESOLVED", "resolution": result["output"].get("syntax_repair") or "valid response recovered"}
                ledger.data["blockers"].pop("teacher_service")
            added += 1
            task_record(ledger, key, "COMPLETE", artifact=done[qid])
            ledger.data["stages"]["annotations"] = {"status": "RUNNING", "complete": len(done), "expected": expected, "pilot_target": 20, "teacher": teacher, "workers": max_workers}
            resource_summary(ledger)
            ledger.save()
            print(f"teacher annotations: {len(done)}/{expected} (workers={max_workers})", flush=True)
            if len(done) == 20:
                ledger.event("annotation_pilot_pass", validated_rows=20, next_action="continue same configuration")
    status = "TARGET_COMPLETE" if len(done) == expected and expected < full_expected else "PASS" if len(done) == expected else "PARTIAL"
    ledger.data["stages"]["annotations"] = {"status": status, "complete": len(done), "expected": expected, "full_runbook_requirement": full_expected, "teacher": teacher}
    if len(done) == expected:
        ledger.data["scheduled_resume"] = {"enabled": False, "reason": "annotation target complete"}
    resource_summary(ledger)


def compile_annotations(ledger, limit=None):
    from arc.agent.config import load_config
    from arc.algorithm.compiler import compile_task
    from arc.algorithm.memory import BuildResult, AuditResult
    from annotate_with_grok import _validate
    config = load_config("configs/locomo.yaml")
    inputs = index_records("data/processed/requirements.train-dev.jsonl")
    full_index = index_records("runs/locomo_arc/requirements.full.jsonl")
    target = "runs/locomo_arc/compilation.train-dev.jsonl"
    done = index_records(target)
    excluded = {x["qa_id"] for x in ledger.data.get("decisions", {}).get("evidence", {}).get("excluded", [])}
    expected = ledger.data.get("effective_counts", {}).get("train_dev_1_4", 1226)
    added = 0
    ledger.save("compilation")
    for qid, reference in inputs.items():
        if qid in excluded:
            continue
        row = read_record(reference)
        _validate(row)
        if row.get("teacher_model", "grok-4.6") != "grok-4.6":
            raise ValueError(f"Compiler refuses wrong-teacher annotation: {qid}")
        key = work_key(ledger, "compilation", qid, row["split"], seed=7, budget=15872)
        if qid in done:
            completed = read_record(done[qid])
            assert completed["annotation"]["valid"] and completed["domain_complete"]
            assert completed["input_sha256"] == reference["sha256"]
            task_record(ledger, key, "VERIFIED_SKIP", artifact=done[qid])
            continue
        full = read_record(full_index[qid])
        original_full = BuildResult(frozenset(range(1, len(row["sources"]) + 1)), full["full_raw"], tuple(full["full_memories"]), AuditResult("PASS", ("audit_skipped",)), AuditResult("PASS", ("audit_skipped",)), full["full_status"], tuple(full["full_usage"]))
        config["execution"] = {"resume_calls": True, "work_key": key}
        task_record(ledger, key, "RUNNING")
        ledger.save()
        try:
            result = compile_task(config, row, limit=16, d=8, full_result=original_full)
            assert result["annotation"]["valid"] and result["domain_complete"]
            errors = [u for e in result["evaluations"] for u in e.get("usage", []) if u.get("status") == "error"]
            if errors:
                failure_path = f"runs/locomo_arc/compilation.failed/{qid.replace(':', '_')}.json"
                atomic_json(failure_path, result)
                ledger.block("compiler_service", "Compilation contains unsuccessful API calls; successful candidate calls are cached for resume.", qa_id=qid, errors=errors, partial_result=failure_path)
                task_record(ledger, key, "BLOCKED", artifact=failure_path)
                resource_summary(ledger)
                return
        except Exception as exc:
            task_record(ledger, key, "BLOCKED", error=str(exc))
            ledger.block("compiler_service", str(exc), qa_id=qid)
            resource_summary(ledger)
            return
        result.update(split=row["split"], category=row["category"], input_sha256=reference["sha256"], provenance={"key": key, "teacher_model": row.get("teacher_model", "grok-4.6"), "limit": 16, "d": 8})
        done[qid] = append_record(target, result)
        task_record(ledger, key, "COMPLETE", artifact=done[qid],
                    successful_solutions=len(result.get("successful_solutions") or result.get("successful_sets") or []),
                    evaluation_statuses=dict(collections.Counter(e["status"] for e in result["evaluations"])))
        added += 1
        ledger.data["stages"]["compilation"] = {"status": "RUNNING", "complete": len(done), "expected": expected}
        ledger.save()
        print(f"compiled: {len(done)}/{expected}", flush=True)
        if limit is not None and added >= limit:
            break
    ledger.data["stages"]["compilation"] = {"status": "PASS" if len(done) == expected else "PARTIAL", "complete": len(done), "expected": expected}
    resource_summary(ledger)


def train(ledger):
    from arc.agent.config import load_config
    from arc.agent.io import write_jsonl
    from arc.algorithm.selector import train_selector_file, load_selector
    config = load_config("configs/locomo.yaml")
    expected = ledger.data.get("effective_counts", {}).get("train_1_4", 885)
    source = Path("runs/locomo_arc/compilation.train-dev.jsonl")
    if not source.exists():
        ledger.block("training_dependency", "No formal training compilation rows exist")
        return
    qa_ids = [r["qa_id"] for _, r in rows(source) if r.get("split") == "train"]
    if len(set(qa_ids)) != expected or len(qa_ids) != expected:
        ledger.block("training_dependency", "Formal training compilation is incomplete", expected=expected, complete=len(set(qa_ids)))
        return
    path = Path("runs/locomo_arc/compilation.train.jsonl")
    if not path.exists():
        write_jsonl(path, (r for _, r in rows(source) if r.get("split") == "train"))
    assert [r["qa_id"] for _, r in rows(path)] == qa_ids
    ledger.data.setdefault("protocol", {"selector_seeds": [7, 17, 27], "budgets": [1024, 2048, 4096], "epochs": 30, "primary_seed": 7, "compile_limit": 16, "compile_d": 8})
    ledger.save("selector_training")
    for seed in ledger.data["protocol"]["selector_seeds"]:
        config["seed"] = seed
        key = work_key(ledger, "selector_training", "__all_train__", "train", seed=seed)
        target = Path(f"models/arc_selector.seed-{seed}.pt")
        def progress(**detail):
            task_record(ledger, key, "RUNNING", progress=detail, checkpoint=str(target))
            ledger.save()
        result = train_selector_file(path, target, epochs=30, config=config, resume=True, progress_callback=progress)
        load_selector(target)
        task_record(ledger, key, "VERIFIED_SKIP" if result.get("skipped") else "COMPLETE", result=result)
        ledger.data["resources"]["new_training_runs"] = ledger.data["resources"].get("new_training_runs", 0) + (not result.get("skipped"))
        ledger.save()
    # Canonical main-seed checkpoint; preserve any unrelated prior target.
    import shutil
    primary = Path("models/arc_selector.pt")
    source_checkpoint = Path("models/arc_selector.seed-7.pt")
    if not primary.exists():
        shutil.copyfile(source_checkpoint, primary)
    elif primary.read_bytes() != source_checkpoint.read_bytes():
        ledger.block("checkpoint_conflict", "models/arc_selector.pt already contains a different checkpoint; preserved")
        return
    config_path = Path("configs/locomo.yaml")
    text = config_path.read_text(encoding="utf-8")
    if '  checkpoint: ""' in text:
        config_path.write_text(text.replace('  checkpoint: ""', '  checkpoint: models/arc_selector.pt'), encoding="utf-8")
    ledger.data["stages"]["selector_training"] = {"status": "PASS", "complete": 3, "epochs": 30, "seeds": [7, 17, 27]}
    ledger.save()


def paired_summary(ours, baseline, *, seed=7, samples=10000):
    import numpy as np
    a = {r["qa_id"]: r for r in ours if r["category"] in (1, 2, 3, 4)}
    b = {r["qa_id"]: r for r in baseline if r["category"] in (1, 2, 3, 4)}
    if set(a) != set(b) or not a:
        raise ValueError("paired bootstrap requires equal, nonempty QA key sets")
    ids = sorted(a)
    gaps = np.array([b[q]["score"] - a[q]["score"] for q in ids])
    rng = np.random.default_rng(seed)
    means = np.concatenate([gaps[rng.integers(0, len(ids), (min(1000, samples-i), len(ids)))].mean(axis=1) for i in range(0, samples, 1000)])
    our_tokens = sum(a[q]["construction_prompt_tokens"] for q in ids)
    base_tokens = sum(b[q]["construction_prompt_tokens"] for q in ids)
    return {"n": len(ids), "quality_loss": float(gaps.mean()), "one_sided_95_upper": float(np.quantile(means, 0.95)),
            "saving_build": 1-our_tokens/base_tokens if base_tokens else None, "our_input_tokens": our_tokens,
            "baseline_input_tokens": base_tokens, "bootstrap": {"unit": "paired_qa", "samples": samples, "seed": seed, "scope": "categories_1_4"}}


def evaluate_stage(ledger, split):
    import copy
    from arc.agent.config import load_config
    from arc.baseline.evaluate import run_eval, _completed
    if ledger.data["stages"].get("selector_training", {}).get("status") != "PASS":
        ledger.block(split + "_dependency", "Formal multi-seed selector training must finish first")
        return
    config = load_config("configs/locomo.yaml")
    excluded = {x["qa_id"] for x in ledger.data.get("decisions", {}).get("evidence", {}).get("excluded", [])}
    qas = [r for _, r in rows("data/processed/locomo_qa.jsonl") if r["sample_id"] in config["splits"][split] and r["qa_id"] not in excluded]
    if split == "dev":
        qas = [r for r in qas if r["category"] in (1, 2, 3, 4)]
        budgets = [1024, 2048, 4096]
    else:
        chosen = ledger.data["stages"].get("dev_budget", {}).get("b_star")
        if chosen is None:
            ledger.block("final_dependency", "No Dev budget satisfies the prespecified quality and savings criteria")
            return
        budgets = [chosen]
    ledger.save(split + "_evaluation")
    def run(arm, budget, seed=7):
        settings = copy.deepcopy(config)
        settings["seed"] = seed
        settings["budgets"]["builder_input_tokens"] = budget
        settings["selector"]["checkpoint"] = f"models/arc_selector.seed-{seed}.pt"
        path = Path(f"runs/locomo_arc/{split}/seed-{seed}/budget-{budget}/{arm}.jsonl")
        def progress(row):
            key = work_key(ledger, "evaluation", row["qa_id"], split, arm=arm, seed=seed, budget=budget)
            task_record(ledger, key, "COMPLETE" if _completed(row) else "FAILED", path=str(path))
            ledger.data["resources"]["new_evaluation_rows"] += 1
            ledger.save()
            if not _completed(row):
                raise RuntimeError(f"Incomplete evaluation: {row['qa_id']} {arm}; inspect saved costs and errors")
        run_eval(settings, [arm], split, output=str(path), resume=path.with_name(path.stem + ".manifest.json").exists(), qa_ids={q["qa_id"] for q in qas}, progress_callback=progress)
        result = [r for _, r in rows(path)]
        assert len(result) == len(qas) and all(_completed(r) for r in result)
        return result
    try:
        # A Full baseline does not depend on selector seed or deployment
        # budget, so one verified measurement is shared across these compares.
        full = run("full_memory", 4096 if split == "dev" else budgets[0])
        comparisons = {}
        for budget in budgets:
            rank = run("rank_pack", budget)
            seed_results = {}
            for seed in (7, 17, 27):
                ours = run("our", budget, seed)
                seed_results[str(seed)] = {"full_memory": paired_summary(ours, full), "rank_pack": paired_summary(ours, rank)}
            comparisons[str(budget)] = seed_results
        if split == "final":
            run("no_memory", budgets[0]); run("naive_rag", budgets[0])
    except Exception as exc:
        ledger.block(split + "_service", str(exc))
        resource_summary(ledger)
        return
    atomic_json(f"runs/locomo_arc/{split}/paired_comparisons.json", comparisons)
    if split == "dev":
        candidates = [budget for budget in budgets if all(x["full_memory"]["one_sided_95_upper"] <= 0.02 and x["full_memory"]["saving_build"] > 0 and x["rank_pack"]["quality_loss"] <= 0 for x in comparisons[str(budget)].values())]
        ledger.data["stages"]["dev_budget"] = {"status": "PASS" if candidates else "NO_FEASIBLE_BUDGET", "b_star": min(candidates) if candidates else None, "comparisons": comparisons}
    else:
        ledger.data["stages"]["final_evaluation"] = {"status": "PASS", "complete_primary": 2000, "additional_our_seed_rows": 800, "comparisons": comparisons}
    resource_summary(ledger)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("inventory", "resources", "inputs", "annotate", "compile", "train", "dev", "final", "assess"), default="inventory")
    parser.add_argument("--limit", type=int, help="New input/annotation rows for a pilot; existing rows are skipped")
    parser.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=1,
                        help="Concurrent workers for input preparation and independent annotation calls")
    args = parser.parse_args()
    os.chdir(ROOT)
    # Serialize every ledger mutation and experiment invocation.
    import fcntl
    lock = Path("runs/.arc_runbook.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ledger = Ledger()
    try:
        if args.stage in ("inputs", "annotate", "compile"):
            if args.stage == "inputs":
                prepare_inputs(ledger, limit=args.limit, workers=args.workers)
            elif args.stage == "annotate":
                annotate(ledger, limit=args.limit, workers=args.workers)
            else:
                compile_annotations(ledger, limit=args.limit)
        else:
            {"inventory": inventory, "resources": resource_summary, "assess": assess, "train": train,
             "dev": lambda l: evaluate_stage(l, "dev"), "final": lambda l: evaluate_stage(l, "final")}[args.stage](ledger)
    except BaseException as exc:
        ledger.data["status"] = "INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED"
        ledger.data["failures"].append({"at": now(), "stage": args.stage, "type": type(exc).__name__, "error": str(exc)[:1000]})
        ledger.save()
        raise
    ledger.invocation["finished_at"] = now()
    ledger.save()


if __name__ == "__main__":
    main()
