#!/usr/bin/env python3
"""Run the ARC runbook unattended, stopping with a durable report on blockers.

Start/resume with: python scripts/run_arc_pipeline.py --workers 2
Logs: runs/arc_pipeline.log and runs/arc_<stage>.log
State: runs/arc_run_state.json; process heartbeat: runs/arc_pipeline_job.json
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

from run_arc_runbook import ROOT, Ledger, atomic_json, now, signature


STAGES = [("inputs", "raw_inputs"), ("annotate", "annotations"),
          ("compile", "compilation"), ("train", "selector_training"),
          ("dev", "dev_budget"), ("final", "final_evaluation")]


def selected_stages(state):
    if state.get("annotation_scope", {}).get("stop_after_annotation_target"):
        return STAGES[:2]
    return STAGES


def latest_quota_block():
    failures = []
    for path in Path("runs/locomo_arc/teacher_calls").glob("*.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("http_status") != 402:
            continue
        try:
            error = json.loads(row.get("response", "{}"))["error"]
            reset = datetime.fromisoformat(error["reset_time"].replace("Z", "+00:00"))
        except (ValueError, KeyError, TypeError):
            continue
        if reset > datetime.now(timezone.utc):
            failures.append((row.get("started_at", ""), {"reset_time": error["reset_time"],
                "limit_type": error.get("limit_type"), "account_current_usd": error.get("current"),
                "account_limit_usd": error.get("limit"), "audit_path": str(path), "qa_id": (row.get("work_key") or {}).get("qa_id")}))
    return max(failures, key=lambda item: item[0])[1] if failures else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=2)
    parser.add_argument("--wait-for-quota-reset", action="store_true",
                        help="Persist BLOCKED and resume after the provider's reported reset, without polling the API")
    parser.add_argument("--force-teacher-retry", action="store_true",
                        help="Ignore historical quota errors and test the currently configured Codex URL/key")
    args = parser.parse_args()
    os.chdir(ROOT)
    Path("runs").mkdir(exist_ok=True)
    singleton = Path("runs/.arc_pipeline.lock").open("a")
    try:
        fcntl.flock(singleton, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("An ARC pipeline process is already active; no duplicate worker launched.", flush=True)
        singleton.close()
        return 0
    decisions = json.loads(Path("runs/arc_run_decisions.json").read_text())
    stages = selected_stages(decisions)
    job = {"schema": "arc_pipeline_job", "pid": os.getpid(), "started_at": now(),
           "status": "WAITING_FOR_ACTIVE_STAGE", "stages": [x[0] for x in STAGES], "commands": [],
           "teacher": "grok-4.6 via active Codex URL/key", "experiment_model": "Qwen/Qwen3.5-4B via SiliconFlow"}
    job.update(stages=[x[0] for x in stages], annotation_target=decisions.get("annotation_scope", {}).get("total_target"),
               wait_for_quota_reset=args.wait_for_quota_reset, force_teacher_retry=args.force_teacher_retry)
    child = None

    def heartbeat(**changes):
        job.update(changes, updated_at=now())
        atomic_json("runs/arc_pipeline_job.json", job)

    def interrupt(signum, _frame):
        heartbeat(status="INTERRUPTED", signal=signum)
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGINT)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    heartbeat()
    stage_lock = Path("runs/.arc_runbook.lock").open("a")
    while True:
        try:
            fcntl.flock(stage_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(stage_lock, fcntl.LOCK_UN)
            break
        except BlockingIOError:
            heartbeat()
            time.sleep(10)

    def run_stage(stage):
        nonlocal child
        command = [sys.executable, "-u", "scripts/run_arc_runbook.py", "--stage", stage]
        if stage in ("inputs", "annotate"):
            command += ["--workers", str(args.workers)]
        logfile = f"runs/arc_{stage}.log"
        entry = {"stage": stage, "command": command, "log": logfile, "started_at": now()}
        job["commands"].append(entry)
        heartbeat(status="RUNNING", stage=stage)
        print(f"{now()} starting {stage}; log={logfile}", flush=True)
        env = os.environ.copy()
        env.update(HF_HUB_OFFLINE="1", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        with Path(logfile).open("a", encoding="utf-8") as handle:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT)
            heartbeat(child_pid=child.pid)
            while child.poll() is None:
                time.sleep(10)
                heartbeat()
            entry.update(returncode=child.returncode, finished_at=now())
            child = None
        heartbeat(child_pid=None)
        return entry["returncode"]

    def wait_for_quota(quota):
        with Path("runs/.arc_runbook.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            ledger = Ledger()
            ledger.block("teacher_service", "Codex/Grok daily quota exceeded; no request repeated before the reported reset time.",
                         teacher_model="grok-4.6", **quota)
            ledger.data["stages"]["annotations"]["status"] = "BLOCKED"
            ledger.data["scheduled_resume"] = {"enabled": args.wait_for_quota_reset, "at": quota["reset_time"],
                                                "pipeline_pid": os.getpid(), "target": job["annotation_target"]}
            ledger.save()
        if run_stage("assess"):
            raise RuntimeError("could not write quota-blocked acceptance report")
        heartbeat(status="BLOCKED", stage="annotate", blocker="teacher_service", **quota)
        if not args.wait_for_quota_reset:
            return False
        print(f"{now()} quota blocked; automatic resume at {quota['reset_time']}", flush=True)
        reset = datetime.fromisoformat(quota["reset_time"].replace("Z", "+00:00"))
        while True:
            remaining = (reset - datetime.now(timezone.utc)).total_seconds() + 1
            if remaining <= 0:
                break
            time.sleep(min(30, remaining))
            heartbeat(status="BLOCKED", stage="annotate", scheduled_resume_at=quota["reset_time"])
        return True

    try:
        state = json.loads(Path("runs/arc_run_state.json").read_text())
        # Reuse the completed inventory; its artifact signatures are checked
        # below before any model work. A fresh inventory is only needed after a
        # prior inventory failed or was interrupted.
        if state["stages"].get("inventory", {}).get("status") != "PASS":
            if run_stage("inventory"):
                raise RuntimeError("inventory failed; inspect runs/arc_inventory.log")
            state = json.loads(Path("runs/arc_run_state.json").read_text())
        for path, artifact in state.get("artifacts", {}).items():
            if path.startswith("data/cache/") and artifact.get("signature") != signature(path):
                raise RuntimeError(f"Frozen retrieval cache changed after verification: {path}")
        for stage, state_name in stages:
            state = json.loads(Path("runs/arc_run_state.json").read_text())
            if stage == "inputs" and state["stages"].get(state_name, {}).get("status") == "PASS":
                heartbeat(status="STAGE_COMPLETE", stage=stage, skipped="verified_existing_inputs")
                continue
            # Offline input preparation is independent of teacher quota.
            if stage == "annotate":
                state = json.loads(Path("runs/arc_run_state.json").read_text())
                verified_count = state.get("annotation_verification", {}).get("annotations", 0)
                target_complete = job["annotation_target"] is not None and verified_count == job["annotation_target"]
                quota = None if target_complete or args.force_teacher_retry else latest_quota_block()
                if quota and not wait_for_quota(quota):
                    break
            while True:
                if run_stage(stage):
                    heartbeat(status="FAILED", stage=stage, reason=f"{stage} exited nonzero")
                    break
                state = json.loads(Path("runs/arc_run_state.json").read_text())
                if state["stages"].get(state_name, {}).get("status") in ("PASS", "TARGET_COMPLETE"):
                    heartbeat(status="STAGE_COMPLETE", stage=stage)
                    break
                quota = latest_quota_block() if stage == "annotate" and not args.force_teacher_retry else None
                if quota and wait_for_quota(quota):
                    continue
                heartbeat(status="BLOCKED", stage=stage, blockers=state.get("blockers", {}))
                break
            if job["status"] in ("BLOCKED", "FAILED"):
                break
        terminal_status = job["status"] if job["status"] in ("BLOCKED", "FAILED") else "CORE_COMPLETE"
        run_stage("assess")
        state = json.loads(Path("runs/arc_run_state.json").read_text())
        heartbeat(status=terminal_status if terminal_status != "CORE_COMPLETE" else state["status"],
                  finished_at=now(), report="runs/arc_run_report.md", blockers=state.get("blockers", {}))
    except KeyboardInterrupt:
        heartbeat(status="INTERRUPTED", finished_at=now())
        return 130
    except Exception as exc:
        heartbeat(status="FAILED", error=str(exc), finished_at=now())
        raise
    finally:
        stage_lock.close()
        singleton.close()
    print(f"{now()} {job['status']}; report=runs/arc_run_report.md", flush=True)
    return 0 if job["status"] in ("BLOCKED", "COMPLETE", "CORE_COMPLETE", "ANNOTATION_TARGET_COMPLETE") else 1


if __name__ == "__main__":
    raise SystemExit(main())
