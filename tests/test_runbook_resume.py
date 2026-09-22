"""Local regression tests; no model requests or experiment measurements."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from arc.agent.reader import ClaudeCodeClient, CompletionError
from arc.baseline.evaluate import _merge_journal, construction_usage
from run_arc_runbook import append_record, index_records, read_record, paired_summary
from annotate_with_grok import _parse_json, _credential


class ResumeTests(unittest.TestCase):
    def test_capped_pipeline_stops_before_compilation_training_and_evaluation(self):
        import run_arc_pipeline
        previous = Path.cwd()
        commands = []
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "runs").mkdir()
            scope = {"total_target": 100, "stop_after_annotation_target": True}
            (root / "runs/arc_run_decisions.json").write_text(json.dumps({"annotation_scope": scope}))
            state = {"stages": {}, "artifacts": {}, "blockers": {}}
            state_path = root / "runs/arc_run_state.json"
            state_path.write_text(json.dumps(state))

            class StageProcess:
                pid = 123
                returncode = 0

                def __init__(self, command, **kwargs):
                    stage = command[command.index("--stage") + 1]
                    commands.append(stage)
                    if stage == "inventory":
                        state["stages"]["inventory"] = {"status": "PASS"}
                    elif stage == "inputs":
                        state["stages"]["raw_inputs"] = {"status": "PASS"}
                    elif stage == "annotate":
                        state["stages"]["annotations"] = {"status": "TARGET_COMPLETE", "complete": 100}
                    elif stage == "assess":
                        state["status"] = "ANNOTATION_TARGET_COMPLETE"
                    else:
                        raise AssertionError(f"unauthorized stage after annotation cap: {stage}")
                    state_path.write_text(json.dumps(state))

                def poll(self):
                    return self.returncode

            import signal
            handlers = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
            try:
                with patch.object(run_arc_pipeline, "ROOT", root), patch.object(sys, "argv", ["pipeline"]), \
                     patch.object(run_arc_pipeline.subprocess, "Popen", StageProcess):
                    self.assertEqual(run_arc_pipeline.main(), 0)
                self.assertEqual(commands, ["inventory", "inputs", "annotate", "assess"])
                job = json.loads((root / "runs/arc_pipeline_job.json").read_text())
                self.assertEqual(job["status"], "ANNOTATION_TARGET_COMPLETE")
            finally:
                os.chdir(previous)
                for sig, handler in handlers.items():
                    signal.signal(sig, handler)

    def test_background_pipeline_recognizes_unexpired_quota(self):
        from run_arc_pipeline import latest_quota_block
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as folder:
            try:
                os.chdir(folder)
                logs = Path("runs/locomo_arc/teacher_calls")
                logs.mkdir(parents=True)
                log = {"http_status": 402, "started_at": "2026-09-19T00:00:00Z", "work_key": {"qa_id": "unit"},
                       "response": json.dumps({"error": {"reset_time": "2099-01-01T00:00:00Z", "limit_type": "daily_quota", "current": 51, "limit": 50}})}
                path = logs / "unit.json"
                path.write_text(json.dumps(log))
                self.assertEqual(latest_quota_block()["qa_id"], "unit")
                log["response"] = json.dumps({"error": {"reset_time": "2000-01-01T00:00:00Z"}})
                path.write_text(json.dumps(log))
                self.assertIsNone(latest_quota_block())
            finally:
                os.chdir(previous)

    def test_compiler_cannot_substitute_qwen_for_teacher(self):
        from arc.algorithm.compiler import compile_task
        with patch("arc.algorithm.compiler.build_once") as build:
            result = compile_task({}, {"qa_id": "unit", "question": "unit", "sources": []})
            self.assertEqual(result["status"], "annotation_error")
            with self.assertRaisesRegex(ValueError, "Grok-4.6"):
                compile_task({}, {"qa_id": "unit", "question": "unit", "sources": [], "requirements": [], "teacher_model": "Qwen/Qwen3.5-4B"})
            build.assert_not_called()

    def test_teacher_credentials_follow_active_codex_provider(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "config.toml").write_text('model_provider="unit"\n[model_providers.unit]\nbase_url="https://unit.invalid/v1"\nwire_api="responses"\n')
            (root / "auth.json").write_text(json.dumps({"OPENAI_API_KEY": "unit-matching-key"}))
            with patch.dict("os.environ", {"OPENAI_API_KEY": "unit-unrelated-key"}):
                self.assertEqual(_credential(root), ("https://unit.invalid/v1", "unit-matching-key"))
            (root / "config.toml").write_text('model_provider="unit"\n[model_providers.unit]\nbase_url="https://unit.invalid/v1"\nwire_api="responses"\nenv_key="UNIT_TEACHER_KEY"\n')
            with patch.dict("os.environ", {"UNIT_TEACHER_KEY": "unit-provider-key"}):
                self.assertEqual(_credential(root), ("https://unit.invalid/v1", "unit-provider-key"))

    def test_paired_comparison_requires_matching_ids(self):
        a = [{"qa_id": "unit1", "category": 1, "score": 1.0, "construction_prompt_tokens": 40}]
        b = [{"qa_id": "unit1", "category": 1, "score": 1.0, "construction_prompt_tokens": 100}]
        result = paired_summary(a, b, samples=100)
        self.assertEqual(result["one_sided_95_upper"], 0.0)
        self.assertAlmostEqual(result["saving_build"], 0.6)
        b[0]["qa_id"] = "different"
        with self.assertRaises(ValueError):
            paired_summary(a, b, samples=100)

    def test_training_resume_matches_uninterrupted_updates(self):
        import torch
        from arc.algorithm.features import FeatureSchema
        from arc.algorithm.selector import train_selector
        torch.set_num_threads(1)
        metadata = {"dimension": 2, "model": "unit-test", "revision": "fixture", "query_prompt_name": "query"}
        row = {"question": "unit question", "sources": [{"id": 1, "source_id": "unit", "text": "unit fact", "speaker": "unit", "token_count": 2,
                "embedding": [1.0, 0.0], "query_vector": [1.0, 0.0], "query_vector_question": "unit question", "embedding_metadata": metadata}],
                "domain": [{"source_ids": [], "cost": 0}, {"source_ids": [1], "cost": 10}],
                "budgets": {"16": {"successful_sets": [[1]]}}}
        schema = FeatureSchema.fit([row], dimension=2)
        with tempfile.TemporaryDirectory() as folder:
            full, resumed = Path(folder) / "full.pt", Path(folder) / "resumed.pt"
            train_selector([row], full, epochs=3, width=8, schema=schema, seed=17, resume=True, input_sha256="fixture")
            def interrupt(**progress):
                if progress["updates"] == 1:
                    raise InterruptedError("unit test interruption after saved optimizer update")
            with self.assertRaises(InterruptedError):
                train_selector([row], resumed, epochs=3, width=8, schema=schema, seed=17, resume=True, input_sha256="fixture", progress_callback=interrupt)
            result = train_selector([row], resumed, epochs=3, width=8, schema=schema, seed=17, resume=True, input_sha256="fixture")
            self.assertEqual(result["updates"], 3)
            a = torch.load(full, weights_only=True)["state_dict"]
            b = torch.load(resumed, weights_only=True)["state_dict"]
            self.assertTrue(all(torch.equal(a[k], b[k]) for k in a))
            skipped = train_selector([row], resumed, epochs=3, width=8, schema=schema, seed=17, resume=True, input_sha256="fixture")
            self.assertTrue(skipped["skipped"])

    def test_missing_outer_brace_only_can_be_recovered(self):
        parsed = _parse_json('{"requirements":[{"id":"r1","text":"fact","packages":[[4]]}]')
        self.assertEqual(parsed["requirements"][0]["packages"], [[4]])
        self.assertEqual(parsed["_syntax_repair"], "appended_missing_root_object_closer")
        with self.assertRaises(ValueError):
            _parse_json('{"requirements":[{"id":"r1","text":"fact","packages":[[4]]')
        self.assertEqual(_parse_json('explanation\n```json\n{"requirements":[]}\n```'), {"requirements": []})

    def test_torn_append_preserves_valid_results_and_tail(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rows.jsonl"
            append_record(path, {"qa_id": "q1", "value": 7})
            with path.open("ab") as out:
                out.write(b'{"qa_id":"q2",')
            index = index_records(path)
            self.assertEqual(set(index), {"q1"})
            self.assertEqual(read_record(index["q1"])["value"], 7)
            self.assertEqual(len(list(Path(folder).glob("*.bin"))), 1)
            append_record(path, {"qa_id": "q2", "value": 8})
            self.assertEqual(len(index_records(path)), 2)

    def test_duplicate_canonical_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rows.jsonl"
            append_record(path, {"qa_id": "q1"})
            append_record(path, {"qa_id": "q1"})
            with self.assertRaisesRegex(ValueError, "duplicate QA"):
                index_records(path)

    def test_successful_call_replayed_and_unknown_call_blocked(self):
        with tempfile.TemporaryDirectory() as folder:
            config = {"paths": {"runs_dir": folder}, "run_id": "unit", "models": {"builder": "test-model"},
                      "claude_code": {"base_url": "http://localhost:9999", "require_credentials": False},
                      "execution": {"resume_calls": True, "work_key": {"qa_id": "unit-q1", "seed": 7}}}
            response = type("Process", (), {"returncode": 0, "stderr": "", "stdout": json.dumps({"result": "unit-test-output", "usage": {"input_tokens": 10, "output_tokens": 2}})})()
            with patch("arc.agent.reader.subprocess.run", return_value=response) as run:
                client = ClaudeCodeClient(config)
                first = client.complete("builder", "unit-test-prompt", 20)
                second = client.complete("builder", "unit-test-prompt", 20)
                self.assertEqual(run.call_count, 1)
                self.assertEqual(first.audit_path, second.audit_path)
                path = Path(first.audit_path)
                body = json.loads(path.read_text()); body["status"] = "in_flight"
                path.write_text(json.dumps(body))
                with self.assertRaisesRegex(CompletionError, "BLOCKED"):
                    client.complete("builder", "unit-test-prompt", 20)
                self.assertEqual(run.call_count, 1)

    def test_merge_prefers_success_and_keeps_failed_keys_retryable(self):
        base = {"sample_id": "s", "qa_id": "q", "arm": "our"}
        failed = {**base, "status": "agent_error"}
        success = {**base, "status": "ok", "agent_usage_complete": True, "construction_usage_complete": True, "errors": [{"reason": "audit_skipped"}]}
        merged = _merge_journal([{"result": failed}, {"result": success}, {"result": failed}])
        self.assertEqual(merged, [success])
        self.assertEqual(construction_usage([])["construction_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
