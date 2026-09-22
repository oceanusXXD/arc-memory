"""Validation gates must distinguish missing evidence from measured failure."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_arc_validation as validation
from arc.algorithm.memory import AuditResult, BuildResult


def compilation(qid, labels):
    return {"qa_id": qid, "split": "train", "domain_complete": True,
            "annotation": {"valid": True}, "successful_sets": labels,
            "budgets": {str(b): {"successful_sets": labels} for b in (1024, 2048, 4096)},
            "evaluations": [{"source_ids": ids, "status": "PASS",
                             "source_audit": {"status": "PASS"},
                             "requirement_audit": {"status": "PASS"}, "usage": []}
                            for ids in labels]}


class ValidationTests(unittest.TestCase):
    def archive(self, rows, limit=1):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.jsonl"
            validation.write_jsonl(path, rows)
            args = argparse.Namespace(compilation=path, output_dir=directory, train_limit=limit)
            with contextlib.redirect_stdout(io.StringIO()):
                return validation.make_train(args)

    def test_limit_counts_usable_rows_and_distinct_sets(self):
        result = self.archive([compilation("no-pass", []), compilation("usable", [[1]])])
        self.assertEqual(result["usable_rows"], 1)
        self.assertEqual(result["total_pass_sets"], 3)
        self.assertEqual(result["total_unique_pass_sets"], 1)
        self.assertEqual(result["rows_with_multiple_distinct_pass_sets"], 0)

    def test_unknown_or_api_error_cannot_supply_training_label(self):
        for failure in ("unknown", "api_error"):
            row = compilation("invalid", [[1]])
            if failure == "unknown":
                row["evaluations"][0]["source_audit"]["status"] = "UNKNOWN"
            else:
                row["evaluations"][0]["usage"] = [{"status": "error"}]
            with self.subTest(failure=failure), self.assertRaisesRegex(ValueError, "compiler audits"):
                self.archive([row])

    def test_oracle_api_error_is_not_a_zero_quality_measurement(self):
        audit = AuditResult("UNKNOWN", ("service unavailable",))
        result = BuildResult(frozenset({1}), "", (), audit, audit, "UNKNOWN",
                             ({"status": "error", "request_count": 1},))
        qa = {"qa_id": "q", "sample_id": "s", "question": "question", "answer": "answer", "category": 1}
        with patch("arc.algorithm.memory.build_once", return_value=result), \
             patch("arc.algorithm.memory.configured_input_cost", return_value=10), \
             patch("arc.agent.runtime.run_agent") as agent:
            measured = validation._gold_oracle_one({}, qa, {"sources": [{"id": 1}]}, [1], "gold_oracle")
        self.assertEqual(measured["status"], "builder_error")
        self.assertIsNone(measured["score"])
        self.assertIsNone(measured["exact_match"])
        agent.assert_not_called()

    def test_report_rejects_matching_but_incomplete_arms(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "evaluation" / "budget-4096"
            root.mkdir(parents=True)
            validation.write_json(root / "evaluation.json", {"qa_ids": ["q1", "q2"]})
            for arm in ("full_memory", "rank_pack", "our"):
                validation.write_jsonl(root / f"{arm}.jsonl", [{"qa_id": "q1", "category": 1}])
            args = argparse.Namespace(output_dir=directory, budget=4096, bootstrap_samples=10)
            with self.assertRaisesRegex(ValueError, "incomplete or duplicate"):
                validation.report(args)


if __name__ == "__main__":
    unittest.main()
