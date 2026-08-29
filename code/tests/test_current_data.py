import json
from pathlib import Path
import tempfile
import unittest

from r2w.data import load_benchmark_conversations


class TestBenchmarkAdapters(unittest.TestCase):
    def test_locomo_sessions_dates_and_evidence_are_normalised(self):
        records = [{
            "sample_id": "record-001",
            "conversation": {
                "speaker_a": "speaker-1", "speaker_b": "speaker-2",
                "session_1_date_time": "1:56 pm on 3 February, 2001",
                "session_1": [{"dia_id": "segment-001", "speaker": "speaker-1", "text": "attribute-a: value-a."}],
            },
            "qa": [{"question": "What is attribute-a?", "answer": "value-a", "evidence": ["segment-001"]}],
        }]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "locomo.json"
            path.write_text(json.dumps(records), encoding="utf-8")
            first = load_benchmark_conversations(path)[0]
        self.assertEqual(first["turns"][0]["timestamp"], "2001-02-03 13:56")
        self.assertTrue(all(evidence in first["dia_to_index"] for query in first["qa"] for evidence in query["evidence"]))

    def test_longmemeval_sessions_remain_atomic_parent_memories(self):
        records = [{
            "question_id": "record-001", "question": "What is attribute-a?", "answer": "value-a",
            "question_type": "single-session", "question_date": "2001/02/03 (Sat) 23:40",
            "haystack_session_ids": ["segment-001", "segment-002"],
            "haystack_dates": ["2001/02/01 (Thu) 02:21", "2001/02/02 (Fri) 09:00"],
            "haystack_sessions": [[{"role": "user", "content": "attribute-a: value-a."}], [{"role": "assistant", "content": "acknowledged."}]],
            "answer_session_ids": ["segment-001"],
        }]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "longmemeval.json"
            path.write_text(json.dumps(records), encoding="utf-8")
            conversation = load_benchmark_conversations(path)[0]
        self.assertEqual([turn["dia_id"] for turn in conversation["turns"]], ["segment-001", "segment-002"])
        self.assertEqual(conversation["turns"][0]["text"], "user: attribute-a: value-a.")
        self.assertEqual(conversation["qa"][0]["evidence"], ["segment-001"])


if __name__ == "__main__":
    unittest.main()
