import unittest

from r2w.pipeline_train import load_training_data
from r2w.teacher import reader_current_date


class LongMemEvalSchemaTests(unittest.TestCase):
    def test_session_records_map_to_r2w_turns_without_query_feature_fields(self):
        records = [
            {
                "question_id": "question-1",
                "question": "Where did Alice move?",
                "answer": "Boston",
                "question_type": "single-session-user",
                "question_date": "2023/05/30 (Tue) 23:40",
                "haystack_session_ids": ["session-a", "session-b"],
                "haystack_dates": [
                    "2023/05/20 (Sat) 02:21",
                    "2023/05/21 (Sun) 09:00",
                ],
                "haystack_sessions": [
                    [{"role": "user", "content": "I moved to Boston."}],
                    [{"role": "assistant", "content": "That is nice."}],
                ],
                "answer_session_ids": ["session-a"],
            }
        ]
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "longmemeval.json"
            path.write_text(json.dumps(records), encoding="utf-8")
            conversations = load_training_data(path)
        self.assertEqual(len(conversations), 1)
        conversation = conversations[0]
        self.assertEqual(conversation["conversation_id"], "question-1")
        self.assertEqual([turn["dia_id"] for turn in conversation["turns"]], ["session-a", "session-b"])
        self.assertEqual(conversation["turns"][0]["text"], "user: I moved to Boston.")
        self.assertEqual(conversation["turns"][0]["timestamp"], "2023-05-20 02:21")
        self.assertEqual(conversation["qa"][0]["evidence"], ["session-a"])
        self.assertEqual(reader_current_date(conversation, conversation["qa"][0]), "2023-05-30 23:40")


if __name__ == "__main__":
    unittest.main()
