import unittest
from pathlib import Path

from r2w.pipeline_train import load_training_data


class LoCoMoSchemaTests(unittest.TestCase):
    def test_real_sample_schema_and_evidence_mapping(self):
        path = (
            Path(__file__).resolve().parents[2]
            / "data"
            / "LoCoMo"
            / "data"
            / "locomo10.json"
        )
        conversations = load_training_data(path)
        self.assertEqual(len(conversations), 10)
        self.assertEqual(len(conversations[0]["turns"]), 419)
        self.assertTrue(conversations[0]["conversation_id"])
        self.assertTrue(
            all(
                evidence in conversation["dia_to_index"]
                for conversation in conversations
                for query in conversation["qa"]
                for evidence in query["evidence"]
            )
        )
        answer_types = {
            type(query.get("answer"))
            for conversation in conversations
            for query in conversation["qa"]
        }
        self.assertTrue({str, int, type(None)} <= answer_types)


if __name__ == "__main__":
    unittest.main()
