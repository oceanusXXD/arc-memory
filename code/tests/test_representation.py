import unittest

from r2w.config import default_config
from r2w.representations import STRUCT_ACTIONS, build_representation, compress_summary
from r2w.text import word_count

from .fakes import FakeLLM, FakeQG, small_conversation


class RepresentationTests(unittest.TestCase):
    def test_all_arms_share_v3_contract(self):
        cfg = default_config(embedding_dim=8)
        turn = small_conversation()["turns"][0]
        turn["_r2w_hq"] = FakeQG().generate("", turn["text"])
        builder = FakeLLM()
        for action in STRUCT_ACTIONS:
            representation = build_representation(turn, action, builder, cfg)
            self.assertEqual(representation.protocol_version, "r2w-repr-v3")
            self.assertIn("turn=D1:1", representation.payload)
            self.assertLessEqual(sum(word_count(value) for value in representation.keys), cfg.key_budget_words)

    def test_summary_is_deterministic_and_bounded(self):
        text = "One useful fact. Another useful fact."
        self.assertEqual(compress_summary(text, 2), "")
        self.assertLessEqual(word_count(compress_summary(text, 3)), 3)


if __name__ == "__main__":
    unittest.main()
