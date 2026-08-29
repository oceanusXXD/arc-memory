import copy
import unittest

from r2w.baselines import (
    evaluate_fixed_baseline,
    evaluate_full_context_baseline,
    evaluate_none_baseline,
)
from r2w.config import default_config

from .fakes import FakeEmbedder, FakeLLM, FakeQG, FakeReader, small_conversation


class BaselineTests(unittest.TestCase):
    def test_raw_is_naive_rag_and_costs_are_separate(self):
        cfg = default_config(embedding_dim=8, retrieval_k=2, teacher_rank_limit=3)
        conversations = [copy.deepcopy(small_conversation())]
        raw = evaluate_fixed_baseline(
            conversations, "raw", FakeEmbedder(8), cfg, FakeReader()
        )
        summary = evaluate_fixed_baseline(
            conversations, "sum", FakeEmbedder(8), cfg, FakeReader()
        )
        none = evaluate_none_baseline(conversations, FakeReader())
        full = evaluate_full_context_baseline(conversations, cfg, FakeReader())
        self.assertEqual(raw.name, "always_raw_naive_rag")
        self.assertEqual(raw.mean_normalized_exact_match, 1.0)
        self.assertEqual(raw.total_write_words, 0.0)
        self.assertGreater(raw.total_index_words, 0.0)
        self.assertGreater(summary.total_write_words, 0.0)
        self.assertEqual(none.total_index_words, 0.0)
        self.assertEqual(full.mean_evidence_recall_at_k, 1.0)

    def test_structured_fixed_baseline_uses_shared_constructor(self):
        cfg = default_config(embedding_dim=8, retrieval_k=2, teacher_rank_limit=3)
        result = evaluate_fixed_baseline(
            [small_conversation()],
            "raw+kv",
            FakeEmbedder(8),
            cfg,
            FakeReader(),
            constructor=FakeLLM(),
            qg=FakeQG(),
        )
        self.assertEqual(result.fixed_action, "raw+kv")
        self.assertGreaterEqual(result.qa_count, 1)


if __name__ == "__main__":
    unittest.main()
