import unittest

import numpy as np

from r2w.config import default_config
from r2w.teacher import ReaderAnswer, build_effect_targets, run_swap_measure, run_uniform_measure

from .fakes import FakeEmbedder, FakeLLM, FakeQG, FakeReader, small_conversation


class MeasurementFormulaTests(unittest.TestCase):
    def test_p_vs_none_uses_endpoint_reader_and_signed_externality(self):
        cfg = default_config(embedding_dim=8, retrieval_k=3, teacher_rank_limit=3, interference_query_count=1, empirical_bayes_n0=0.0)
        conversation = small_conversation()
        class TargetReader:
            def answer(self, question, payloads, current_date):
                return ReaderAnswer("Boston" if any("moved to Boston" in value for value in payloads) else "unknown", ())

        reader = TargetReader()
        uniform = run_uniform_measure(conversation, FakeEmbedder(8), FakeLLM(), FakeQG(), cfg, reader)
        swap = run_swap_measure(conversation, FakeEmbedder(8), cfg, uniform, reader, np.ones((3, 10), dtype=bool))
        targets = build_effect_targets(conversation, uniform, cfg, swap)
        raw = 0
        self.assertTrue(np.all(targets.measured))
        self.assertGreater(targets.effects[0, raw], 0.0)
        self.assertTrue(np.allclose(targets.form_effects[:, raw], 0.0))
        self.assertEqual(targets.costs.shape, (3, 10, 3))
        self.assertTrue(np.all((targets.hit_rates >= 0.0) & (targets.hit_rates <= 1.0)))
        # 不 clamp：该量可以正、零或负，但必须由未相关层端到端差分直接生成。
        self.assertTrue(np.isfinite(targets.psi).all())
        self.assertTrue(np.all(targets.n_rel[:, raw] >= 0))


if __name__ == "__main__":
    unittest.main()
