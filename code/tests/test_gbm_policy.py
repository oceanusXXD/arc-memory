import unittest

import numpy as np

from r2w.calibration import CostStatistics, DecisionPolicy
from r2w.config import default_config
from r2w.model import stage2_matrix, train_gbm_ensemble


class GBMAndPolicyTests(unittest.TestCase):
    def test_five_fold_pooled_ensemble_predicts_per_arm(self):
        cfg = default_config(gbm_estimators=4, gbm_min_child_samples=1)
        count, arms = 15, 10
        rng = np.random.default_rng(3)
        stage1 = rng.normal(size=(count, 4)).astype(np.float32)
        base = rng.normal(size=(count, 7)).astype(np.float32)
        draft = rng.normal(size=(count, arms, 5)).astype(np.float32)
        effects = np.tile(np.linspace(0.0, 0.9, arms), (count, 1)).astype(np.float32)
        hits = np.full((count, arms), 0.4, dtype=np.float32)
        groups = np.repeat(np.arange(5), 3)
        model = train_gbm_ensemble(cfg, stage1, effects.max(axis=1), base, draft, effects, hits, np.ones_like(effects, dtype=bool), np.zeros_like(effects), groups)
        mean, sigma = model.predict_admission(stage1[:2])
        predicted_effects, predicted_hits, predicted_sigma = model.predict_arms(base[:2], draft[:2])
        self.assertEqual(len(model.effect_models), 5)
        self.assertEqual(mean.shape, (2,))
        self.assertEqual(predicted_effects.shape, (2, arms))
        self.assertTrue(np.all((predicted_hits >= 0.0) & (predicted_hits <= 1.0)))
        self.assertEqual(stage2_matrix(base[:1], draft[:1]).shape[0], arms)

    def test_gate_and_raw_fallback_follow_two_line_rule(self):
        policy = DecisionPolicy(0.0, 0.0, 0.0, 1.0, 1.0, 0.95, CostStatistics(0.0, 0.0, 0.0), "x")
        costs = np.zeros((10, 3), dtype=np.float32)
        none = policy.decide(-0.1, 0.0, np.zeros(10), np.zeros(10), np.zeros(10), costs)
        self.assertEqual(none.action_index, "none")
        effects = np.zeros(10, dtype=np.float32)
        effects[3] = 2.0
        feasible = np.zeros(10, dtype=bool)
        feasible[0] = True
        raw = policy.decide(1.0, 0.0, effects, np.zeros(10), np.zeros(10), costs, feasible)
        self.assertEqual(raw.action_index, 0)


if __name__ == "__main__":
    unittest.main()
