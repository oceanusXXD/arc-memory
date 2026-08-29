import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from r2w.config import default_config
from r2w.model import stage2_matrix
from r2w.pipeline_query_oracle_gbm import (
    FEATURE_NAMES,
    _base_features,
    _costs_from_drafts,
    _draft_statistics,
    _load_manifest,
    _select_examples,
)
from r2w.pipeline_train import load_training_data
from r2w.representations import STRUCT_ACTIONS


class QueryOracleGBMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        code_root = Path(__file__).resolve().parents[1]
        repository_root = code_root.parent
        cls.manifest_path = (
            code_root / "oracle_manifests" / "locomo_query_value_oracle.json"
        )
        cls.data_path = repository_root / "data" / "LoCoMo" / "data" / "locomo10.json"

    def test_real_manifest_maps_to_query_and_conversation_holdout(self):
        manifest = _load_manifest(self.manifest_path)
        items = _select_examples(load_training_data(self.data_path), manifest)
        train = [item for item in items if item["label"]["split"] == "train"]
        test = [item for item in items if item["label"]["split"] == "test"]
        train_groups = {item["label"]["conversation_id"] for item in train}
        test_groups = {item["label"]["conversation_id"] for item in test}
        self.assertEqual((len(train), len(test)), (42, 18))
        self.assertEqual((len(train_groups), len(test_groups)), (7, 3))
        self.assertFalse(train_groups & test_groups)
        self.assertTrue(all(len(item["oracle_values"]) == len(STRUCT_ACTIONS) for item in items))

    def test_features_ignore_answer_category_and_oracle_fields(self):
        manifest = _load_manifest(self.manifest_path)
        item = _select_examples(load_training_data(self.data_path), manifest)[0]
        expected = _base_features(item)
        changed = copy.deepcopy(item)
        changed["query"]["answer"] = "SHOULD NEVER ENTER FEATURES"
        changed["query"]["category"] = "SHOULD NEVER ENTER FEATURES"
        changed["label"]["profile"] = "SHOULD NEVER ENTER FEATURES"
        changed["oracle_values"] = np.full(len(STRUCT_ACTIONS), -999.0, dtype=np.float32)
        np.testing.assert_array_equal(_base_features(changed), expected)
        self.assertEqual(expected.shape, (len(FEATURE_NAMES),))

    def test_offline_drafts_expand_through_production_stage2_matrix(self):
        cfg = default_config()
        manifest = _load_manifest(self.manifest_path)
        item = _select_examples(load_training_data(self.data_path), manifest)[0]
        base = _base_features(item)[None, :]
        drafts = _draft_statistics(item, cfg)[None, :, :]
        matrix = stage2_matrix(base, drafts)
        self.assertEqual(drafts.shape, (1, len(STRUCT_ACTIONS), 5))
        self.assertEqual(matrix.shape[0], len(STRUCT_ACTIONS))
        self.assertEqual(matrix.shape[1], len(FEATURE_NAMES) + 16 + 5)
        self.assertEqual(float(drafts[0, 0, 4]), 0.0)
        self.assertEqual(float(drafts[0, 5, 4]), 1.0)

        costs = _costs_from_drafts(drafts[0])
        raw_payload = float(drafts[0, 0, 0])
        self.assertEqual(tuple(costs[0]), (0.0, raw_payload, raw_payload))
        nonraw_total = float(drafts[0, 1, 0] + drafts[0, 1, 1])
        self.assertEqual(
            tuple(costs[1]),
            (nonraw_total, nonraw_total, float(drafts[0, 1, 0])),
        )

    def test_manifest_rejects_incomplete_action_values(self):
        manifest = _load_manifest(self.manifest_path)
        profile = next(iter(manifest["value_profiles"].values()))
        del profile["values"][STRUCT_ACTIONS[-1]]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "给齐十个 action value"):
                _load_manifest(path)

    def test_manifest_rejects_tied_or_nonfinite_values_and_bad_fields(self):
        cases = []
        tied = _load_manifest(self.manifest_path)
        profile = next(iter(tied["value_profiles"].values()))
        maximum = max(profile["values"].values())
        profile["values"][STRUCT_ACTIONS[0]] = maximum
        cases.append((tied, "唯一最佳 action"))

        nonfinite = _load_manifest(self.manifest_path)
        profile = next(iter(nonfinite["value_profiles"].values()))
        profile["values"][STRUCT_ACTIONS[0]] = float("nan")
        cases.append((nonfinite, "必须是有限数"))

        bad_field = _load_manifest(self.manifest_path)
        bad_field["examples"][0]["question"] = 42
        cases.append((bad_field, "question 必须是非空字符串"))

        with tempfile.TemporaryDirectory() as directory:
            for index, (manifest, message) in enumerate(cases):
                with self.subTest(message=message):
                    path = Path(directory) / f"bad-{index}.json"
                    path.write_text(json.dumps(manifest), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, message):
                        _load_manifest(path)


if __name__ == "__main__":
    unittest.main()
