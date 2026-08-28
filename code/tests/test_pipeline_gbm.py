import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from r2w.config import default_config
from r2w.pipeline_infer import decide_turn, load_artifacts
from r2w.pipeline_calibrate import recalibrate
from r2w.pipeline_train import train_pipeline

from .fakes import FakeEmbedder, FakeLLM, FakeQG, FakeReader, small_conversation


class FakeNER:
    def counts(self, text):
        return (0, 0, 0)


class PipelineArtifactTests(unittest.TestCase):
    def test_train_load_and_decide_with_fake_components(self):
        cfg = default_config(
            embedding_dim=8,
            retrieval_k=2,
            teacher_rank_limit=3,
            interference_query_count=1,
            gbm_estimators=3,
            gbm_min_child_samples=1,
            empirical_bayes_n0=0.0,
        )
        conversations = []
        for index in range(10):
            value = copy.deepcopy(small_conversation())
            value["conversation_id"] = f"fake-{index}"
            conversations.append(value)
        embedder, constructor, qg, reader = FakeEmbedder(8), FakeLLM(), FakeQG(), FakeReader()
        with tempfile.TemporaryDirectory() as directory:
            summary = train_pipeline(cfg, conversations, directory, embedder=embedder, constructor=constructor, qg=qg, reader=reader, ner=FakeNER(), llm=constructor)
            self.assertEqual(summary["train_conversations"], 6)
            loaded = load_artifacts(directory, embedder=embedder, ner=FakeNER())
            loaded_cfg, _, loaded_features, loaded_model, loaded_cases, loaded_policy = loaded
            self.assertTrue(Path(directory, "model.joblib").exists())
            result = decide_turn(loaded_cfg, loaded_features, loaded_model, loaded_cases, loaded_policy, conversations[0]["turns"][0], [], "Alice", "Bob", qg=qg, constructor=constructor, certificate_checker=lambda representation, turn: 1.0)
            self.assertIn(result["action"], ("none", *["raw", "raw+kv", "raw+event", "raw+graph", "raw+hq", "sum", "sum+kv", "sum+event", "sum+graph", "sum+hq"]))
            self.assertIn("gate_value", result)
            recalibrated = recalibrate(directory, 0.0, 0.02, 0.0)
            self.assertEqual(recalibrated["config_hash"], loaded_cfg.hash())

    def test_gate_skips_all_nonraw_construction(self):
        cfg = default_config(
            embedding_dim=8,
            retrieval_k=2,
            teacher_rank_limit=3,
            interference_query_count=1,
            gbm_estimators=3,
            gbm_min_child_samples=1,
            empirical_bayes_n0=0.0,
        )
        conversations = []
        for index in range(10):
            value = copy.deepcopy(small_conversation())
            value["conversation_id"] = f"fake-{index}"
            conversations.append(value)
        embedder, constructor, qg, reader = FakeEmbedder(8), FakeLLM(), FakeQG(), FakeReader()
        with tempfile.TemporaryDirectory() as directory:
            train_pipeline(cfg, conversations, directory, embedder=embedder, constructor=constructor, qg=qg, reader=reader, ner=FakeNER(), llm=constructor)
            loaded_cfg, _, features, model, cases, policy = load_artifacts(directory, embedder=embedder, ner=FakeNER())
            model.predict_admission = lambda values: (
                np.asarray([-1.0], dtype=np.float32),
                np.asarray([0.0], dtype=np.float32),
            )

            class ExplodingConstructor:
                def json(self, prompt):
                    raise AssertionError("第二关未被放行时不能构建结构化草稿")

            class ExplodingQG:
                def generate(self, previous, text):
                    raise AssertionError("第二关未被放行时不能加载 QG")

            result = decide_turn(
                loaded_cfg,
                features,
                model,
                cases,
                policy,
                conversations[0]["turns"][0],
                [],
                "Alice",
                "Bob",
                qg=ExplodingQG(),
                constructor=ExplodingConstructor(),
            )
            self.assertEqual(result["action"], "none")
            self.assertTrue(result["stage2_skipped"])


if __name__ == "__main__":
    unittest.main()
