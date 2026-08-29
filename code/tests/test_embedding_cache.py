import tempfile
import unittest
from pathlib import Path

import numpy as np

from r2w.config import default_config
from r2w.embedding import EmbeddingBackend


class EmbeddingCacheTests(unittest.TestCase):
    def test_duplicate_texts_share_one_backend_request_and_persist(self):
        cfg = default_config(embedding_dim=3)
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "embeddings.npz"
            backend = EmbeddingBackend(cfg)
            backend.cache_path = cache_path
            requested_batches = []

            def fake_api_encode(texts):
                requested_batches.append(texts)
                return np.asarray(
                    [[3.0, 4.0, 0.0], [0.0, 0.0, 2.0]], dtype=np.float32
                )

            backend._api_encode = fake_api_encode
            vectors = backend.encode(["first", "first", "second"])

            self.assertEqual(requested_batches, [["first", "second"]])
            np.testing.assert_array_equal(vectors[0], vectors[1])
            self.assertTrue(cache_path.exists())

            reloaded = EmbeddingBackend(cfg)
            reloaded.cache_path = cache_path
            reloaded._load_text_cache(cache_path)
            np.testing.assert_allclose(reloaded.encode(["second", "first"]), vectors[[2, 0]])


if __name__ == "__main__":
    unittest.main()
