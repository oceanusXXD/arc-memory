import unittest

from r2w.config import default_config
from r2w.online import ShadowAwareMemoryWriter
from r2w.representations import build_representation

from .fakes import FakeEmbedder, FakeLLM, small_conversation


class OnlineStorageTests(unittest.TestCase):
    def test_none_archives_and_nonraw_keeps_raw_shadow(self):
        cfg = default_config(embedding_dim=8, none_shadow_probability=1.0)
        writer = ShadowAwareMemoryWriter(cfg, FakeEmbedder(8))
        turn = small_conversation()["turns"][0]
        writer.write_decision("none", None, turn, {})
        self.assertIn("D1:1", writer.archive)
        self.assertGreater(len(writer.none_shadow.units), 0)
        value = build_representation(turn, "raw+kv", FakeLLM(), cfg)
        writer.write_decision("raw+kv", value, turn | {"dia_id": "D1:9"}, {})
        self.assertGreater(len(writer.hot.units), 0)
        self.assertGreater(len(writer.raw_shadow.units), 0)


if __name__ == "__main__":
    unittest.main()
