import unittest
from r2w.actions import Representation
from r2w.config import R2WConfig
from r2w.retrieval import ParentRRFIndex
from tests.fakes import FakeEmbedder
class TestRetrieval(unittest.TestCase):
    def test_parent_occupies_one_slot(self):
        r1=Representation("raw+kv","p1","body1",("apple","apple fruit","apple red"),{})
        r2=Representation("raw","p2","body2",("banana",),{})
        idx=ParentRRFIndex([r1,r2],FakeEmbedder(),R2WConfig(retrieval_k=2))
        out=idx.retrieve("apple",2)
        self.assertEqual(len({x.parent_id for x in out}),len(out))
        self.assertLessEqual(len(out),2)
    def test_parent_first_changes_multi_key_advantage(self):
        r1=Representation("raw+kv","p1","b1",("alpha","alpha","alpha"),{})
        r2=Representation("raw","p2","b2",("alpha beta",),{})
        idx=ParentRRFIndex([r1,r2],FakeEmbedder(),R2WConfig())
        out=idx.retrieve("alpha",2)
        self.assertEqual(set(x.parent_id for x in out),{"p1","p2"})
