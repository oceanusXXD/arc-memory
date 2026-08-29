import unittest
from r2w.actions import Representation
from r2w.config import R2WConfig
from r2w.retrieval import ParentRRFIndex
from r2w.replay import replay_memory, request_signature
from r2w.audit import audit_replay
from tests.fakes import FakeEmbedder,FakeRunner
class TestReplay(unittest.TestCase):
    def test_paired_ht_and_identity(self):
        cfg=R2WConfig(s_min=1.0,retrieval_k=2)
        raw=Representation("raw","parent-001","subject-1 status value-a.",("subject-1 status value-a.",),{})
        other=Representation("raw","parent-002","subject-2 status value-b.",("subject-2 status value-b.",),{})
        kv=Representation("raw+kv","parent-001","subject-1 status value-a.",("subject-1 status value-a.","subject-1 status value-a"),{})
        idx=ParentRRFIndex([raw,other],FakeEmbedder(),cfg)
        qs=[{"question":"subject-1 status","answer":"value-a","evidence":["parent-001"]},{"question":"subject-2 status","answer":"value-b","evidence":["parent-002"]}]
        rep=replay_memory(idx,"parent-001",{"none":None,"raw":raw,"raw+kv":kv},qs,FakeRunner(),cfg)
        audit=audit_replay(rep)
        self.assertTrue(audit["positive_probability_ok"])
        self.assertTrue(audit["double_contrast_identity_ok"])
    def test_equal_signature_zero(self):
        cfg=R2WConfig(s_min=1.0,retrieval_k=1)
        raw=Representation("raw","parent-001","same",("same",),{})
        idx=ParentRRFIndex([raw],FakeEmbedder(),cfg)
        rep=replay_memory(idx,"parent-001",{"none":None,"raw":raw},[{"question":"field","answer":"token"}],FakeRunner(),cfg)
        self.assertGreaterEqual(rep.labels[1].delta_e,0.0)

    def test_signature_uses_complete_query_contract(self):
        runner=FakeRunner()
        common=(("parent-001",), ("body",), ({"turn_id":"parent-001"},))
        first=request_signature({"question":"field","answer":"value-a"},*common,runner)
        second=request_signature({"question":"field","answer":"value-b"},*common,runner)
        self.assertNotEqual(first,second)
