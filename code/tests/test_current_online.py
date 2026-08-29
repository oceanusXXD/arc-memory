import unittest
from r2w.config import R2WConfig
from r2w.actions import SimpleTokenCounter,Representation
from r2w.retrieval import ParentRRFIndex
from r2w.online import OnlineWriter
from tests.fakes import FakeEmbedder,FakeConstructor,FakeSupport,FakeQA,FakeValueModel,FakeResources
TURN={"dia_id":"turn-001","speaker":"speaker-1","text":"subject-1 status value-a.","timestamp":"2001-02-03","session":1}
class TestOnline(unittest.TestCase):
    def test_online_uses_one_bundle_and_selects_safe(self):
        cfg=R2WConfig(kappa_e=0,kappa_f=0,key_budget_tokens=100)
        emb=FakeEmbedder(); base=Representation("raw","parent-000","subject-2 status value-b.",("subject-2 status value-b.",),{})
        idx=ParentRRFIndex([base],emb,cfg); c=FakeConstructor()
        w=OnlineWriter(cfg,emb,SimpleTokenCounter(),c,FakeSupport(),FakeQA(),FakeValueModel(),FakeResources())
        d=w.decide(TURN,[],idx)
        self.assertEqual(c.calls,1); self.assertEqual(d.action,"raw+kv")

    def test_online_uses_predicted_read_cost(self):
        class ExpensiveReadModel(FakeValueModel):
            def read_cost(self,action,features): return .5 if action=="raw+kv" else 0.
        cfg=R2WConfig(kappa_e=0,kappa_f=0,key_budget_tokens=100,lambda_read=1)
        emb=FakeEmbedder(); base=Representation("raw","parent-000","subject-2 status value-b.",("subject-2 status value-b.",),{})
        idx=ParentRRFIndex([base],emb,cfg)
        w=OnlineWriter(cfg,emb,SimpleTokenCounter(),FakeConstructor(),FakeSupport(),FakeQA(),ExpensiveReadModel(),FakeResources())
        self.assertEqual(w.decide(TURN,[],idx).action,"raw")

    def test_candidate_failure_keeps_raw_none_selection(self):
        class FailingConstructor:
            def json(self,prompt): raise RuntimeError("backend unavailable")
        cfg=R2WConfig(kappa_e=0,kappa_f=0)
        emb=FakeEmbedder(); base=Representation("raw","parent-000","subject-2 status value-b.",("subject-2 status value-b.",),{})
        idx=ParentRRFIndex([base],emb,cfg)
        d=OnlineWriter(cfg,emb,SimpleTokenCounter(),FailingConstructor(),FakeSupport(),FakeQA(),FakeValueModel(),FakeResources()).decide(TURN,[],idx)
        self.assertEqual(d.action,"raw")
        self.assertEqual(d.feasible,("none","raw"))
        self.assertIn("fallbacks",d.predictions)

    def test_value_model_failure_uses_versioned_fallback(self):
        class BrokenValueModel:
            def cheap(self,action,features): raise RuntimeError("predictor unavailable")
        cfg=R2WConfig(value_model_fallback="none")
        emb=FakeEmbedder(); base=Representation("raw","parent-000","subject-2 status value-b.",("subject-2 status value-b.",),{})
        idx=ParentRRFIndex([base],emb,cfg)
        d=OnlineWriter(cfg,emb,SimpleTokenCounter(),FakeConstructor(),FakeSupport(),FakeQA(),BrokenValueModel(),FakeResources()).decide(TURN,[],idx)
        self.assertEqual(d.action,"none")
        self.assertTrue(d.predictions["fallbacks"][0].startswith("value_model_unavailable"))
