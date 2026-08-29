import unittest
from r2w import R2WConfig
from r2w.actions import SimpleTokenCounter, build_candidate_bundle, build_representation
from tests.fakes import FakeConstructor

TURN={"dia_id":"turn-001","speaker":"speaker-1","text":"subject-1 status value-a.","timestamp":"2001-02-03","session":1}
class TestActions(unittest.TestCase):
    def test_core_is_six(self):
        self.assertEqual(R2WConfig().actions,("none","raw","sum","raw+kv","raw+event","raw+hq"))
    def test_bundle_is_one_call(self):
        c=FakeConstructor(); cfg=R2WConfig(); b=build_candidate_bundle(TURN,[],c,cfg)
        for a in cfg.storage_actions:
            build_representation(TURN,a,None if a=="raw" else b,cfg,SimpleTokenCounter())
        self.assertEqual(c.calls,1)
    def test_extended_is_opt_in(self):
        self.assertEqual(len(R2WConfig(use_extended_actions=True).actions),11)
    def test_over_budget_rejects_not_truncates(self):
        cfg=R2WConfig(key_budget_tokens=1); c=FakeConstructor(); b=build_candidate_bundle(TURN,[],c,cfg)
        with self.assertRaises(ValueError): build_representation(TURN,"raw+kv",b,cfg,SimpleTokenCounter())
