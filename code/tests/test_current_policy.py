import unittest
from r2w.config import R2WConfig
from r2w.policy import Estimate,Stage2Estimate,early_reject,safe_stage2
class TestPolicy(unittest.TestCase):
    def test_early_reject(self):
        cfg=R2WConfig(kappa_e=0,lambda_select=0)
        self.assertTrue(early_reject({a:Estimate(-.1,0) for a in cfg.storage_actions},0,cfg))
    def test_stage2_can_return_none(self):
        cfg=R2WConfig(kappa_e=0,kappa_f=0)
        e={"raw":Stage2Estimate(-.1,0),"sum":Stage2Estimate(-.2,0,0)}
        self.assertEqual(safe_stage2(e,{"none","raw","sum"},cfg),"none")
    def test_raw_protection(self):
        cfg=R2WConfig(kappa_e=0,kappa_f=1)
        e={"raw":Stage2Estimate(.2,0),"sum":Stage2Estimate(.25,0,.1),"raw+kv":Stage2Estimate(.35,0,.01)}
        self.assertEqual(safe_stage2(e,{"none","raw","sum","raw+kv"},cfg),"raw+kv")
