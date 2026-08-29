import unittest
from r2w.folds import locomo_five_folds
from r2w.costs import ResourceLedger,SelectResources,CommitResources,IndexResources,ReadResources,post_selection_value,enter_value
from r2w.config import R2WConfig
class TestFoldsCosts(unittest.TestCase):
    def test_natural_folds(self):
        ids=[f"dialogue-{i:02d}" for i in range(1,11)]
        f=locomo_five_folds(ids)
        self.assertEqual(f[0]["test"],("dialogue-01","dialogue-02")); self.assertEqual(f[0]["val"],("dialogue-03","dialogue-04")); self.assertEqual(len(f[0]["train"]),6)
    def test_four_resource_value(self):
        cfg=R2WConfig(lambda_commit=1,lambda_index=0,lambda_read=1,H_Q=10)
        l=ResourceLedger(commit=CommitResources(cost=1),index=IndexResources(),read=ReadResources(expected_body_tokens=.2))
        self.assertAlmostEqual(post_selection_value(1,l,cfg),.7)
        self.assertAlmostEqual(enter_value([.7],SelectResources(cost=0),cfg),.7)

    def test_scalarizer_choice_is_hashed_and_applied(self):
        cfg=R2WConfig(lambda_read=1,read_scalar="cost")
        ledger=ResourceLedger(read=ReadResources(expected_body_tokens=99,cost=.25))
        self.assertAlmostEqual(post_selection_value(1,ledger,cfg),.75)
        self.assertNotEqual(cfg.hash(),R2WConfig(lambda_read=1).hash())
